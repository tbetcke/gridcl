# Orientation

Orientation answers two different questions. The first is about surfaces and curves: which of
the two unit normals at a point is "the" normal, and how to choose it consistently over the
whole mesh. The second is about the edges and faces inside a mesh: two cells that share an edge
may list its vertices in opposite orders, and a finite element code needs to know how each cell
sees each shared entity. tetraloma answers the first with orientation flags and the second with
permutation bits.

## Manifold orientation

A surface in 3D or a curve in 2D has two unit normals at each point, $n$ and $-n$. The
[normal](geometric-factors.md#normals) that tetraloma computes from $J$ follows the order of each
cell's vertices, and a mesh file need not list neighbouring cells in matching orders. Each cell
therefore carries a flag, bit 0 of its word in the cell block's `flags`: where it is set, the
normal is reversed. Orienting a mesh means choosing these flags so that the normals form one
continuous field.

Two neighbouring cells are consistently oriented when they run through their shared facet in
opposite directions, like the faces of a closed polyhedron whose normals all point out. For two
triangles that share an edge, each induces a direction on the edge from its own boundary cycle;
the directions must be opposite. For two intervals that share a point, one must end where the
other starts.

`orient(rule)` on a codimension-1 [`Mesh`](../api/tetraloma/struct.Mesh.html) computes the
flags in four steps:

1. **Manifold check.** Every facet (an edge of a surface, a point of a curve) must belong to
   at most two cells. Otherwise the mesh is not a manifold, and `orient` returns
   `GridError::NonManifold`.
2. **Parity of each interior facet.** For each facet with two cells $a$ and $b$, the parity is
   0 if the two cells run through it in opposite directions and 1 otherwise. The signs
   $s_a, s_b \in \{-1, +1\}$ of the two cells must then satisfy
   $s_a s_b = (-1)^{\text{parity}}$.
3. **Propagation.** Each cell starts with its own origin id as a label and repeatedly adopts
   the smallest label among its neighbours, carrying the parity along. At the end, the cells
   of each connected component share one label, and each cell knows its sign relative to the
   component's first cell. Between rounds, ranks exchange the labels of cells on both sides of
   every facet on a process boundary, so the result is the same on any number of ranks. If one
   component needs two different signs for the same cell, as on a Möbius strip, the surface is
   not orientable and `orient` returns `GridError::NonOrientable`.
4. **Direction of each component.** The
   [`OrientationRule`](../api/tetraloma/enum.OrientationRule.html) picks which of the two
   consistent orientations each component gets:
   - `Outward`: for a closed component, the one with positive enclosed signed measure
     $\frac{1}{g}\oint x \cdot n$, so the normals point out of the enclosed region. A component
     with boundary encloses nothing, and `Outward` returns `GridError::NeedsReferencePoint` for
     it.
   - `AwayFrom(p)`: the one with $\int (x - p) \cdot n \ge 0$, normals pointing away from $p$.
   - `Towards(p)`: the one with $\int (x - p) \cdot n \le 0$, normals pointing towards $p$.

   The integrals use Gauss rules of degree $3k$ for geometry of degree $k$, and they are
   finished in f64. If an integral is exactly 0, the component keeps the orientation from
   step 3.

`orient` ignores the flags the cells had before and writes new ones. It creates the facets and
the facet-to-cell connectivity it needs. The labels and parities depend only on origin ids, so
the consistent orientation of each component does not depend on the partition.

Boundary meshes need no propagation. `boundary()` and `facet_submesh` orient each facet cell
so that its normal agrees with the outward [facet normal](geometric-factors.md#facet-normals)
of its parent cell. The tutorial [Tags and submeshes](../tutorials/tags-submeshes.md#orientation)
shows both ways.

The example below takes the triangle $(1, 0, 0)$, $(3, 0, 0)$, $(2, 2, 2)$, whose computed
normal is $(0, -1, 1)/\sqrt 2$, and orients it relative to the point $p = (0, 4, -4)$. The
normal points away from $p$, so `AwayFrom(p)` leaves the flag clear and `Towards(p)` sets it.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mut mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;
let p = [0.0, 4.0, -4.0];

mesh.orient(OrientationRule::AwayFrom(p))?;
assert_eq!(mesh.cell_blocks()[0].flags.to_host()?, [0]);

mesh.orient(OrientationRule::Towards(p))?;
assert_eq!(mesh.cell_blocks()[0].flags.to_host()?, [1]);

// With the flag set, the normal is reversed: s = -1.
let rule = QuadRule::for_grid(&mesh, 1)?;
let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
let r = 0.5f64.sqrt();
assert!(n[0][0].abs() < 1e-12);
assert!((n[0][1] - r).abs() < 1e-12 && (n[0][2] + r).abs() < 1e-12);
# Ok(()) }
```

## Permutation bits

Inside a mesh, an edge shared by several cells has one stored direction, but each cell lists
the edge's vertices in its own local order. The same holds for faces, which can also be rotated
or reflected relative to each other. A finite element code that places degrees of freedom on
edges and faces must match them between cells, and needs, for every cell, the relation between
the cell's local view of each entity and the entity's stored orientation.

tetraloma fixes the stored orientation by origin ids, the low-to-high rule of ndelement, Basix
and DOLFINx:

- an edge runs from its vertex with the lower origin id to the one with the higher;
- a face starts at its vertex with the lowest origin id and continues towards the lower of
  that vertex's two neighbours on the face.

Every rank that holds an entity sees the same origin ids, so all ranks agree on the
orientation, and it does not change when you redistribute the mesh. tetraloma also stores the
vertex list of each edge and face in this orientation, so a device kernel can orient an entity
without a lookup.

For every cell of dimension 2 or 3, tetraloma computes one 32-bit word of permutation bits,
with ndelement's layout. From the least significant bit:

- one bit per edge, in edge order: set when the edge's first local vertex has the higher
  origin id, that is, when the cell's view runs against the stored direction;
- then three bits per face, in face order (three-dimensional cells only): the low two hold the
  rotation, the position of the face's lowest vertex in the cell's cycle of the face; the third
  is set when the face is reflected.

Intervals have no bits. The bits are computed from the origin ids of each cell's vertices when
the mesh is built, independently of entity creation, and `permutation_info(b)` returns them for
cell block `b`. [Orientation bits](../elements/orientation-bits.md#edge-bits) gives the layout
of every cell type and a worked example.

In the example below a triangle's vertices have the origin ids 9, 30 and 4. Edge 0 runs from 9
to 30, along the stored direction, so bit 0 is clear. Edge 1 runs from 9 to 4 and edge 2 from 30
to 4, both against it, so bits 1 and 2 are set, and the word is $110_2 = 6$. The function
[`orientation_bits`](../api/tetraloma_refcell/fn.orientation_bits.html) of tetraloma-refcell
computes the same word on the host.

```rust
use tetraloma::prelude::*;
use tetraloma_refcell::orientation_bits;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

assert_eq!(orientation_bits(CellType::Triangle, &[9, 30, 4]), 0b110);

// The same triangle in a mesh, with the node origin ids 9, 30 and 4.
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], Some(&[9, 30, 4]))
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;
let bits = mesh.permutation_info(0).expect("triangles have edge bits").to_host()?;
assert_eq!(bits, [0b110]);

// Face 0 of a hexahedron with the origin ids 17, 4, 9, 12 in its tensor slots: the lowest
// vertex, 4, sits at position 1 of the cycle 17-4-12-9, and its successor 12 is lower
// than its predecessor 17, so the rotation is 1 and the reflection bit is clear.
let hex = orientation_bits(CellType::Hexahedron, &[17, 4, 9, 12, 100, 101, 102, 103]);
assert_eq!((hex >> 12) & 0b111, 0b001);
# Ok(()) }
```
