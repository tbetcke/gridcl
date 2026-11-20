# Orientation bits

Two cells that share an edge or a face usually see it differently. One cell's edge 2 may run
from vertex A to vertex B, while the neighbour's edge 0 runs from B to A. Anything stored on the
shared entity, such as the nodes inside an edge or the degrees of freedom of a finite element,
must still mean the same thing to both cells. tetraloma solves this as ndelement, Basix and
DOLFINx do: every edge and face has one reference orientation, and every cell stores, in a few
bits, how its own local view of each edge and face relates to that reference. The chapter
[Orientation](../concepts/orientation.md#permutation-bits) explains the idea; this chapter gives
the exact rules and the bit layout.

The reference orientation comes from the **origin ids** of the vertices: the ids you give the
mesh builder, or the node tags of a Gmsh file. Origin ids do not depend on how the mesh is
partitioned, so every process that holds a shared entity agrees on its orientation, and the
orientation does not change when the mesh is redistributed. Never compute orientations from
local or global vertex numbers, which do change.

tetraloma-refcell computes the bits of one cell with
[`orientation_bits`](../api/tetraloma_refcell/fn.orientation_bits.html), and the reference
order of one entity's vertices with
[`canonical_order`](../api/tetraloma_refcell/fn.canonical_order.html). Both take the origin ids
of the vertices in the cell's or the entity's local order; only the order of the ids matters,
not their values.

## Edge bits

An edge's reference orientation runs from its lower origin id to its higher one. Each cell of
dimension 2 or 3 has one bit per edge, in edge order from the least significant bit. The bit is
set when the edge's first listed vertex (see
[Reference cell definitions](reference-cells.md)) has the higher origin id, that is, when the
cell's local direction of the edge is against the reference direction. An entity table stores
the edge as (lower, higher).

For example, give the vertices of a triangle the origin ids 30, 4 and 9. Edge 0 runs from id 30
to id 4 and edge 1 from 30 to 9, both against the reference direction, so bits 0 and 1 are set.
Edge 2 runs from 4 to 9, with the reference direction, so bit 2 is clear:

```rust
use tetraloma::CellType;
use tetraloma_refcell::{canonical_order, orientation_bits};

assert_eq!(orientation_bits(CellType::Triangle, &[30, 4, 9]), 0b011);
// The edge (30, 4) is stored from its second local vertex to its first: (4, 30).
assert_eq!(canonical_order(CellType::Interval, &[30, 4]), [1, 0]);
// An interval cell has no bits: its only edge is the cell itself.
assert_eq!(orientation_bits(CellType::Interval, &[30, 4]), 0);
```

## Face bits

A face's reference orientation starts at its lowest origin id and continues towards the lower of
that vertex's two neighbours on the face. The neighbours are taken on the face's boundary cycle:
the list order for a triangle face, and the tensor cycle 0–1–3–2 of the listed vertices for a
quadrilateral face.

After the edge bits, each face of a three-dimensional cell gets three bits, in face order:

- the low two bits hold the **rotation**: the position on the cycle of the face's lowest
  vertex, counted from the first listed vertex;
- the third bit is the **reflection**: it is set when that lowest vertex's successor on the
  cycle has a higher origin id than its predecessor.

For face $f$ of a cell with $E$ edges, the field starts at bit $b = E + 3f$: bits $b$ and $b+1$
hold the rotation, bit $b + 2$ the reflection. Stored in reference orientation, a triangle face
is simply sorted by origin id. A quadrilateral face is stored in tensor order again: slot 0 holds
the lowest vertex, slot 1 its lower neighbour, slot 2 its other neighbour and slot 3 the opposite
vertex.

A worked example, with origin ids as the numbers:

```text
Quadrilateral face, origin ids in the cell's local tensor slots:
  slot 0 (0,0) = 17   slot 1 (1,0) = 4   slot 2 (0,1) = 9   slot 3 (1,1) = 12
Boundary cycle (slots 0-1-3-2):    17 - 4 - 12 - 9
Lowest vertex:                     4, at cycle position 1 -> rotation 1
Its neighbours on the cycle:       successor 12, predecessor 17 -> 12 < 17, reflection 0
Face bits:                         0b001
Stored face (slots 0..3):          [4, 12, 17, 9]
  (lowest 4, lower neighbour 12, other neighbour 17, opposite vertex 9)

Triangle face with origin ids (9, 30, 4)  -> stored [4, 9, 30]
```

The test below checks the example on face 0 of a hexahedron, whose bits start at $12$ because a
hexahedron has 12 edges. It then decodes every field of one tetrahedron:

```rust
use tetraloma::CellType;
use tetraloma_refcell::{canonical_order, orientation_bits};

// Face 0 of a hexahedron carries the ids 17, 4, 9, 12 in its tensor slots.
let ids = [17, 4, 9, 12, 100, 101, 102, 103];
let bits = orientation_bits(CellType::Hexahedron, &ids);
assert_eq!((bits >> 12) & 0b111, 0b001);
let face = [17, 4, 9, 12];
let stored: Vec<i64> =
    canonical_order(CellType::Quadrilateral, &face).iter().map(|&s| face[s]).collect();
assert_eq!(stored, [4, 12, 17, 9]);
let triangle = [9, 30, 4];
let stored: Vec<i64> =
    canonical_order(CellType::Triangle, &triangle).iter().map(|&s| triangle[s]).collect();
assert_eq!(stored, [4, 9, 30]);

// A tetrahedron with origin ids 40, 7, 23, 15: six edge bits, then four 3-bit face fields.
let bits = orientation_bits(CellType::Tetrahedron, &[40, 7, 23, 15]);
// Edges (0,1), (0,2), (0,3) and (2,3) run from the higher id to the lower.
assert_eq!(bits & 0b11_1111, 0b10_0111);
let fields: Vec<u32> = (0..4).map(|f| (bits >> (6 + 3 * f)) & 0b111).collect();
// Face (40, 7, 23): lowest 7 at position 1, successor 23 < predecessor 40.
assert_eq!(fields[0], 0b001);
// Face (40, 7, 15): lowest 7 at position 1, successor 15 < predecessor 40.
assert_eq!(fields[1], 0b001);
// Face (40, 23, 15): lowest 15 at position 2, successor 40 > predecessor 23.
assert_eq!(fields[2], 0b110);
// Face (7, 23, 15): lowest 7 at position 0, successor 23 > predecessor 15.
assert_eq!(fields[3], 0b100);
```

## Storage

A cell's bits fit in one `u32`, laid out as above from the least significant bit. Cells of
dimension 1 have no bits, cells of dimension 2 only edge bits. A hexahedron needs the most, 30
bits, so bits 30 and 31 are always zero. The value is the same as ndelement's
`compute_orientation` for the same vertex ids; ndelement returns it as an `i32`, tetraloma stores
the same bit pattern as a `u32`.

| Cell | Bits used | Layout (bit indices, least significant first) |
| --- | --- | --- |
| Interval | 0 | – |
| Triangle | 3 | edges 0–2 |
| Quadrilateral | 4 | edges 0–3 |
| Tetrahedron | 18 | edges 0–5, faces 6–17 (4 × 3) |
| Hexahedron | 30 | edges 0–11, faces 12–29 (6 × 3) |
| Prism | 24 | edges 0–8, faces 9–23 (5 × 3) |
| Pyramid | 23 | edges 0–7, faces 8–22 (5 × 3) |

A mesh computes the word of every cell on the device when it is created, from the cells' vertex
origin ids, so the result is the same on any number of processes.
[`DeviceGrid::permutation_info`](../api/tetraloma/trait.DeviceGrid.html) returns the words of one
cell block, one per cell, or `None` for a block of intervals. tetraloma also stores the vertex
lists of edges and faces in reference orientation
([`EntityBlock`](../api/tetraloma/struct.EntityBlock.html)'s `vertices`), so a device kernel can
orient an entity without looking anything up; DOLFINx keeps the orientation only in the bits.

This program builds a mesh of one tetrahedron with the origin ids of the previous example and
reads its word back to the host, for inspection only:

```rust
use tetraloma::prelude::*;
use tetraloma_refcell::orientation_bits;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
let coords = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
let ids = [40, 7, 23, 15];
let tet = CoordinateElement::<D3> {
    cell: Cell3::Tetrahedron,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: VolumeMesh<f32> = MeshBuilder::new(&client)
    .nodes(&coords, Some(&ids))
    .cell_block(tet, &[0, 1, 2, 3], None)
    .build()?;
let words = mesh
    .permutation_info(0)
    .expect("tetrahedra have orientation bits")
    .to_host()?;
assert_eq!(words, [orientation_bits(CellType::Tetrahedron, &ids)]);
assert_eq!(words[0] >> 30, 0);
# Ok(())
# }
```
