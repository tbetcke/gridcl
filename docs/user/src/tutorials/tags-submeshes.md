# Tags and submeshes

A mesh often carries labels: the cells of each material, the faces where a boundary condition
holds, the interface between two regions. tetraloma stores these labels as tags on the device.
A tag selects entities, and a submesh turns the selected entities into a mesh of its own. The
boundary of a volume mesh becomes a surface mesh in this way, with its normals pointing out of
the volume, ready for a boundary element method.

This tutorial reads the example file `tetraloma-examples/data/two_regions.msh`. It holds the box
$[0, 2] \times [0, 1]^2$ as 52 cells in two regions: 8 hexahedra fill $[0, 1]^3$, and 40
tetrahedra and 4 pyramids fill $[1, 2] \times [0, 1]^2$. The file names four physical groups:
`hex` and `tet` for the two regions, `domain` for every cell, and `interface` for the four
square faces where the regions meet.

## Physical groups

A physical group is a named set of entities of one dimension, as Gmsh defines it. When you read a
Gmsh file, each physical group becomes a tag value, and its name goes into the name table of the
mesh. [`Mesh::tags`](../api/tetraloma/struct.Mesh.html) returns the
[`TagRegistry`](../api/tetraloma/struct.TagRegistry.html), which holds per dimension one set of
physical tags, the name table and the size of each group.

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
let client = cubecl::Device::cpu()?.client();
// path: tetraloma-examples/data/two_regions.msh
let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
let head: VolumeMesh<f64> = any.try_into()?;

let tags = head.tags();
for (name, dim, value) in tags.names() {
    let size = tags.group_size(dim, value);
    println!("{name:?}: dimension {dim}, value {value}, {size:?} entities");
}
assert_eq!(tags.lookup(3, "hex"), Some(1));
assert_eq!(tags.group_size(3, 1), Some(8)); // the hexahedra
assert_eq!(tags.group_size(3, 3), Some(52)); // "domain": every cell
assert_eq!(tags.group_size(2, 11), Some(4)); // "interface": four faces

// The entities of one group, selected on the device.
let hex: MeshTags = tags.group(3, 1)?;
assert_eq!((hex.dim, hex.indices.len()), (3, 8));
# Ok(()) }
```

A [`MeshTags`](../api/tetraloma/struct.MeshTags.html) holds a dimension and two device arrays of
equal length: the local index of each tagged entity and its value. An entity in several groups
appears once per group, so each cell of this file appears twice in dimension 3, once for its
region and once for `domain`. The tags cover owned and ghost entities alike, and a ghost has the
same values as its owner.

The file's lower-dimensional groups are matched to the mesh's entities by their corner vertices.
Reading a file therefore creates the entities of every tagged dimension. Gmsh's
elementary tags, the numbers of the geometric entities, travel the same way; `elementary(dim)`
returns them.

### Tags from arrays

Meshes built from arrays get their tags from the builder. You give the corner origin ids of each
tagged entity and its value with `physical_tags`, and the names with `tag_names`. Each name is
a triple (name, dimension, value).

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
// The unit square as two triangles, nodes with origin ids 10 to 13.
let coords = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, Some(&[10, 11, 12, 13]))
    .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
    // The second triangle in group 5; the bottom edge (10, 11) in group 1.
    .physical_tags(CellType::Triangle, &[13, 12, 11], &[5])
    .physical_tags(CellType::Interval, &[10, 11], &[1])
    .tag_names(&[("wall", 1, 1), ("domain", 2, 5)])
    .build()?;
assert_eq!(mesh.tags().named(Codim::Bnd, "wall")?.indices.len(), 1);
assert_eq!(mesh.tags().named(Codim::Vol, "domain")?.indices.to_host()?, [1]);
# Ok(()) }
```

The last line reads the tagged indices back to the host to show them. Programs keep tags on the
device and pass them to the methods below.

## Regions

A region is a set of cells, the volume part of a physical group. You select one by name with
[`TagRegistry::named`](../api/tetraloma/struct.TagRegistry.html) and a
[`Codim`](../api/tetraloma/enum.Codim.html). A codimension counts down from the mesh's
topological dimension: `Codim::Vol` is the cells, `Codim::Bnd` their facets, `Codim::BBnd` the
facets of those, and `Codim::BBBnd` one further. The same code then addresses the cells of a
curve, a surface or a volume mesh, as explained in [Typed dimensions](../concepts/typed-dimensions.md#codimension).

`cell_submesh` turns tagged cells into a mesh of the same dimensions, together with an
[`EntityMap`](../api/tetraloma/struct.EntityMap.html) back to the parent:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
# let client = cubecl::Device::cpu()?.client();
# let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
// head: two_regions.msh, read as in "Physical groups"
# let head: VolumeMesh<f64> = any.try_into()?;
let hex = head.tags().named(Codim::Vol, "hex")?;
let (hexahedra, map) = head.cell_submesh(&hex)?;
assert_eq!(map.child_to_parent().len(), 8); // the parent index of each child cell
assert!((hexahedra.measure(1)? - 1.0).abs() < 1e-10);

let tet = head.tags().named(Codim::Vol, "tet")?;
let (rest, _) = head.cell_submesh(&tet)?;
assert_eq!(rest.cell_blocks().len(), 2); // tetrahedra and pyramids
assert!((rest.measure(1)? - 1.0).abs() < 1e-10);
# Ok(()) }
```

A submesh has one cell block per parent block that holds selected cells, in the parent's order.
It keeps the parent's origin and origin ids, and the parent's cell tags become its cell tags.
The entity map gives, on the device, the parent index of each child cell, vertex and node, and the
child index of each parent entity. Code that computes on a region uses the submesh. Code that
moves results back to the parent uses the map.

## Boundary meshes

The boundary of a mesh is the set of its facets with exactly one cell. `boundary` returns them as
a mesh one dimension lower, and needs the facets to exist: create them first with
`create_entities(tdim - 1)`. The boundary of a volume mesh is a surface mesh, the boundary of a
surface mesh is a curve mesh, and the compiler rejects the boundary of a curve.

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
# let client = cubecl::Device::cpu()?.client();
# let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
// head: two_regions.msh, read as in "Physical groups"
# let mut head: VolumeMesh<f64> = any.try_into()?;
head.create_entities(2)?; // the faces; reading the file created them already
let (skin, map): (SurfaceMesh<f64>, EntityMap) = head.boundary()?;
assert_eq!(map.parent_dim(), 2); // child cells are parent faces
// The surface of [0, 2] × [0, 1]² has area 2 · 2 + 2 · 2 + 2 · 1 = 10.
assert!((skin.measure(1)? - 10.0).abs() < 1e-10);

// Every normal points out of the box, whose centre is (1, 1/2, 1/2).
let rule = QuadRule::for_grid(&skin, 1)?;
let x = skin.physical_points(&rule)?;
let n = skin.normals(&rule)?;
for b in 0..skin.cell_blocks().len() {
    let x = x.blocks[b].to_host_cell_major()?; // host reads, for inspection
    let n = n.blocks[b].to_host_cell_major()?;
    for (x, n) in x.iter().zip(&n) {
        let outward = (x[0] - 1.0) * n[0] + (x[1] - 0.5) * n[1] + (x[2] - 0.5) * n[2];
        assert!(outward > 0.0);
    }
}
# Ok(()) }
```

The surface has a block of triangles and a block of quadrilaterals, because the box's boundary
has both. Each boundary cell has one parent cell, and the surface takes its orientation from it,
so the normals point out of the volume without further work (see [Orientation](#orientation)).

Volume meshes have one more submesh: `edge_submesh` turns tagged edges into a curve in three
dimensions. It needs the edges and the edge-to-cell connectivity, from `create_entities(1)` and
`create_connectivity(1, 3)`. The
[`Mesh`](../api/tetraloma/struct.Mesh.html) page lists the requirements of each method.

## Surface meshes

Often only part of the boundary matters: the scalp of a head model, or the wall of a channel.
`facet_submesh` takes a tag of boundary facets and returns them as a surface mesh, oriented like
`boundary`. Here a single hexahedron has its bottom face in the group `bottom`:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
// The unit cube as one hexahedron, vertices in tensor order; the face z = 0 is "bottom".
let coords = [
    0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
    0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0,
];
let hex = CoordinateElement::<D3> { cell: Cell3::Hexahedron, degree: 1, family: NodeFamily::Equispaced };
let cube: VolumeMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, None)
    .cell_block(hex, &[0, 1, 2, 3, 4, 5, 6, 7], None)
    .physical_tags(CellType::Quadrilateral, &[0, 1, 2, 3], &[1])
    .tag_names(&[("bottom", 2, 1)])
    .build()?;

let bottom = cube.tags().named(Codim::Bnd, "bottom")?; // boundary facets only
let (face, map): (SurfaceMesh<f64>, EntityMap) = cube.facet_submesh(&bottom)?;
assert_eq!(map.child_to_parent().len(), 1);
assert!((face.measure(1)? - 1.0).abs() < 1e-12);

// The normal points out of the cube, downwards.
let rule = QuadRule::for_grid(&face, 1)?;
let n = face.normals(&rule)?.blocks[0].to_host_cell_major()?; // a host read, for inspection
assert!(n.iter().all(|n| (n[2] + 1.0).abs() < 1e-12));
# Ok(()) }
```

The program `surface_submesh` does the same on a curved mesh: the unit ball as 64 hexahedra of
degree 2, whose boundary faces carry two generated groups, `boundary-100` and `boundary-101`. It
creates the faces first:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/surface_submesh.rs:entities}}
```

Then, for each group, it builds the surface mesh:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/surface_submesh.rs:facet_submesh}}
```

and evaluates its geometric factors with a quadrature rule of degree 6:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/surface_submesh.rs:factors}}
```

The normals, Jacobians and pseudo-inverses are explained in
[Geometric data](geometric-data.md#normals). The program checks that each normal has length 1 and
points out of the ball, that $K J = I$ at every point, and that the two areas add up to the
sphere's $4\pi$ within the error of the degree-2 geometry. Run it with:

```sh
cargo run --release -p tetraloma-examples --example surface_submesh
```

`facet_submesh` accepts boundary facets only. The `interface` group of the example file lies
between the two regions, so each of its faces has two cells, and the call fails:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let path = concat!(env!("CARGO_MANIFEST_DIR"), "/data/two_regions.msh");
# let client = cubecl::Device::cpu()?.client();
# let any: AnyMesh<f64> = gmsh::read(&client, Serial, path, GmshOptions::default())?;
// head: two_regions.msh, read as in "Physical groups"
# let head: VolumeMesh<f64> = any.try_into()?;
let interface = head.tags().named(Codim::Bnd, "interface")?;
assert_eq!(interface.indices.len(), 4);
let result = head.facet_submesh(&interface);
assert!(matches!(result, Err(GridError::InteriorFacet)));
# Ok(()) }
```

Interface meshes between regions are not supported.

## Orientation

A surface in three dimensions has two unit normals at each point, $n$ and $-n$. The orientation
of a cell picks one of them. Bit 0 of each cell's entry in `CellBlock::flags` records whether the
cell is flipped against the normal that its vertex order gives. `normals` applies these flags.
[Orientation](../concepts/orientation.md#manifold-orientation) explains the mathematics.

Boundary meshes and facet submeshes need no further work. Each child cell has one parent cell,
and its flag is set so that its normal agrees with the outward normal of that parent. Children
two dimensions below their parent, such as the boundary curve of a surface in three dimensions
or an edge submesh, have no normal, and their flags are 0.

A surface you build from arrays, or read from a file, has the orientation of its vertex orders,
which need not agree from cell to cell. `orient` makes it consistent. It exists for meshes of
codimension 1 only, surfaces in three dimensions and curves in the plane. Here the eight
triangles of an octahedron come with mixed vertex orders:

```rust
# use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
// The octahedron with vertices ±e_x, ±e_y and ±e_z.
let coords = [
    1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    0.0, -1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0,
];
let triangles = [0, 2, 4, 2, 1, 4, 0, 3, 4, 1, 3, 4, 0, 2, 5, 1, 2, 5, 3, 0, 5, 1, 3, 5];
let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
let mut octahedron: SurfaceMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, None)
    .cell_block(tri, &triangles, None)
    .build()?;

octahedron.orient(OrientationRule::Outward)?;

let rule = QuadRule::for_grid(&octahedron, 1)?;
let x = octahedron.physical_points(&rule)?.blocks[0].to_host_cell_major()?; // host reads,
let n = octahedron.normals(&rule)?.blocks[0].to_host_cell_major()?; // for inspection
for (x, n) in x.iter().zip(&n) {
    let outward: f64 = x.iter().zip(n).map(|(a, b)| a * b).sum();
    assert!(outward > 0.0); // every normal points away from the centre
}
# Ok(()) }
```

`orient` first checks that every facet has at most two cells (`GridError::NonManifold`
otherwise). It then makes neighbouring cells agree, one connected component at a time, and fails
with `GridError::NonOrientable` on a surface such as a Möbius strip. Last, it picks the direction
of each component from the [`OrientationRule`](../api/tetraloma/enum.OrientationRule.html):

- `Outward` makes the enclosed signed volume $\frac{1}{d} \oint x \cdot n \, ds$ positive, where
  $d$ is the geometric dimension. It needs closed components and returns
  `GridError::NeedsReferencePoint` for a component with a boundary.
- `AwayFrom(p)` makes $\int (x - p) \cdot n \, ds$ non-negative, and `Towards(p)` makes it
  non-positive. These work for open components too.

`orient` overwrites the flags the mesh had, and creates the facets and the facet-to-cell
connectivity it needs. On a distributed mesh it is collective: every rank calls it.
