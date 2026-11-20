# File formats

tetraloma reads and writes Gmsh MSH 4.1 files and writes VTK files (VTU, PVTU and VTKHDF). Both
formats store a cell as a list of node numbers, but in their own order, which differs from
tetraloma's order of [Lagrange nodes](lagrange.md). This chapter lists the cell types of both
formats and the node maps between their orders and tetraloma's. The readers and writers apply
the maps for you; you need them only when you exchange node arrays with other software
yourself. The tutorial [Reading and writing files](../tutorials/read-write.md) shows the
readers and writers at work.

Every node map in this chapter follows one convention. Entry $k$ of a map is tetraloma's index
of the file's $k$-th node:

$$\text{internal}[\text{map}[k]] = \text{file}[k].$$

To turn a row of node numbers in file order into a row in tetraloma's order, you therefore
scatter: `internal[map[k]] = file[k]` for every `k`. To go the other way you gather:
`file[k] = internal[map[k]]`. DOLFINx's arrays use the same direction, although DOLFINx's
documentation states the inverse.

tetraloma-refcell does not store the maps as tables. It generates them, for every degree, from
each format's own recursive rule for placing nodes, and matches the resulting points with
tetraloma's: [`gmsh_to_internal`](../api/tetraloma_refcell/fn.gmsh_to_internal.html) and
[`vtk_to_internal`](../api/tetraloma_refcell/fn.vtk_to_internal.html). The tables below show
them for $p \le 2$; call the functions for higher degrees.

## Gmsh

The Gmsh reader ([`gmsh::read`](../api/tetraloma/gmsh/fn.read.html)) accepts ASCII and binary
MSH 4.1 files in either byte order, and the writer
([`GmshWriter`](../api/tetraloma/gmsh/struct.GmshWriter.html)) writes ASCII by default or binary
on request. The highest element dimension in the file is the mesh's topological dimension, and
lower-dimensional elements become tags. The node tags of the corner nodes become the vertices'
origin ids, and the element tags the cells' origin ids.

The reader accepts the complete Lagrange elements of Gmsh, as listed here by type id with the
number of nodes in brackets:

| Cell | Complete Lagrange: type id (nodes) | Serendipity |
| --- | --- | --- |
| Point | 15 (1) | – |
| Interval | 1 (2), 8 (3), 26 (4), 27 (5), 28 (6), 62–66 (7–11) | – |
| Triangle | 2 (3), 9 (6), 21 (10), 23 (15), 25 (21), 42–46 (28–66) | – |
| Quadrilateral | 3 (4), 10 (9), 36 (16), 37 (25), 38 (36), 47–51 (49–121) | 16 (8) |
| Tetrahedron | 4 (4), 11 (10), 29 (20), 30 (35), 31 (56), 71–75 (84–286) | – |
| Hexahedron | 5 (8), 12 (27), 92 (64), 93 (125), 94–98 (216–1000) | 17 (20) |
| Prism | 6 (6), 13 (18), 90 (40), 91 (75), 106–110 (126–550) | 18 (15) |
| Pyramid | 7 (5), 14 (14), 118–124 (30, 55, 91, 140, 204, 285, 385) | 19 (13) |

The serendipity elements 16 to 19 lack the face and interior nodes of the complete quadratic
elements. The reader adds them by evaluating the serendipity map, so they arrive as complete
Lagrange cells of degree 2 (8 nodes become 9, 20 become 27, 15 become 18 and 13 become 14). Every
other incomplete type is rejected with an error.

Gmsh numbers the nodes of an element as vertices, then the interiors of the edges, then the
interiors of the faces (each face recursively, as a lower-order element), then the interior of
the cell. Its quadrilateral and hexahedron vertices run counter-clockwise, and its edges have
their own order and directions. The maps for $p \le 2$:

| Cell (nodes) | Gmsh → internal |
| --- | --- |
| Interval (2), (3) | identity |
| Triangle (3) | 0 1 2 |
| Triangle (6) | 0 1 2 3 5 4 |
| Quadrilateral (4) | 0 1 3 2 |
| Quadrilateral (9) | 0 1 3 2 4 6 7 5 8 |
| Tetrahedron (4) | 0 1 2 3 |
| Tetrahedron (10) | 0 1 2 3 4 7 5 6 9 8 |
| Hexahedron (8) | 0 1 3 2 4 5 7 6 |
| Hexahedron (27) | 0 1 3 2 4 5 7 6 8 9 10 11 12 13 15 14 16 17 18 19 20 21 22 23 24 25 26 |
| Prism (6), (18) | identity |
| Pyramid (5) | 0 1 3 2 4 |
| Pyramid (14) | 0 1 3 2 4 5 6 7 8 9 10 12 11 13 |

```rust
use tetraloma::CellType;
use tetraloma_refcell::gmsh_to_internal;

let identity = |n: usize| (0..n).collect::<Vec<usize>>();
let rows: [(CellType, u32, Vec<usize>); 14] = [
    (CellType::Interval, 1, identity(2)),
    (CellType::Interval, 2, identity(3)),
    (CellType::Triangle, 1, vec![0, 1, 2]),
    (CellType::Triangle, 2, vec![0, 1, 2, 3, 5, 4]),
    (CellType::Quadrilateral, 1, vec![0, 1, 3, 2]),
    (CellType::Quadrilateral, 2, vec![0, 1, 3, 2, 4, 6, 7, 5, 8]),
    (CellType::Tetrahedron, 1, vec![0, 1, 2, 3]),
    (CellType::Tetrahedron, 2, vec![0, 1, 2, 3, 4, 7, 5, 6, 9, 8]),
    (CellType::Hexahedron, 1, vec![0, 1, 3, 2, 4, 5, 7, 6]),
    (
        CellType::Hexahedron,
        2,
        [vec![0, 1, 3, 2, 4, 5, 7, 6, 8, 9, 10, 11, 12, 13, 15, 14], (16..27).collect()].concat(),
    ),
    (CellType::Prism, 1, identity(6)),
    (CellType::Prism, 2, identity(18)),
    (CellType::Pyramid, 1, vec![0, 1, 3, 2, 4]),
    (CellType::Pyramid, 2, vec![0, 1, 3, 2, 4, 5, 6, 7, 8, 9, 10, 12, 11, 13]),
];
for (cell, p, row) in rows {
    assert_eq!(gmsh_to_internal(cell, p)?, row, "{cell:?}, p = {p}");
}
# Ok::<(), tetraloma_refcell::RefCellError>(())
```

The reader applies these maps. The next program writes a file with one quadratic triangle (type
9) whose edge from vertex 1 to vertex 2 is curved: its midpoint, the file's fifth node, sits at
$(0.6, 0.6)$ instead of $(0.5, 0.5)$. In tetraloma's order this node is number 5, the node of
edge 2. If the reader put it anywhere else, the cell would have a different shape. The
quadratic bulge adds $\tfrac23 \cdot |\delta| \cdot L$ to the area of the straight triangle, with
$|\delta| = 0.1\sqrt2$ the displacement of the midpoint and $L = \sqrt2$ the length of the edge,
so the area is $\tfrac12 + \tfrac{2}{15}$:

```rust
use tetraloma::prelude::*;

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
$Nodes\n1 6 1 6\n2 1 0 6\n1\n2\n3\n4\n5\n6\n\
0 0 0\n1 0 0\n0 1 0\n0.5 0 0\n0.6 0.6 0\n0 0.5 0\n$EndNodes\n\
$Elements\n1 1 1 1\n2 1 9 1\n1 1 2 3 4 5 6\n$EndElements\n";
let path = std::env::temp_dir().join(format!("tetraloma-book-p2-{}.msh", std::process::id()));
std::fs::write(&path, text)?;
let mesh = PlanarMesh::<f64>::try_from(gmsh::read(&client, Serial, &path, GmshOptions::default())?)?;
std::fs::remove_file(&path)?;
let area = mesh.measure(4)?;
assert!((area - (0.5 + 2.0 / 15.0)).abs() < 1e-12, "area {area}");
# Ok(())
# }
```

## VTK

The VTK writer ([`VtkWriter`](../api/tetraloma/vtk/struct.VtkWriter.html)) writes VTU files,
PVTU files for distributed meshes, and VTKHDF files with the feature `hdf5`. Cells of degree 1 are
written as VTK's linear cells, cells of higher degree as VTK's Lagrange cells. Every geometry node
becomes a VTK point, converted to equispaced nodes first. tetraloma writes VTU files with the
version attribute 2.2, because VTK renumbers the nodes of higher-order hexahedra in files of
versions below 2.1.

| Cell | Linear | Lagrange (p ≥ 2) |
| --- | --- | --- |
| Point | 1 | – |
| Interval | 3 | 68 |
| Triangle | 5 | 69 |
| Quadrilateral | 9 | 70 |
| Tetrahedron | 10 | 71 |
| Hexahedron | 12 | 72 |
| Prism | 13 | 73 |
| Pyramid | 14 | none |

VTK has no usable Lagrange pyramid: its type 74 is a placeholder that VTK does not implement. A
pyramid of degree 2 or more is therefore written from its five vertices as type 14, with one
warning per file, and its other nodes stay in the file as points that no cell uses. Write such
meshes to Gmsh files to keep their full degree. With `.lagrange(false)` the writer writes every
cell from its vertices.

VTK numbers the nodes of a Lagrange cell as vertices, then the edges with VTK's own edge order and
directions, then the faces, then the interior. Triangles, and the faces and interiors of
tetrahedra, are numbered recursively; quadrilaterals, hexahedra and prisms (including the
prism's triangular faces) lexicographically. So for $p \ge 2$ the VTK and Gmsh orders differ
even where their vertex orders agree. The maps for $p \le 2$:

| Cell (nodes) | VTK → internal |
| --- | --- |
| Interval (2), (3) | identity |
| Triangle (3) | 0 1 2 |
| Triangle (6) | 0 1 2 3 5 4 |
| Quadrilateral (4) | 0 1 3 2 |
| Quadrilateral (9) | 0 1 3 2 4 6 7 5 8 |
| Tetrahedron (4) | 0 1 2 3 |
| Tetrahedron (10) | 0 1 2 3 4 7 5 6 8 9 |
| Hexahedron (8) | 0 1 3 2 4 5 7 6 |
| Hexahedron (27) | 0 1 3 2 4 5 7 6 8 11 13 9 16 18 19 17 10 12 15 14 22 23 21 24 20 25 26 |
| Prism (6) | identity |
| Prism (18) | 0 1 2 3 4 5 6 9 7 12 14 13 8 10 11 15 17 16 |
| Pyramid (5) | 0 1 3 2 4 |
| Pyramid (14) | not in VTK |

```rust
use tetraloma::CellType;
use tetraloma_refcell::{RefCellError, vtk_to_internal};

let rows: [(CellType, u32, Vec<usize>); 13] = [
    (CellType::Interval, 1, vec![0, 1]),
    (CellType::Interval, 2, vec![0, 1, 2]),
    (CellType::Triangle, 1, vec![0, 1, 2]),
    (CellType::Triangle, 2, vec![0, 1, 2, 3, 5, 4]),
    (CellType::Quadrilateral, 1, vec![0, 1, 3, 2]),
    (CellType::Quadrilateral, 2, vec![0, 1, 3, 2, 4, 6, 7, 5, 8]),
    (CellType::Tetrahedron, 1, vec![0, 1, 2, 3]),
    (CellType::Tetrahedron, 2, vec![0, 1, 2, 3, 4, 7, 5, 6, 8, 9]),
    (CellType::Hexahedron, 1, vec![0, 1, 3, 2, 4, 5, 7, 6]),
    (
        CellType::Hexahedron,
        2,
        vec![
            0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14, 22, 23, 21,
            24, 20, 25, 26,
        ],
    ),
    (CellType::Prism, 1, vec![0, 1, 2, 3, 4, 5]),
    (CellType::Prism, 2, vec![0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16]),
    (CellType::Pyramid, 1, vec![0, 1, 3, 2, 4]),
];
for (cell, p, row) in rows {
    assert_eq!(vtk_to_internal(cell, p)?, row, "{cell:?}, p = {p}");
}
assert_eq!(
    vtk_to_internal(CellType::Pyramid, 2),
    Err(RefCellError::NotInFormat { format: "VTK", cell: CellType::Pyramid, degree: 2 })
);
# Ok::<(), RefCellError>(())
```

The direction of a map is easy to get wrong, so here it is applied to coordinates. VTK lists the
nine nodes of a quadratic quadrilateral counter-clockwise: the four vertices, the midpoints of the
edges from vertex 0 to 1, 1 to 2, 2 to 3 and 3 to 0, and the centre. Scattering them with the map
gives tetraloma's nodes:

```rust
use tetraloma::{CellType, NodeFamily};
use tetraloma_refcell::{lagrange_points, vtk_to_internal};

let vtk = [
    [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    [0.5, 0.0], [1.0, 0.5], [0.5, 1.0], [0.0, 0.5],
    [0.5, 0.5],
];
let map = vtk_to_internal(CellType::Quadrilateral, 2)?;
let mut internal = [[0.0; 2]; 9];
for (k, &i) in map.iter().enumerate() {
    internal[i] = vtk[k];
}
let points = lagrange_points(CellType::Quadrilateral, 2, NodeFamily::Equispaced)?;
assert_eq!(internal.concat(), points);
# Ok::<(), tetraloma_refcell::RefCellError>(())
```
