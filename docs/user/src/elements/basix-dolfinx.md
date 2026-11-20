# Basix and DOLFINx

Basix is the element library of the FEniCS project, and DOLFINx its finite element solver. They
use the same reference cells as tetraloma, but they number the edges and faces of triangles and
tetrahedra differently. This chapter gives the conversion, so that you can pass meshes and node
arrays between tetraloma and code written for Basix or DOLFINx. ndelement, Symfem and DefElement
number everything exactly as tetraloma does, so they need no conversion.

tetraloma's library has no conversion function of its own. The conversion is short, though, and
the tests in this chapter build it from tetraloma-refcell's tables.

## Cell and vertex numbering

The reference cells agree everywhere: every cell has the same vertex coordinates, in the same
order, in tetraloma, Basix and DOLFINx, including the tensor order of the quadrilateral and the
hexahedron. The sub-entities of intervals, quadrilaterals, hexahedra, prisms and pyramids are
numbered the same way too.

Triangles and tetrahedra differ. tetraloma numbers their edges and faces by sorted vertex lists
in lexicographic order. Basix and DOLFINx use the reverse order, in which edge $i$ of a
triangle and face $i$ of a tetrahedron lie opposite vertex $i$. Index $i$ in one numbering is
therefore index $n - 1 - i$ in the other, with $n$ the number of entities of that dimension:

| Entities | tetraloma index $i$ → Basix and DOLFINx index |
| --- | --- |
| Triangle edges | $2 - i$ |
| Tetrahedron edges | $5 - i$ |
| Tetrahedron faces | $3 - i$ |
| All other cells and entities; all vertices | $i$ |

Each entity keeps its vertex list; only its number changes. These are Basix's tables for the two
cells, with tetraloma's number of the same entity alongside:

| Basix triangle edge | Vertices | tetraloma edge |
| --- | --- | --- |
| 0 | (1, 2) | 2 |
| 1 | (0, 2) | 1 |
| 2 | (0, 1) | 0 |

| Basix tetrahedron edge | Vertices | tetraloma edge |
| --- | --- | --- |
| 0 | (2, 3) | 5 |
| 1 | (1, 3) | 4 |
| 2 | (1, 2) | 3 |
| 3 | (0, 3) | 2 |
| 4 | (0, 2) | 1 |
| 5 | (0, 1) | 0 |

| Basix tetrahedron face | Vertices | tetraloma face |
| --- | --- | --- |
| 0 | (1, 2, 3) | 3 |
| 1 | (0, 2, 3) | 2 |
| 2 | (0, 1, 3) | 1 |
| 3 | (0, 1, 2) | 0 |

```rust
use tetraloma::CellType;
use tetraloma_refcell::sub_entities;

// Basix's tables, as above.
let triangle_edges = [[1, 2], [0, 2], [0, 1]];
let tetrahedron_edges = [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]];
let tetrahedron_faces = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]];
for (i, edge) in triangle_edges.iter().enumerate() {
    assert_eq!(sub_entities(CellType::Triangle, 1)[2 - i], edge);
    assert!(!edge.contains(&i)); // Basix edge i is opposite vertex i.
}
for (i, edge) in tetrahedron_edges.iter().enumerate() {
    assert_eq!(sub_entities(CellType::Tetrahedron, 1)[5 - i], edge);
}
for (i, face) in tetrahedron_faces.iter().enumerate() {
    assert_eq!(sub_entities(CellType::Tetrahedron, 2)[3 - i], face);
    assert!(!face.contains(&i)); // Basix face i is opposite vertex i.
}
```

## Node orderings

Basix places Lagrange nodes by the same block rule as tetraloma: vertices, then edges, faces and
the interior, each sub-entity's nodes following its own vertex list (see
[Lagrange nodes](lagrange.md#node-ordering)). Since an entity keeps its vertex list in both
numberings, the nodes inside each block come in the same order. Only the blocks of the edges of
triangles, and of the edges and faces of tetrahedra, swap places, by the index conversion above.
For every other cell the two node orders are identical.

The test below builds the node map of a triangle or tetrahedron, with entry $b$ tetraloma's index
of Basix node $b$. It then uses DOLFINx's maps from Gmsh files, which give Basix indices, to
derive the maps of the chapter [File formats](file-formats.md). DOLFINx's map for the 6-node
triangle is `0 1 2 5 3 4`, and for the 10-node triangle `0 1 2 7 8 3 4 6 5 9`:

```rust
use tetraloma::{CellType, NodeFamily};
use tetraloma_refcell::{
    gmsh_to_internal, lagrange_entity_nodes, lagrange_points, num_lagrange_nodes,
    num_sub_entities, tdim,
};

/// Entry b: tetraloma's index of Basix node b, on a triangle or tetrahedron of degree p.
fn basix_to_internal(cell: CellType, p: u32) -> Vec<usize> {
    let mut map: Vec<usize> = (0..num_lagrange_nodes(cell, p)).collect();
    // Edges, and the faces of a tetrahedron; vertices and the interior keep their numbers.
    for dim in 1..tdim(cell) {
        let n = num_sub_entities(cell, dim);
        for e in 0..n {
            // Every block of this dimension has the same size, so Basix's block e takes the
            // place of tetraloma's block e, and holds the nodes of tetraloma's entity n − 1 − e.
            let basix = lagrange_entity_nodes(cell, p, dim, e);
            let ours = lagrange_entity_nodes(cell, p, dim, n - 1 - e);
            for (b, t) in basix.zip(ours) {
                map[b] = t;
            }
        }
    }
    map
}

# fn main() -> Result<(), tetraloma_refcell::RefCellError> {
// DOLFINx's Gmsh maps hold Basix indices; converted, they are tetraloma's Gmsh maps.
for (p, dolfinx) in [(2, vec![0, 1, 2, 5, 3, 4]), (3, vec![0, 1, 2, 7, 8, 3, 4, 6, 5, 9])] {
    let to_internal = basix_to_internal(CellType::Triangle, p);
    let converted: Vec<usize> = dolfinx.iter().map(|&b| to_internal[b]).collect();
    assert_eq!(converted, gmsh_to_internal(CellType::Triangle, p)?);
}

// On the quadratic tetrahedron, Basix node 4 is the midpoint of Basix edge 0, (2, 3).
let cell = CellType::Tetrahedron;
let to_internal = basix_to_internal(cell, 2);
assert_eq!(to_internal, [0, 1, 2, 3, 9, 8, 7, 6, 5, 4]);
let points = lagrange_points(cell, 2, NodeFamily::Equispaced)?;
let t = to_internal[4];
assert_eq!(points[3 * t..3 * t + 3], [0.0, 0.5, 0.5]);
# Ok(())
# }
```

## Converting a mesh

To pass a tetraloma mesh to code that expects Basix or DOLFINx numbering, convert what refers to
local numbers of sub-entities or nodes. Everything else carries over unchanged.

- **Vertices and coordinates** need no change: the cell vertex lists and the node coordinates
  are the same in both conventions.
- **Geometry node rows** of triangles and tetrahedra of degree 2 or more need reordering. A
  [`GeometryBlock`](../api/tetraloma/struct.GeometryBlock.html)'s `dofmap` holds one row of
  node numbers per cell in tetraloma's order; the row in Basix order has entry
  $\text{row}_\text{Basix}[b] = \text{row}[\text{map}[b]]$, with the map of the previous
  section. Cells of degree 1 and all other cell types keep their rows.
- **Local sub-entity numbers**, for example the local facet of a cell that a boundary
  condition refers to, convert with the index table: local edge $i$ of a triangle is Basix edge
  $2 - i$, local face $i$ of a tetrahedron is Basix face $3 - i$.
- **Orientation bits** do not translate field by field. DOLFINx stores the faces first, puts the
  reflection in the lowest bit, and defines its rotation relative to both the local and the
  global lowest vertex. Recompute DOLFINx's bits from the vertex origin ids with DOLFINx's own
  rule instead; the [orientation bits](orientation-bits.md) of tetraloma come from the same ids.
- **DOF transformations**, the matrices that correct the degrees of freedom on an entity
  according to its orientation, depend only on the type of the sub-entity. They are the same in
  both numberings; only the block of the cell's transformation in which each one acts moves
  with the entity's number.

For example, the node row of one quadratic triangle and of one quadratic tetrahedron:

```rust
use tetraloma::CellType;
use tetraloma_refcell::{lagrange_entity_nodes, num_lagrange_nodes, num_sub_entities, tdim};

// The map of the previous section, as a closure.
let basix_to_internal = |cell: CellType, p: u32| {
    let mut map: Vec<usize> = (0..num_lagrange_nodes(cell, p)).collect();
    for dim in 1..tdim(cell) {
        let n = num_sub_entities(cell, dim);
        for e in 0..n {
            let ours = lagrange_entity_nodes(cell, p, dim, n - 1 - e);
            for (b, t) in lagrange_entity_nodes(cell, p, dim, e).zip(ours) {
                map[b] = t;
            }
        }
    }
    map
};

// A row of node numbers in tetraloma's order: vertices, then the edges (0,1), (0,2), (1,2).
let row = [10, 11, 12, 13, 14, 15];
let map = basix_to_internal(CellType::Triangle, 2);
let basix_row: Vec<u32> = map.iter().map(|&i| row[i]).collect();
// Basix's edges are (1,2), (0,2), (0,1).
assert_eq!(basix_row, [10, 11, 12, 15, 14, 13]);

let row = [20, 21, 22, 23, 24, 25, 26, 27, 28, 29];
let map = basix_to_internal(CellType::Tetrahedron, 2);
let basix_row: Vec<u32> = map.iter().map(|&i| row[i]).collect();
assert_eq!(basix_row, [20, 21, 22, 23, 29, 28, 27, 26, 25, 24]);
```
