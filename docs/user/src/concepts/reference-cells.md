# Reference cells and maps

Every cell of a mesh is the image of one fixed cell, its reference cell, under a map. You do
all calculus on the reference cell, once per cell type, and the map carries the results to each
physical cell. This chapter defines the reference cells, the numbering of their parts and the
map. The full tables, with figures, are in
[Reference cell definitions](../elements/reference-cells.md).

## Reference cells

tetraloma uses the reference cells of DefElement, which ndelement, Basix and Symfem share. Each
reference cell lies in the unit box $[0, 1]^t$, where $t$ is its topological dimension. A point
of the reference cell has reference coordinates $\xi = (\xi_0, \dots, \xi_{t-1})$.

| Cell | $t$ | Vertices (local index: coordinates) |
| --- | --- | --- |
| Interval | 1 | 0: (0), 1: (1) |
| Triangle | 2 | 0: (0, 0), 1: (1, 0), 2: (0, 1) |
| Quadrilateral | 2 | 0: (0, 0), 1: (1, 0), 2: (0, 1), 3: (1, 1) |
| Tetrahedron | 3 | 0: (0, 0, 0), 1: (1, 0, 0), 2: (0, 1, 0), 3: (0, 0, 1) |
| Hexahedron | 3 | the eight corners of $[0, 1]^3$, with $\xi_0$ fastest, then $\xi_1$, then $\xi_2$ |
| Prism | 3 | the triangle at $\xi_2 = 0$ (0, 1, 2), then at $\xi_2 = 1$ (3, 4, 5) |
| Pyramid | 3 | the unit square at $\xi_2 = 0$ (0 to 3, tensor order), then the apex 4: (0, 0, 1) |

Quadrilaterals and hexahedra number their vertices in tensor (lexicographic) order: the first
coordinate runs fastest. The boundary of the reference quadrilateral therefore runs through the
vertices 0, 1, 3, 2, not 0, 1, 2, 3. Gmsh and VTK number the vertices of these cells
counter-clockwise instead. The readers and writers convert between the two orders, so you meet
the tensor order only when you build a mesh from arrays or read the vertices of a cell. The
tensor order keeps the product structure that fast kernels on these cells use.

The volumes of the reference cells are 1 for the interval, the quadrilateral and the
hexahedron, $1/2$ for the triangle and the prism, and $1/6$ for the tetrahedron. The pyramid
has volume $1/3$.

## Sub-entities

The edges and faces of a reference cell are its sub-entities. Each has a local index and a list
of local vertices. tetraloma numbers the sub-entities of each dimension by their sorted vertex
lists, in lexicographic order. One rule therefore numbers every cell:

- triangle edges: 0: (0, 1), 1: (0, 2), 2: (1, 2); edge $i$ is opposite vertex $2 - i$;
- tetrahedron faces: 0: (0, 1, 2), 1: (0, 1, 3), 2: (0, 2, 3), 3: (1, 2, 3); face $i$ is
  opposite vertex $3 - i$;
- quadrilateral edges: 0: (0, 1), 1: (0, 2), 2: (1, 3), 3: (2, 3).

An edge's vertex list gives it a direction, from the first vertex to the second. A
quadrilateral face lists its vertices in its own tensor order, like the quadrilateral cell.
Prisms and pyramids have faces of two types: the prism has two triangles and three
quadrilaterals, and the pyramid one quadrilateral and four triangles. Every cell therefore
stores the type of each of its faces next to the vertex list.

For triangles and tetrahedra this numbering is the reverse of the one Basix and DOLFINx use.
[Basix and DOLFINx](../elements/basix-dolfinx.md) gives the conversion. The functions
[`sub_entities`](../api/tetraloma_refcell/fn.sub_entities.html) and
[`vertices`](../api/tetraloma_refcell/fn.vertices.html) of tetraloma-refcell return these
tables:

```rust
use tetraloma::CellType;
use tetraloma_refcell::{sub_entities, sub_entity_types, vertices};

// The reference triangle and its edges.
assert_eq!(vertices(CellType::Triangle), [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]);
assert_eq!(sub_entities(CellType::Triangle, 1), [[0, 1], [0, 2], [1, 2]]);

// The faces of a prism: a triangle, three quadrilaterals and a triangle.
let faces = sub_entity_types(CellType::Prism, 2);
assert_eq!(faces[0], CellType::Triangle);
assert_eq!(faces[1], CellType::Quadrilateral);
assert_eq!(faces[4], CellType::Triangle);
assert_eq!(sub_entities(CellType::Prism, 2)[1], [0, 1, 3, 4]);
```

## The reference-to-physical map

A cell $K$ of the mesh is the image of its reference cell $\hat K$ under a map

$$
X : \hat K \to K, \qquad x = X(\xi).
$$

The map is a polynomial (for pyramids, a rational function) built from the cell's nodes. If
the nodes have coordinates $x_k$, it is

$$
X(\xi) = \sum_{k=0}^{n-1} x_k \, \phi_k(\xi),
$$

where $\phi_k$ are the Lagrange basis functions of the cell's coordinate element: $\phi_k$ is 1
at node $k$ and 0 at every other node. [Lagrange geometry](geometry.md) describes these
functions and the order of the nodes.

The simplest case is a straight-sided simplex (an interval, triangle or tetrahedron with one
node per vertex). Its map is affine:

$$
X(\xi) = x_0 + \sum_{d=0}^{t-1} \xi_d \, (x_{d+1} - x_0).
$$

The columns $x_{d+1} - x_0$ are the cell's edge vectors from vertex 0, and they form the
Jacobian of the map (see [Geometric factors](geometric-factors.md)). A straight-sided
quadrilateral has the bilinear map

$$
X(\xi) = (1 - \xi_0)(1 - \xi_1)\, x_0 + \xi_0 (1 - \xi_1)\, x_1
       + (1 - \xi_0)\, \xi_1\, x_2 + \xi_0 \xi_1\, x_3,
$$

which is affine only when the quadrilateral is a parallelogram.

The point $x = X(\xi)$ has $g$ coordinates, and $\xi$ has $t$. When $g > t$, as for a
triangle in 3D, the map describes a piece of a surface. Going from $x$ back to $\xi$ is the
[pull-back](pull-back.md).

The example below maps the centroid of the reference triangle, $(1/3, 1/3)$, to a physical
triangle in the plane. The one-point Gauss rule of the triangle sits at that centroid, and an
affine map sends it to the mean of the physical vertices.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// One triangle with vertices (1, 0), (3, 1) and (1, 2).
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[1.0, 0.0, 3.0, 1.0, 1.0, 2.0], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;

// The Gauss rule of degree 1 has one point, the centroid of the reference triangle.
let rule = QuadRule::for_grid(&mesh, 1)?;
let pts = rule.get(Cell2::Triangle)?;
assert_eq!(pts.len(), 1);
assert!((pts.points()[0] - 1.0 / 3.0).abs() < 1e-14);
assert!((pts.points()[1] - 1.0 / 3.0).abs() < 1e-14);

// X(1/3, 1/3) is the mean of the vertices, (5/3, 1).
let x = mesh.physical_points(&rule)?.blocks[0].to_host_cell_major()?;
assert!((x[0][0] - 5.0 / 3.0).abs() < 1e-12);
assert!((x[0][1] - 1.0).abs() < 1e-12);
# Ok(()) }
```
