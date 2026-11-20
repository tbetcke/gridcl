# Lagrange geometry

In tetraloma the geometry of a mesh is a field: a vector-valued Lagrange function on the mesh
whose value at a point is that point's position in space. Straight-sided cells use the field of
degree 1. Curved cells use a higher degree, so that the cells can follow a curved boundary
such as a sphere. This chapter explains the coordinate element that defines the field, how
high-order cells work, and the two families of nodes.

## Coordinate elements

A coordinate element ([`CoordinateElement`](../api/tetraloma/struct.CoordinateElement.html))
is a triple: the cell type, the degree $p \ge 1$ and the node family. It fixes a set of nodes
$\hat x_0, \dots, \hat x_{n-1}$ on the reference cell and the Lagrange basis functions
$\phi_0, \dots, \phi_{n-1}$ on them. The basis functions span the polynomials of degree $p$ on
the cell (on intervals, triangles and tetrahedra, total degree $p$; on quadrilaterals and
hexahedra, degree $p$ in each coordinate; prisms combine the triangle and the interval, and
pyramids use a space of rational functions), and they satisfy

$$
\phi_k(\hat x_j) = \delta_{kj} =
\begin{cases} 1 & k = j, \\ 0 & k \ne j. \end{cases}
$$

A cell with node coordinates $x_0, \dots, x_{n-1}$ then has the map
$X(\xi) = \sum_k x_k \phi_k(\xi)$, and $X(\hat x_k) = x_k$: the map passes through every node.

The nodes are ordered in blocks: first the vertices, then the nodes inside each edge in edge
order, then those inside each face, then those inside the cell. Inside an edge the nodes run
from the edge's first vertex to its second. [Lagrange nodes](../elements/lagrange.md#node-ordering)
gives the full ordering and the node counts. A triangle of degree $p$ has
$(p + 1)(p + 2)/2$ nodes, for example 6 at $p = 2$.

All cells of one type in a mesh share one coordinate element, so the cells of a block have the
same number of nodes. A mesh may use different degrees for different cell types. The mesh keeps
one array of unique nodes, and each block's geometry block
([`GeometryBlock`](../api/tetraloma/struct.GeometryBlock.html)) holds, for every cell, the
indices of its nodes in that array. A node shared by two cells is stored once.

tetraloma computes the basis functions on the host in f64. It expands them in an orthonormal
polynomial basis (Legendre, Dubiner, or a rational basis on pyramids) and solves for the
coefficients with the generalised Vandermonde matrix. The values are narrowed to the mesh's
scalar type when they are uploaded to the device.

## High-order geometry

With $p \ge 2$, the map $X$ is no longer affine, and a cell can be curved. The example below
builds a quadratic triangle whose third edge bulges outward. Its corners are $(0, 0)$, $(1, 0)$
and $(0, 1)$, and the nodes inside the first two edges sit at the edge midpoints. The node
inside edge 2 (from vertex 1 to vertex 2) is at $(0.6, 0.6)$ instead of the midpoint
$(0.5, 0.5)$.

The basis function of that node is $\phi_5(\xi) = 4 \xi_0 \xi_1$, and every other node lies
where the identity map puts it. So the map is

$$
X(\xi) = \xi + 0.4\, \xi_0 \xi_1 \begin{pmatrix} 1 \\ 1 \end{pmatrix},
\qquad
J(\xi) = \begin{pmatrix} 1 + 0.4\,\xi_1 & 0.4\,\xi_0 \\ 0.4\,\xi_1 & 1 + 0.4\,\xi_0 \end{pmatrix},
$$

with $\det J = 1 + 0.4\,(\xi_0 + \xi_1)$. The area is
$\int_{\hat T} \det J \, d\xi = 1/2 + 0.4 \cdot 2 \cdot 1/6 = 19/30$: the straight triangle's
$1/2$ plus the parabolic segment of Archimedes, $4/3$ of the triangle with corners $(1, 0)$,
$(0, 1)$ and $(0.6, 0.6)$.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// Six nodes: the three vertices, then one node inside each edge (0, 1), (0, 2), (1, 2).
let coords = [
    0.0, 0.0, 1.0, 0.0, 0.0, 1.0, // vertices
    0.5, 0.0, 0.0, 0.5, 0.6, 0.6, // edge nodes; the last one is moved outward
];
let p2 = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 2,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&coords, None)
    .cell_block(p2, &[0, 1, 2, 3, 4, 5], None)
    .build()?;

// det J has degree 1, so a rule of degree 2 integrates the area exactly.
assert!((mesh.measure(2)? - 19.0 / 30.0).abs() < 1e-12);

// The Jacobians at the points of a rule, against the formula above (column-major: j[d][g]).
let rule = QuadRule::for_grid(&mesh, 2)?;
let xi = rule.get(Cell2::Triangle)?.points().to_vec();
let j = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
for (q, j) in j.iter().enumerate() {
    let (x0, x1) = (xi[2 * q], xi[2 * q + 1]);
    let want = [[1.0 + 0.4 * x1, 0.4 * x1], [0.4 * x0, 1.0 + 0.4 * x0]];
    for d in 0..2 {
        for g in 0..2 {
            assert!((j[d][g] - want[d][g]).abs() < 1e-12);
        }
    }
}
# Ok(()) }
```

High-order cells are how a mesh represents a curved domain accurately. On a sphere meshed with
curved triangles of degree $p$, the distance between the mesh and the sphere falls roughly like
$h^{p+1}$ for cells of size $h$. The tutorial
[Quadrature on a mesh](../tutorials/quadrature.md#high-order-geometry) uses such meshes up to
degree 8.

## GLL and equispaced nodes

Two families of nodes are available ([`NodeFamily`](../api/tetraloma/enum.NodeFamily.html)):

- Equispaced nodes lie on a uniform lattice: on an edge of degree $p$, at the fractions
  $i/p$. They exist on every cell type. Gmsh and VTK use them, so they are the nodes of every
  file tetraloma reads or writes.
- Gauss–Lobatto–Legendre (GLL) nodes are the end points of the interval and the roots of
  $P_p'$, the derivative of the Legendre polynomial of degree $p$, mapped to $[0, 1]$. They
  cluster towards the ends. They exist on intervals, quadrilaterals and hexahedra, where they
  form tensor grids. A GLL element on any other cell type gives
  `GridError::UnsupportedNodeFamily`.

The family matters at high degree. Interpolation on equispaced nodes becomes badly conditioned
as $p$ grows (the Runge phenomenon), while GLL nodes keep it well conditioned. Equispaced nodes
are well conditioned up to about $p = 10$; above that, use GLL nodes on tensor cells. Both
families span the same polynomials, so one converts to the other exactly, up to rounding, by
evaluating the map at the other family's nodes. The writers do this on the device and write
GLL geometry as equispaced nodes.

The example compares the two families for an interval of degree 4. The vertices come first,
then the three interior nodes in order. The interior GLL nodes are the roots of $P_4'$, which
are $0$ and $\pm\sqrt{3/7}$ on $[-1, 1]$.

```rust
use tetraloma::{CellType, NodeFamily};
use tetraloma_refcell::lagrange_points;
# fn main() -> Result<(), Box<dyn std::error::Error>> {

let equi = lagrange_points(CellType::Interval, 4, NodeFamily::Equispaced)?;
assert_eq!(equi, [0.0, 1.0, 0.25, 0.5, 0.75]);

let gll = lagrange_points(CellType::Interval, 4, NodeFamily::Gll)?;
let r = (3.0f64 / 7.0).sqrt();
let want = [0.0, 1.0, (1.0 - r) / 2.0, 0.5, (1.0 + r) / 2.0];
for (x, w) in gll.iter().zip(want) {
    assert!((x - w).abs() < 1e-14);
}

// GLL nodes exist on tensor cells only.
assert!(lagrange_points(CellType::Triangle, 4, NodeFamily::Gll).is_err());
# Ok(()) }
```
