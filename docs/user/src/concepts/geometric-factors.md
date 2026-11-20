# Geometric factors

To integrate or differentiate on a physical cell, you work on the reference cell and correct
for the map $X$ between them. The corrections are the geometric factors: the Jacobian, its
determinant, its (pseudo-)inverse, and the normals. tetraloma computes them on the device at
the points of a quadrature rule, for every cell of a block at once, and returns them as
[`BlockFields`](../api/tetraloma/struct.BlockFields.html): one field per cell block, with one
value per cell and point. The tutorial [Geometric data](../tutorials/geometric-data.md) shows
the calls; this chapter defines what they compute.

Throughout, $t$ is the topological dimension, $g$ the geometric dimension and $\xi$ a point of
the reference cell. The examples use one straight triangle in 3D with the vertices

$$
x_0 = (1, 0, 0), \qquad x_1 = (3, 0, 0), \qquad x_2 = (2, 2, 2),
$$

whose numbers are small enough to check by hand. The examples copy each field to the host
with `to_host_cell_major` only to compare it with these numbers; a program that computes with
the factors passes them to its own kernels and leaves them on the device.

## The Jacobian

The Jacobian $J(\xi)$ is the matrix of first derivatives of the map $X$. It has $g$ rows, one
per physical coordinate, and $t$ columns, one per reference coordinate:

$$
J_{\gamma d}(\xi) = \frac{\partial X_\gamma}{\partial \xi_d}(\xi)
= \sum_k x_{k,\gamma}\, \frac{\partial \phi_k}{\partial \xi_d}(\xi),
\qquad \gamma = 0, \dots, g - 1, \quad d = 0, \dots, t - 1.
$$

Column $d$, written $J_d$, is the tangent vector of the cell along reference direction $d$.
For an affine simplex the columns are the edge vectors $J_d = x_{d+1} - x_0$. For the triangle
above,

$$
J = \begin{pmatrix} J_0 & J_1 \end{pmatrix}
  = \begin{pmatrix} 2 & 1 \\ 0 & 2 \\ 0 & 2 \end{pmatrix}.
$$

tetraloma stores $J$ column-major: entry $(\gamma, d)$ is component $d\,g + \gamma$ of the
field. On the host, [`to_host_cell_major`](../api/tetraloma/struct.QuadField.html) returns one
array of $t$ columns of $g$ entries per point, so `j[d][γ]` is $J_{\gamma d}$. The type
system fixes the shape: a surface mesh in 3D gives `[[F; 3]; 2]` per point, and a volume mesh
`[[F; 3]; 3]`.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// The triangle (1, 0, 0), (3, 0, 0), (2, 2, 2) in 3D.
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;

// One cell, one point: the columns J0 = (2, 0, 0) and J1 = (1, 2, 2).
let j = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
assert_eq!(j.len(), 1);
let want = [[2.0, 0.0, 0.0], [1.0, 2.0, 2.0]];
for d in 0..2 {
    for g in 0..3 {
        assert!((j[0][d][g] - want[d][g]).abs() < 1e-12);
    }
}
# Ok(()) }
```

## The determinant

In codimension 0 ($t = g$), $J$ is square and $\det J$ is the factor by which the map scales
volume: $dx = \det J \, d\xi$. It is positive when the cell keeps the orientation of the
reference cell and negative when the map mirrors it. tetraloma returns the signed value, so a
negative determinant tells you that a cell's vertices are listed in the wrong order.

When $t < g$, $J$ is not square, and the volume factor is the pseudo-determinant

$$
\sqrt{\det\left(J^{\mathsf T} J\right)}.
$$

$J^{\mathsf T} J$ is the $t \times t$ matrix of dot products of the tangent vectors (the
metric tensor of the surface), and the formula gives the length of $J_0$ for a curve and the
area $|J_0 \times J_1|$ of the parallelogram spanned by $J_0$ and $J_1$ for a surface in 3D. It
is never negative. `detj` on [`Mesh`](../api/tetraloma/struct.Mesh.html) chooses the formula
from the mesh's type.

For the triangle in 3D,

$$
J^{\mathsf T} J = \begin{pmatrix} 4 & 2 \\ 2 & 9 \end{pmatrix},
\qquad
\sqrt{\det\left(J^{\mathsf T} J\right)} = \sqrt{32} = 4\sqrt 2 .
$$

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2), as above
# let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
#     .cell_block(tri, &[0, 1, 2], None)
#     .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;
let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
assert!((detj[0] - 4.0 * 2f64.sqrt()).abs() < 1e-12);
# Ok(()) }
```

## The pseudo-inverse

In codimension 0 the inverse $J^{-1}$ turns reference derivatives into physical ones: for a
function $u(x) = \hat u(\xi)$, the chain rule gives
$\nabla_x u = J^{-\mathsf T} \nabla_\xi \hat u$. A surface has no inverse Jacobian, because $J$
is not square. tetraloma uses the pseudo-inverse instead,

$$
K = \left(J^{\mathsf T} J\right)^{-1} J^{\mathsf T},
$$

a $t \times g$ matrix. It satisfies $K J = I_t$, and $J K$ is the orthogonal projection onto
the tangent space of the cell. When $t = g$, $K = J^{-1}$. With $K$ the surface gradient of $u$
is $\nabla_\Gamma u = K^{\mathsf T} \nabla_\xi \hat u$, a tangent vector in $\mathbb R^g$; this
is what boundary element methods need.

For the triangle in 3D,

$$
K = \frac{1}{32}\begin{pmatrix} 9 & -2 \\ -2 & 4 \end{pmatrix}
    \begin{pmatrix} 2 & 0 & 0 \\ 1 & 2 & 2 \end{pmatrix}
  = \begin{pmatrix} 1/2 & -1/8 & -1/8 \\ 0 & 1/4 & 1/4 \end{pmatrix}.
$$

tetraloma stores $K$ column-major too: entry $(d, \gamma)$ is component $\gamma\,t + d$, and on
the host `k[γ][d]` is $K_{d\gamma}$, one array of $g$ columns of $t$ entries per point.
`pseudo_inverse_jacobians` exists for every mesh; `inverse_jacobians`, which returns the same
values, exists only in codimension 0 (see [Typed dimensions](typed-dimensions.md)).

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2), as above
# let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
#     .cell_block(tri, &[0, 1, 2], None)
#     .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;
let k = mesh.pseudo_inverse_jacobians(&rule)?.blocks[0].to_host_cell_major()?;
let j = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;

// Column γ of K holds (K_0γ, K_1γ).
let want = [[0.5, 0.0], [-0.125, 0.25], [-0.125, 0.25]];
for g in 0..3 {
    for d in 0..2 {
        assert!((k[0][g][d] - want[g][d]).abs() < 1e-12);
    }
}

// K J = I.
for a in 0..2 {
    for b in 0..2 {
        let kj: f64 = (0..3).map(|g| k[0][g][a] * j[0][b][g]).sum();
        let id = if a == b { 1.0 } else { 0.0 };
        assert!((kj - id).abs() < 1e-12);
    }
}
# Ok(()) }
```

## Measure

The measure of a cell $T = X(\hat T)$ is its length, area or volume:

$$
|T| = \int_{\hat T} \det J(\xi) \, d\xi
\;\approx\; \sum_q \det J(\xi_q)\, w_q,
$$

where $\xi_q$ and $w_q$ are the points and weights of a quadrature rule on the reference cell
$\hat T$ (see [Quadrature](quadrature.md)), and $\det J$ stands for the pseudo-determinant when
$t < g$. In codimension 0 the sum uses the signed $\det J$, so a mirrored cell counts
negatively. For an affine cell $\det J$ is constant, and a rule of degree 0 or 1 is exact; for a
curved cell of degree $p$, $\det J$ is a polynomial of higher degree (in codimension 0, at most
$t(p - 1)$ on simplices), and the rule must be correspondingly finer.

`measure(degree)` on [`Mesh`](../api/tetraloma/struct.Mesh.html) sums over the owned cells of
all ranks with Gauss rules of the given degree. The sum is formed in three stages, so that its
result is reproducible and accurate:

1. On the device, each cell sums $\det J(\xi_q)\, w_q$ over its points, in point order, in the
   mesh's scalar type.
2. Still on the device, a fixed-order blocked sum adds the cells of each block. No atomics are
   involved, so the order does not depend on how the work is scheduled.
3. Each block's total is read back once and converted to f64. The host adds the blocks in f64,
   and the ranks add their totals in f64.

`measure` therefore returns an `f64`, also for an f32 mesh. For the triangle in 3D, the area is
half the parallelogram: $|T| = \tfrac12 \cdot 4\sqrt 2 = 2\sqrt 2$, which is the
pseudo-determinant times the area $1/2$ of the reference triangle.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2), as above
# let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
#     .cell_block(tri, &[0, 1, 2], None)
#     .build()?;
let area: f64 = mesh.measure(1)?;
assert!((area - 2.0 * 2f64.sqrt()).abs() < 1e-12);
# Ok(()) }
```

## Normals

A mesh of codimension 1, a surface in 3D or a curve in 2D, has a unit normal at every point.
tetraloma computes it from the columns of $J$:

$$
n = s\,\frac{J_0 \times J_1}{|J_0 \times J_1|} \quad (t = 2,\ g = 3),
\qquad
n = s\,\frac{(J_{1 0},\, -J_{0 0})}{|J_0|} \quad (t = 1,\ g = 2).
$$

Without the sign $s$, the direction of $n$ follows from the order of the cell's vertices: on a
surface, the right-hand rule along $J_0$ then $J_1$; on a curve, the tangent turned clockwise,
which points outward on a curve that runs counter-clockwise around a region. The sign
$s = \pm 1$ is the cell's orientation flag, bit 0 of the cell block's `flags`: $s = -1$ where
it is set and $+1$ otherwise. A freshly built mesh has every flag clear. `orient` sets the flags
so that the normals of neighbouring cells agree (see
[Orientation](orientation.md#manifold-orientation)).

A curve in 3D has codimension 2 and no unique normal, and a volume mesh has codimension 0, so
`normals` exists only on codimension-1 meshes. For the triangle in 3D,

$$
J_0 \times J_1 = (2, 0, 0) \times (1, 2, 2) = (0, -4, 4),
\qquad
n = \frac{1}{\sqrt 2}\,(0, -1, 1).
$$

Its length $4\sqrt 2$ is the pseudo-determinant again: for a surface in 3D,
$|J_0 \times J_1| = \sqrt{\det(J^{\mathsf T} J)}$.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2), as above
# let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
#     .cell_block(tri, &[0, 1, 2], None)
#     .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;
let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
let r = 0.5f64.sqrt();
let want = [0.0, -r, r];
for g in 0..3 {
    assert!((n[0][g] - want[g]).abs() < 1e-12);
}
// The flag of the one cell is clear, so s = +1.
assert_eq!(mesh.cell_blocks()[0].flags.to_host()?, [0]);
# Ok(()) }
```

## Facet normals

A cell of codimension 0 has an outward unit normal on each of its facets. These normals are
what you need for boundary integrals and for the fluxes between neighbouring cells. Each
reference cell has fixed outward unit normals $\hat n$ on its facets; for the reference
triangle they are $(0, -1)$ on edge 0, $(-1, 0)$ on edge 1 and $(1, 1)/\sqrt 2$ on edge 2. The
physical normal is

$$
n = \frac{K^{\mathsf T} \hat n}{\left|K^{\mathsf T} \hat n\right|},
\qquad K = J^{-1}.
$$

To see why, take a vector $\hat v$ at a point of the reference facet. The map sends it to
$J \hat v$, and $(K^{\mathsf T} \hat n) \cdot (J \hat v) = \hat n \cdot (K J \hat v)
= \hat n \cdot \hat v$. So $K^{\mathsf T} \hat n$ is orthogonal to every tangent $J \hat v$ of
the physical facet ($\hat n \cdot \hat v = 0$), and it has a positive dot product with every
vector that the map sends out of the cell ($\hat n \cdot \hat v > 0$). It points outward
whatever the sign of $\det J$.

`facet_normals` takes a facet rule ([`FacetQuadRule`](../api/tetraloma/struct.FacetQuadRule.html))
and returns the normals at its points, facet by facet (see
[Quadrature](quadrature.md#facet-rules)). In the example below, the triangle $(0, 0)$,
$(2, 0)$, $(0, 1)$ has $J = \operatorname{diag}(2, 1)$ and $K = \operatorname{diag}(1/2, 1)$.
On edge 2, $K^{\mathsf T} \hat n \propto (1/2, 1)$, so $n = (1, 2)/\sqrt 5$, perpendicular to
the edge from $(2, 0)$ to $(0, 1)$.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// The triangle (0, 0), (2, 0), (0, 1) in the plane.
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 2.0, 0.0, 0.0, 1.0], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;

// One point on each edge, facet by facet.
let rule = FacetQuadRule::for_grid(&mesh, 1)?;
let n = mesh.facet_normals(&rule)?.blocks[0].to_host_cell_major()?;
let s = 5f64.sqrt();
let want = [[0.0, -1.0], [-1.0, 0.0], [1.0 / s, 2.0 / s]];
assert_eq!(n.len(), 3);
for (n, w) in n.iter().zip(want) {
    assert!((n[0] - w[0]).abs() < 1e-12 && (n[1] - w[1]).abs() < 1e-12);
}
# Ok(()) }
```
