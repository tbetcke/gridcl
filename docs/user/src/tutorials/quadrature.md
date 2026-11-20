# Quadrature on a mesh

Integrals over a mesh become sums over its cells, and each cell's integral becomes a weighted
sum over points of its reference cell. This tutorial shows the quadrature rules of tetraloma,
uses them to integrate a function over a curved mesh, and follows the error of the geometry as
its degree grows to 8. [Quadrature](../concepts/quadrature.md) gives the mathematics of the
rules.

## Rules

A [`QuadRule`](../api/tetraloma/struct.QuadRule.html) holds one
[`PointSet`](../api/tetraloma/struct.PointSet.html) per cell type, so one rule serves a mesh
with several cell types. `QuadRule::gauss(&client, cells, degree)` makes Gauss rules exact for
polynomials of total degree `degree` on the given cell types. `QuadRule::for_grid(&mesh,
degree)` does the same for every cell type of a mesh, and is the usual way to make one. A point
set holds its reference points, `tdim` coordinates per point, and its weights, both in `f64` on
the host:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
let rule = QuadRule::<f64, D2>::gauss(&client, &[Cell2::Triangle, Cell2::Quadrilateral], 2)?;

let triangle = rule.get(Cell2::Triangle)?;
// The weights add up to the area 1/2 of the reference triangle.
let area: f64 = triangle.weights().iter().sum();
assert!((area - 0.5).abs() < 1e-14);
// Exact for ξ₀², whose integral over the reference triangle is 1/12.
let integral: f64 = triangle
    .points()
    .chunks_exact(2)
    .zip(triangle.weights())
    .map(|(xi, w)| w * xi[0] * xi[0])
    .sum();
assert!((integral - 1.0 / 12.0).abs() < 1e-14);

// On the quadrilateral: 2 Gauss points in each direction.
assert_eq!(rule.get(Cell2::Quadrilateral)?.len(), 4);
# Ok(())
# }
```

Triangles, tetrahedra and pyramids use collapsed Gauss–Jacobi rules. Quadrilaterals and
hexahedra use products of one-dimensional Gauss rules, and prisms a triangle rule times an
interval rule. `get` returns `GridError::CellNotInRule` for a cell type the rule does not
hold. A [`FacetQuadRule`](../api/tetraloma/struct.FacetQuadRule.html) holds points on every
facet of a cell instead, for integrals over cell boundaries and for `facet_normals`.

## Integrating over a mesh

With the physical points $X(\xi_q)$ and determinants $\det J(\xi_q)$ of
[Geometric data](geometric-data.md), the integral of a function $f$ over the mesh is

$$
\int_\Omega f \, dx \approx \sum_{\text{cells}} \sum_q w_q \, \det J(\xi_q) \, f(X(\xi_q)).
$$

The sum is exact if $\det J \cdot (f \circ X)$ is a polynomial of at most the rule's degree on
every cell. On the quarter annulus of degree 4, $X$ has degree 4 in $\xi_1$, so $x^2 \det J$
has degree 15 there, and a rule of degree 16 integrates it exactly. The snippet below computes
$\int x^2 \, dx$, whose exact value on the true annulus is $15\pi/16$. It forms the sum on the
host for inspection; a real program computes it in a kernel, as
[Writing kernels](kernels.md) shows:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::quarter_annulus;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = quarter_annulus(&client, 4, 1, 2)?;
let rule = QuadRule::for_grid(&mesh, 16)?;
let weights = rule.get(Cell2::Quadrilateral)?.weights();

// For inspection: points and determinants on the host, cell by cell.
let x = mesh.physical_points(&rule)?.blocks[0].to_host_cell_major()?;
let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
let integral: f64 = x
    .iter()
    .zip(&detj)
    .enumerate()
    .map(|(i, (p, d))| weights[i % weights.len()] * d * p[0] * p[0])
    .sum();

let exact = 15.0 * std::f64::consts::PI / 16.0;
assert!((integral - exact).abs() / exact < 1e-5);
# Ok(())
# }
```

The remaining error, about $2 \times 10^{-6}$, is the error of the geometry: the cells of
degree 4 only approximate the circular arcs. The measure is the special case $f = 1$, and
`measure(degree)` computes it on the device.

## High-order geometry

The error of the geometry falls fast with the degree $p$ of the cells. On a cell of size $h$,
interpolation of a smooth boundary by polynomials of degree $p$ has an error of order
$h^{p+1}$, and so has the area. The snippet below builds the quarter annulus at degrees 1, 2, 4
and 8, from one cell in $r$ and two in $\theta$, and compares its area with $3\pi/4$:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::quarter_annulus;

let client = cubecl::Device::cpu()?.client();
let exact = 3.0 * std::f64::consts::PI / 4.0;
let mut last = f64::INFINITY;
for p in [1, 2, 4, 8] {
    let mesh: PlanarMesh<f64> = quarter_annulus(&client, p, 1, 2)?;
    // det J has degree 2p − 1 in ξ₁, so degree 2p is exact.
    let error = (mesh.measure(2 * p)? - exact).abs() / exact;
    println!("p = {p}: relative error {error:.1e}");
    assert!(error < last);
    last = error;
}
assert!(last < 1e-11);
# Ok(())
# }
```

It prints about these errors:

| Degree $p$ | Relative error of the area |
| --- | --- |
| 1 | $1.0 \times 10^{-1}$ |
| 2 | $7.8 \times 10^{-4}$ |
| 4 | $1.2 \times 10^{-6}$ |
| 8 | $9.3 \times 10^{-13}$ |

At degree 8 two cells reproduce the arcs to twelve digits. In `f32`, the rounding of single
precision, about $10^{-7}$ per operation, ends the decrease much earlier, so high degrees pay
off mainly in `f64`. The program `geometric_factors` shows the same convergence on the curved
sphere of [Measures](geometric-data.md#measures).

Equispaced nodes interpolate well up to about degree 10. Above that, use
Gauss–Lobatto–Legendre nodes on intervals, quadrilaterals and hexahedra, as
[GLL and equispaced nodes](../concepts/geometry.md#gll-and-equispaced-nodes) explains. A
coordinate element of high degree costs more per point, but tetraloma chooses a suitable
kernel for each degree and device by itself.
