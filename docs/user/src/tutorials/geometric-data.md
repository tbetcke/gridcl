# Geometric data

Every finite element or boundary element code needs the same few quantities at its quadrature
points: the physical points, the Jacobians of the map from the reference cell, their
determinants and (pseudo-)inverses, and the normals. tetraloma computes each of them on the
device with one method call. This tutorial calls each method on small meshes whose values you
can check by hand. [Geometric factors](../concepts/geometric-factors.md) gives the
mathematics behind them.

Each method takes a [`QuadRule`](../api/tetraloma/struct.QuadRule.html), which fixes the
reference points, and returns a [`BlockFields`](../api/tetraloma/struct.BlockFields.html): one
[`QuadField`](../api/tetraloma/struct.QuadField.html) per cell block, with a value per cell
and point. A field stays on the device, stored in tiles of consecutive cells so that kernels
read it fast. `to_host_cell_major` copies it to the host in plain
$[\text{cell}][\text{point}]$ order (cell-major); the snippets use it to inspect the values.

## Points

`physical_points(&rule)` returns the physical point $X(\xi_q)$ of every reference point
$\xi_q$ of the rule, in every cell. These are absolute coordinates: the method adds the mesh's
origin back. The rule's [`PointSet`](../api/tetraloma/struct.PointSet.html) for a cell type
holds the reference points and weights. The snippet below maps a Gauss rule onto the curved
quarter annulus of [Higher-degree geometry](build-mesh.md#higher-degree-geometry):

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::quarter_annulus;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = quarter_annulus(&client, 2, 1, 2)?;
let rule = QuadRule::for_grid(&mesh, 3)?; // 2 × 2 Gauss points on each quadrilateral
let x = mesh.physical_points(&rule)?;

let field = &x.blocks[0]; // the only cell block
assert_eq!((field.num_cells(), field.num_points()), (2, 4));
// For inspection: every point lies in the annulus 1 ≤ r ≤ 2.
for p in field.to_host_cell_major()? {
    let r = p[0].hypot(p[1]);
    assert!((1.0..=2.0).contains(&r));
}
# Ok(())
# }
```

## Jacobians

The Jacobian $J$ of the map $X$ at a point has the entries
$J_{\gamma d} = \partial X_\gamma / \partial \xi_d$. It has $g$ rows and $t$ columns, for the
geometric dimension $g$ and the topological dimension $t$, and tetraloma stores it column by
column (column-major). On the host a Jacobian of a planar mesh is a `[[F; 2]; 2]`, and
`j[d]` is the column $\partial X / \partial \xi_d$. For a triangle of degree 1 the columns are
the edge vectors from vertex 0 to vertices 1 and 2, the same at every point:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
// One triangle with the vertices (0, 0), (2, 0) and (1, 1).
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 2.0, 0.0, 1.0, 1.0], None)
    .cell_block(triangle, &[0, 1, 2], None)
    .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?; // one point

let j = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
// The columns (2, 0) and (1, 1).
let entries: Vec<f64> = j[0].iter().flatten().copied().collect();
let want = [2.0, 0.0, 1.0, 1.0];
assert!(entries.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));
# Ok(())
# }
```

A surface mesh in 3D has $3 \times 2$ Jacobians, `[[F; 3]; 2]` on the host, and a curve in 3D
has $3 \times 1$ ones. The shape follows from the mesh type, so a mismatch is a compile error,
not a wrong index at run time.

## Determinants

`detj(&rule)` returns $\det J$ at every point. When the cells have lower dimension than the
space, as on a surface in 3D, $J$ is not square, and the method returns
$\sqrt{\det(J^{\mathsf T} J)}$, the pseudo-determinant, instead: the factor by which the map
stretches area or length. Either way,
$\int_T f \, dx = \int_{\hat T} f(X(\xi)) \, \det J(\xi) \, d\xi$ over the reference cell
$\hat T$ of a cell $T$. For a triangle of a planar mesh, $\det J$ is positive if the vertices run
counter-clockwise. The triangle above has $\det J = 2$, twice its area, because the reference
triangle has area $1/2$:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
// mesh: the triangle (0, 0), (2, 0), (1, 1) of the previous snippet.
# let triangle = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
# let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[0.0, 0.0, 2.0, 0.0, 1.0, 1.0], None)
#     .cell_block(triangle, &[0, 1, 2], None)
#     .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;
let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
assert!((detj[0] - 2.0).abs() < 1e-12);
assert!((mesh.measure(1)? - 1.0).abs() < 1e-12);
# Ok(())
# }
```

## Pseudo-inverses

Gradients need the inverse map. For cells of full dimension, `inverse_jacobians(&rule)`
returns $J^{-1}$, which exists only on such meshes. `pseudo_inverse_jacobians(&rule)` returns
$K = (J^{\mathsf T} J)^{-1} J^{\mathsf T}$ on every mesh: it equals $J^{-1}$ when $J$ is square,
and on a surface it
maps a vector in space to the reference coordinates of its tangential part. $K$ has $t$
rows and $g$ columns, stored column by column, so on the host `k[γ]` is column $\gamma$. The
snippet below computes both for a triangle in 3D:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
// One triangle in 3D with the vertices (0, 0, 0), (2, 0, 0) and (0, 1, 1).
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: SurfaceMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0], None)
    .cell_block(triangle, &[0, 1, 2], None)
    .build()?;
let rule = QuadRule::for_grid(&mesh, 1)?;

// J has the columns (2, 0, 0) and (0, 1, 1); det(JᵀJ) = 8.
let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
assert!((detj[0] - 8.0f64.sqrt()).abs() < 1e-12);

// K = (JᵀJ)⁻¹Jᵀ has the rows (1/2, 0, 0) and (0, 1/2, 1/2); k[g] is column g.
let k = mesh.pseudo_inverse_jacobians(&rule)?.blocks[0].to_host_cell_major()?;
let entries: Vec<f64> = k[0].iter().flatten().copied().collect();
let want = [0.5, 0.0, 0.0, 0.5, 0.0, 0.5];
assert!(entries.iter().zip(want).all(|(a, b)| (a - b).abs() < 1e-12));
# Ok(())
# }
```

For the planar triangle of [Jacobians](#jacobians), `inverse_jacobians` returns the columns
$(1/2, 0)$ and $(-1/2, 1)$. On a surface mesh `inverse_jacobians` does not exist, and calling it
is a compile error.

## Normals

On a mesh of codimension 1, a surface in 3D or a curve in 2D, `normals(&rule)` returns the unit
normal at every point. On a surface it is $n = s\,(J_0 \times J_1) / |J_0 \times J_1|$, with the
columns $J_0$, $J_1$ of the Jacobian. The sign $s$ is $-1$ for a cell whose orientation flag is
set and $+1$ otherwise; `orient` sets the flags so that neighbouring normals agree, as
[Orientation](tags-submeshes.md#orientation) shows. For cells of full dimension,
`facet_normals` returns the outward unit normal of every facet of every cell, at the points of
a [`FacetQuadRule`](../api/tetraloma/struct.FacetQuadRule.html):

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;

let client = cubecl::Device::cpu()?.client();
let triangle = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
let h = 0.5f64.sqrt();

// surface: the triangle (0, 0, 0), (2, 0, 0), (0, 1, 1) of the previous snippet.
# let surface: SurfaceMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0, 1.0], None)
#     .cell_block(triangle, &[0, 1, 2], None)
#     .build()?;
let rule = QuadRule::for_grid(&surface, 1)?;
let n = surface.normals(&rule)?.blocks[0].to_host_cell_major()?;
// J₀ × J₁ = (2, 0, 0) × (0, 1, 1) = (0, −2, 2).
assert!(close(&n[0], &[0.0, -h, h]));

// planar: the triangle (0, 0), (2, 0), (1, 1) of the Jacobians.
# let planar: PlanarMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[0.0, 0.0, 2.0, 0.0, 1.0, 1.0], None)
#     .cell_block(triangle, &[0, 1, 2], None)
#     .build()?;
let facet_rule = FacetQuadRule::for_grid(&planar, 1)?; // one point per edge
let n = planar.facet_normals(&facet_rule)?.blocks[0].to_host_cell_major()?;
// The edges (0, 1), (0, 2) and (1, 2), in the order of the reference triangle.
assert_eq!(n.len(), 3);
assert!(close(&n[0], &[0.0, -1.0]));
assert!(close(&n[1], &[-h, h]));
assert!(close(&n[2], &[h, h]));
# Ok(())
# }
```

A curve in 2D has the normal $n = s\,(J_{10}, -J_{00}) / |J_0|$ of its tangent $J_0$, which points
outward on a closed curve that runs counter-clockwise. `normals` exists only on meshes of
codimension 1, and `facet_normals` only on meshes of codimension 0; the compiler rejects the
other calls.

## Measures

`measure(degree)` returns the total length, area or volume of the owned cells, with Gauss rules
exact for polynomials of the given degree. It sums $\det J$ times the weights on the device and
finishes the sum in `f64` on the host, so it is accurate even for an `f32` mesh. On a
distributed mesh it sums over all ranks, and every rank receives the total. For the two
triangles above it returns $1$ and $\sqrt{2}$.

The program `geometric_factors` computes all five factors on the unit sphere as 20 curved
triangles of degree $p = 1$, $4$ and $8$, at a Gauss rule of degree $2p$:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/geometric_factors.rs:factors}}
```

It checks that every normal is a unit vector pointing outward, and compares the measure with
the area $4\pi$ of the sphere:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/geometric_factors.rs:area}}
```

The error of the area falls from about $0.24$ at $p = 1$, the flat icosahedron, to below
$10^{-4}$ at $p = 8$.
[Quadrature on a mesh](quadrature.md#high-order-geometry) follows this convergence on a mesh
you build yourself.
