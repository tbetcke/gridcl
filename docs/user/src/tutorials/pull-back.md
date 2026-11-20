# Pulling back points

The geometric factors map reference points to physical points. The pull-back goes the other
way: given a physical point $x$ and a cell, it finds the reference point $\xi$ with
$X(\xi) = x$. You need it to locate points in a mesh, and to interpolate a field of one mesh at
the points of another. This tutorial pulls back points of a curved mesh, checks the result,
and shows what happens to points outside the cell. The program `pull_back` holds every step:

```sh
cargo run --release -p tetraloma-examples --example pull_back
```

## Points to reference coordinates

[`Mesh::pull_back`](../api/tetraloma/struct.Mesh.html) takes physical points in the layout
that `physical_points` returns: one field per cell block, with a fixed number of points per
cell. Point $q$ of cell $c$ is pulled back with respect to cell $c$. The method returns a
[`PullBack`](../api/tetraloma/struct.PullBack.html) with two fields of the same layout: the
reference points `reference`, and the size `last_step` of the last Newton step at each point.

The simplest test pulls back the points that `physical_points` made. The snippet below does
this for the centres of the four cells of the unit square of
[From arrays](build-mesh.md#from-arrays), which all come back to the centre of the reference
square:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::unit_square;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = unit_square(&client)?;
let rule = QuadRule::for_grid(&mesh, 1)?; // one point: the centre of each cell
let x = mesh.physical_points(&rule)?;
let pb = mesh.pull_back(&x, &PullBackOptions::default())?;

// For inspection: four cells, one point each, at ξ = (1/2, 1/2).
let xi = pb.reference.blocks[0].to_host_cell_major()?;
assert_eq!(xi.len(), 4);
assert!(xi.iter().all(|p| (p[0] - 0.5).abs() < 1e-12 && (p[1] - 0.5).abs() < 1e-12));
# Ok(())
# }
```

On a curved cell $X$ is a polynomial, and the pull-back solves $X(\xi) = x$ by Newton's
method. It starts at the centroid of the reference cell's vertices and repeats
$\xi \leftarrow \xi + K(\xi)\,(x - X(\xi))$, with the pseudo-inverse $K$ of the Jacobian. On a
surface or a curve in space this is the Gauss–Newton method: it finds the point of the cell
closest to $x$. [The pull-back](../concepts/pull-back.md#the-iteration) derives the iteration.
The program pulls back the Gauss points of the curved quarter annulus of degree 3:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/pull_back.rs:curved}}
```

## Checking the result

A point has converged when its last step $|\Delta\xi|_\infty$ is at most the tolerance of the
[`PullBackOptions`](../api/tetraloma/struct.PullBackOptions.html). The default tolerance is
$10^{-13}$ in `f64` and $10^{-5}$ in `f32`, and the default number of iterations is 16. Every
point runs all iterations; converged points keep their value, so the result does not depend on
how the device groups points into vectors. The program compares each $\xi$ with the rule's
point $\xi_q$:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/pull_back.rs:check}}
```

For strongly curved cells you may need more iterations, and for a coarse search a looser
tolerance. A point that has not converged shows a `last_step` above the tolerance, and a step
that is not a finite number counts as the largest value of the scalar type. With no iterations
at all, nothing converges:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
use tetraloma::prelude::*;
use tetraloma_examples::build_mesh::unit_square;

let client = cubecl::Device::cpu()?.client();
let mesh: PlanarMesh<f64> = unit_square(&client)?;
let x = mesh.physical_points(&QuadRule::for_grid(&mesh, 1)?)?;

let none = PullBackOptions { tolerance: Some(1e-10), max_iterations: 0 };
let pb = mesh.pull_back(&x, &none)?;
let steps = pb.last_step.blocks[0].to_host_cell_major()?;
assert!(steps.iter().all(|&s| s > 1e-10));
# Ok(())
# }
```

Check `last_step` before you use a reference point: an unconverged $\xi$ is an iterate, not an
answer.

## Points outside a cell

A physical point that lies outside its cell still has a reference point, outside the reference
cell, as long as the map $X$ extends to it. To find out whether $x$ lies in the cell, you pull
it back and test $\xi$ against the reference cell. For a quadrilateral the test is
$0 \le \xi_0, \xi_1 \le 1$; for a triangle it is $\xi_0, \xi_1 \ge 0$ and $\xi_0 + \xi_1 \le 1$.
[Reference cells and maps](../concepts/reference-cells.md#reference-cells) lists the reference
cells.

The iteration confines every iterate to the box $[-0.1, 1.1]^t$, with $t$ the topological dimension, which holds every
reference cell with a margin of $0.1$. A point whose reference point lies outside this box
does not converge: its `last_step` stays above the tolerance. The program shows both cases with
two unit squares side by side:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/pull_back.rs:two_squares}}
```

It pulls back the points of copies of the squares moved by $0.25$ and $0.5$ along $x$. Each
point then has the exact reference point $\xi_q + (\text{shift}, 0)$. The Gauss points have
$\xi_0 \approx 0.21$ or $0.79$. A shift of $0.25$ moves them to $0.46$, inside, and $1.04$,
outside but within the box. A shift of $0.5$ moves the second ones to $1.29$, beyond the box:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/pull_back.rs:outside}}
```

It prints four points inside and four outside for the first shift, and four inside and four not
converged for the second. Such a point lies in another cell or outside the mesh, so a point
search tries the next candidate cell. [Convergence and failure](../concepts/pull-back.md#convergence-and-failure)
says more about when the iteration fails.

The points to pull back need not come from `physical_points`. Any
[`BlockFields`](../api/tetraloma/struct.BlockFields.html) of the same layout works, such as
one that a kernel of yours fills with the points of another mesh; [Writing kernels](kernels.md)
shows how kernels read and write these fields.
