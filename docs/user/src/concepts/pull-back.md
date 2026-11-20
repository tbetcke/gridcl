# The pull-back

The map $X$ of a cell takes reference coordinates to physical ones. The pull-back goes the
other way: given a physical point $x$ and a cell, it finds the reference point $\xi$ with
$X(\xi) = x$. You need it to evaluate a finite element function at an arbitrary point, to
locate points in a mesh, and to interpolate from one mesh to another. This chapter explains the
problem, how tetraloma starts and runs its iteration, and how to tell whether it worked. The
tutorial [Pulling back points](../tutorials/pull-back.md) shows the calls.

## The problem

For an affine cell, $X(\xi) = x_0 + J \xi$ with a constant $J$, and the pull-back is one linear
solve, $\xi = J^{-1}(x - x_0)$. A bilinear quadrilateral or a curved cell of degree $p \ge 2$
has a nonlinear map, and the equation $X(\xi) = x$ has no closed-form solution in general. It
is solved by iteration.

When the cell is a curve or a surface ($t < g$), the equation has $g$ components but only $t$
unknowns. A point off the surface has no exact preimage, so tetraloma solves the least-squares
problem instead:

$$
\min_{\xi}\ \tfrac12 \left| X(\xi) - x \right|^2 .
$$

Its solution is the reference point of the point on the cell closest to $x$ (locally), and for
a point on the cell it is the exact preimage. In codimension 0 the minimum is 0 and the two
formulations agree.

`Mesh::pull_back` takes physical points per cell and point, in the layout that
`physical_points` returns, and pulls each one back through its own cell. Each point is solved
independently, so a device kernel handles many cells at once.

## The affine start

The iteration starts at the centroid of the reference cell's vertices, $\xi^{(0)} = \bar\xi$:
$(1/3, 1/3)$ on the triangle, $(1/2, 1/2)$ on the quadrilateral, $(1/4, 1/4, 1/4)$ on the
tetrahedron, and so on. The first step replaces the map by its affine approximation at that
point,

$$
X(\xi) \approx X(\bar\xi) + J(\bar\xi)\,(\xi - \bar\xi),
$$

and solves the approximation exactly (in the least-squares sense when $t < g$):

$$
\xi^{(1)} = \bar\xi + K(\bar\xi)\,\bigl(x - X(\bar\xi)\bigr),
\qquad K = \left(J^{\mathsf T} J\right)^{-1} J^{\mathsf T},
$$

with $K$ the [pseudo-inverse](geometric-factors.md#the-pseudo-inverse) of $J$. On an affine
cell the approximation is the map itself, so this first step already lands on the answer, and
every later step is zero up to rounding. On a curved cell it gives a starting guess that is
already close whenever the cell is close to affine.

## The iteration

Every step repeats the affine solve at the current point:

$$
\Delta^{(n)} = K\bigl(\xi^{(n)}\bigr)\,\bigl(x - X(\xi^{(n)})\bigr),
\qquad
\xi^{(n+1)} = \operatorname{clamp}\bigl(\xi^{(n)} + \Delta^{(n)}\bigr).
$$

In codimension 0 this is Newton's method for $X(\xi) = x$; when $t < g$ it is the Gauss–Newton
method for the least-squares problem. Newton's method converges quadratically near a solution,
roughly doubling the number of correct digits per step. Gauss–Newton does the same for points
on the cell and converges more slowly for points off it.

The clamp keeps each coordinate of $\xi$ in $[-0.1, 1.1]$: every reference cell lies in
$[0, 1]^t$, and the box adds a margin of 0.1 on each side. Without it, a step on a strongly
curved cell can jump to a second, spurious solution of $X(\xi) = x$ far outside the cell. On
pyramids, the basis functions are evaluated with $\xi_2$ at most $1 - 2^{-10}$, away from the
apex, where the rational functions are singular.

Each point stops changing ("freezes") once $|\Delta^{(n)}|_\infty$, the largest component of
its step, is at most the tolerance. The kernel still runs all `max_iterations` steps for every
point, with frozen points left unchanged by arithmetic masks. Because no point decides when the
loop ends, the result is the same however many points a vector instruction handles together.

[`PullBackOptions`](../api/tetraloma/struct.PullBackOptions.html) sets the two parameters:

- `tolerance`: `None` means $10^{-13}$ for f64 meshes and $10^{-5}$ for f32 meshes;
- `max_iterations`: 16 by default. Strongly curved cells may need more, and an affine mesh
  needs only 2.

The example pulls back the Gauss points of a triangle with two iterations. The first step
lands on each point, so the second step is below the tolerance and the result is exact to
rounding.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

// An affine triangle (1, 0), (3, 1), (1, 2).
let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let mesh: PlanarMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[1.0, 0.0, 3.0, 1.0, 1.0, 2.0], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;

// The physical Gauss points of degree 4, pulled back with two iterations.
let rule = QuadRule::for_grid(&mesh, 4)?;
let x = mesh.physical_points(&rule)?;
let opts = PullBackOptions { tolerance: None, max_iterations: 2 };
let pb = mesh.pull_back(&x, &opts)?;

let want = rule.get(Cell2::Triangle)?.points();
let xi = pb.reference.blocks[0].to_host_cell_major()?;
for (q, p) in xi.iter().enumerate() {
    assert!((p[0] - want[2 * q]).abs() < 1e-12);
    assert!((p[1] - want[2 * q + 1]).abs() < 1e-12);
}
// The second step is zero up to rounding: every point has converged.
let steps = pb.last_step.blocks[0].to_host_cell_major()?;
assert!(steps.iter().all(|&s| s <= 1e-13));
# Ok(()) }
```

## Convergence and failure

The result ([`PullBack`](../api/tetraloma/struct.PullBack.html)) has two fields, both with one
value per cell and point: `reference`, the computed $\xi$, and `last_step`, the size
$|\Delta|_\infty$ of the point's last step. A point has converged when its last step is at most
the tolerance. If it is larger, the iteration did not settle within `max_iterations`, and
$\xi$ should not be trusted. A step with a non-finite component, and every step when
`max_iterations` is 0, counts as the largest finite value of the scalar type.

A converged $\xi$ need not lie in the reference cell. A physical point slightly outside the
cell has a preimage slightly outside the reference cell, and the iteration finds it as long as
it lies inside the box $[-0.1, 1.1]^t$. To test whether a point lies in a cell, check that the
iteration converged and that $\xi$ satisfies the reference cell's inequalities, for example
$\xi_0 \ge 0$, $\xi_1 \ge 0$ and $\xi_0 + \xi_1 \le 1$ on the triangle, within a small
tolerance.

A point whose preimage lies outside the box does not converge. The clamp pulls each iterate back
to the box, the next step points out again, and the step never shrinks. Its `last_step` stays
large, which marks the point as outside. A point far from the cell therefore shows up as not
converged.

The example pulls back the centroid of a triangle three and four times the size of the
reference triangle through the reference triangle itself, whose map is the identity. The
preimages are $(1, 1)$ and $(4/3, 4/3)$. The first lies outside the triangle but inside the
box, and it converges. The second lies outside the box: the iterates stop at the box's edge,
$1.1$, and the last step stays near $4/3 - 1.1 \approx 0.23$.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
// A triangle (0, 0), (s, 0), (0, s).
let triangle = |s: f64| -> tetraloma::Result<PlanarMesh<f64>> {
    MeshBuilder::new(&client)
        .nodes(&[0.0, 0.0, s, 0.0, 0.0, s], None)
        .cell_block(tri, &[0, 1, 2], None)
        .build()
};
let reference = triangle(1.0)?;
let rule = QuadRule::for_grid(&reference, 1)?;

// The centroid (s/3, s/3) of the larger triangle, pulled back through the reference triangle.
for (s, inside_box) in [(3.0, true), (4.0, false)] {
    let x = triangle(s)?.physical_points(&rule)?;
    let pb = reference.pull_back(&x, &PullBackOptions::default())?;
    let xi = pb.reference.blocks[0].to_host_cell_major()?[0];
    let step = pb.last_step.blocks[0].to_host_cell_major()?[0];
    if inside_box {
        // Converged to (1, 1), outside the triangle: ξ0 + ξ1 = 2 > 1.
        assert!(step <= 1e-13);
        assert!((xi[0] - 1.0).abs() < 1e-12 && (xi[1] - 1.0).abs() < 1e-12);
    } else {
        // Not converged: the iterates are clamped to 1.1 and the step stays near 0.23.
        assert!(step > 0.2 && step < 0.25);
        assert!((xi[0] - 1.1).abs() < 1e-6 && (xi[1] - 1.1).abs() < 1e-6);
    }
}
# Ok(()) }
```
