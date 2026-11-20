# Precision

tetraloma computes in the scalar type of the mesh, `f32` or `f64`, and both are first-class:
every kernel is tested in both. This chapter explains what changes between the two, how
accurate the results are, and which results you can expect to be identical from run to run.

## f32 and f64

The scalar type is the first parameter of [`Mesh`](../api/tetraloma/struct.Mesh.html). It sets
the type of the node coordinates on the device, of every geometric factor, and of the
arithmetic in the kernels. Some work always happens in f64, whatever the mesh's type:

- the input of the mesh builder, which takes coordinates as `f64`;
- the quadrature points and weights, and the tabulation of the basis functions, which are
  computed on the host in f64 and narrowed to the mesh's type when they are uploaded;
- the mesh's origin (see [Tolerances](#tolerances));
- the final sums of `measure` and of the orientation integrals.

f32 halves the memory of a mesh and its fields, and many GPUs compute in f32 much faster than in
f64. Not every device supports f64: Metal has no f64 at all. Building an f64 mesh on a device
without f64 fails with `GridError::UnsupportedPrecision` (see
[Compute devices](../devices.md#f64-support)). The CPU runtime supports both.

Write code that works in both precisions as code generic over `F: Real`. Host arithmetic on
values of type `F` goes through `widen()` to `f64` and `narrow()` back: the arithmetic methods
that `F` gets from CubeCL are meant for device code, not for the host. The example runs the
same check of the triangle $(1, 0, 0)$, $(3, 0, 0)$, $(2, 2, 2)$ from
[Geometric factors](geometric-factors.md) in both precisions, each with its own tolerance.

```rust
use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// det J, the normal and the area of the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2) in the
/// precision F, against the exact values, within the relative tolerance `tol`.
fn check<F: Real>(client: &Client, tol: f64) -> tetraloma::Result<()> {
    let tri = CoordinateElement::<D2> {
        cell: Cell2::Triangle,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    let mesh: SurfaceMesh<F> = MeshBuilder::new(client)
        .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
        .cell_block(tri, &[0, 1, 2], None)
        .build()?;
    let rule = QuadRule::for_grid(&mesh, 1)?;
    let close = |a: F, b: f64| (a.widen() - b).abs() <= tol * b.abs().max(1.0);

    let detj = mesh.detj(&rule)?.blocks[0].to_host_cell_major()?;
    assert!(close(detj[0], 4.0 * 2f64.sqrt()));

    let n = mesh.normals(&rule)?.blocks[0].to_host_cell_major()?;
    let r = 0.5f64.sqrt();
    assert!(close(n[0][0], 0.0) && close(n[0][1], -r) && close(n[0][2], r));

    // measure returns f64 in both precisions.
    let area: f64 = mesh.measure(1)?;
    assert!((area - 2.0 * 2f64.sqrt()).abs() <= tol * 2.0 * 2f64.sqrt());
    Ok(())
}

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
check::<f32>(&client, 1e-5)?;
check::<f64>(&client, 1e-12)?;
# Ok(()) }
```

## Tolerances

A result in f32 carries a relative rounding error of about $10^{-7}$ per operation, and one in
f64 about $10^{-16}$. tetraloma's tests compare results with exact values to a relative
tolerance of $10^{-5}$ in f32 and $10^{-12}$ in f64, which leaves room for the accumulated
rounding of a geometric factor or a quadrature sum. Use the same tolerances, scaled by the size
of your values, when you compare results.

Some algorithms take a tolerance of their own. The [pull-back](pull-back.md#the-iteration)
stops refining a point when its step is at most $10^{-13}$ in f64 or $10^{-5}$ in f32, unless
you set `PullBackOptions::tolerance`. Very high geometry degrees amplify rounding, so the
attainable accuracy falls somewhat as the degree grows, particularly in f32.

Rounding in f32 hurts most when coordinates are large compared with the cells, as for a small
part far from the origin of a large model: at $10^5$, neighbouring f32 numbers are about
$0.008$ apart. tetraloma therefore stores the coordinates relative to an origin, the centre of
the mesh's bounding box. The origin is kept in f64 on the host; the builder subtracts it in f64
before narrowing, and `physical_points`, `read_node`, the readers and the writers add it back.
Differences of nearby nodes, and with them the Jacobians, keep nearly full f32 accuracy. A
submesh keeps its parent's origin. When you read a Gmsh file into an f32 mesh, the reader
narrows the coordinates and logs one warning per file.

The example builds a small triangle with edges of length 0.1 at $x = 10^5$, in f32. Stored as
absolute f32 coordinates, its edge along $x$ would come out as $0.1015625$, off by 1.6 per cent;
stored relative to the origin, its area is correct to the f32 tolerance.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

let tri = CoordinateElement::<D2> {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let far: PlanarMesh<f32> = MeshBuilder::new(&client)
    .nodes(&[100_000.1, 0.0, 100_000.2, 0.0, 100_000.1, 0.1], None)
    .cell_block(tri, &[0, 1, 2], None)
    .build()?;

// The origin is the centre of the bounding box, in f64.
let [ox, oy] = far.origin();
assert!((ox - 100_000.15).abs() < 1e-9 && (oy - 0.05).abs() < 1e-12);

// The area 0.005, to the f32 tolerance.
let area = far.measure(1)?;
assert!((area - 0.005).abs() <= 1e-5 * 0.005);
# Ok(()) }
```

## Determinism

tetraloma's results do not depend on how a device schedules its work, so the same program on
the same device gives the same bits every time. The rules behind this:

- **No float atomics.** Sums of floating-point numbers, such as `measure` and the orientation
  integrals, run in a fixed order on the device, so rounding is the same on every run.
- **Element-wise kernels.** Geometric factors, the pull-back and the other kernels that compute
  each cell independently give bitwise identical results whether the CPU runtime processes one
  cell or many cells per vector instruction. On GPUs they agree within the tolerances above.
- **Reductions.** A floating-point reduction combines partial results in a fixed order. The
  result may differ in the last bits between vector sizes, but tetraloma fixes the vector size
  per kernel family and device class, so it is reproducible on a given device. Integer
  reductions and scans are exact.
- **The pull-back** runs a fixed number of iterations for every point, so no point's result
  depends on its neighbours in a vector.
- **Topology.** Entity creation, the numbering of entities and the permutation bits do not
  depend on the number of ranks. Owners of shared vertices and entities are chosen from origin
  ids, so for a given input and partition they are the same on every run.

Results on different devices, for example the CPU runtime and a GPU, agree within the
tolerances but not bit for bit: compilers and hardware may contract or reorder operations
differently.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let tri = CoordinateElement::<D2> { cell: Cell2::Triangle, degree: 1, family: NodeFamily::Equispaced };
// mesh: the triangle (1, 0, 0), (3, 0, 0), (2, 2, 2), as above
# let mesh: SurfaceMesh<f32> = MeshBuilder::new(&client)
#     .nodes(&[1.0, 0.0, 0.0, 3.0, 0.0, 0.0, 2.0, 2.0, 2.0], None)
#     .cell_block(tri, &[0, 1, 2], None)
#     .build()?;
// The same computation twice gives the same bits.
let a = mesh.measure(4)?;
let b = mesh.measure(4)?;
assert_eq!(a.to_bits(), b.to_bits());

let rule = QuadRule::for_grid(&mesh, 4)?;
let j1 = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
let j2 = mesh.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
assert_eq!(j1, j2);
# Ok(()) }
```
