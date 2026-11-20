# A first program

This chapter walks through a complete program. It opens a device, builds a mesh of a rectangle
from two arrays, computes geometric data at quadrature points, checks it against values you can
work out by hand, and writes the mesh to a file that ParaView can show.

## The program

The program is the function `run` in `tetraloma-examples/src/first_program.rs` of the repository.
It is generic over the scalar type `F`, so the same code runs in `f32` and in `f64`:

```text
pub fn run<F: Real>(client: &Client, out_dir: &Path) -> tetraloma::Result<()>
```

The mesh is the rectangle $[0, 2] \times [0, 1]$, divided into $4 \times 2$ squares, with every
square cut along a diagonal into two triangles: 16 triangles and 15 nodes in all.

```text
(0,1) +----+----+----+----+ (2,1)
      |   /|   /|   /|   /|
      |  / |  / |  / |  / |
      | /  | /  | /  | /  |
      |/   |/   |/   |/   |
      +----+----+----+----+
      |   /|   /|   /|   /|
      |  / |  / |  / |  / |
      | /  | /  | /  | /  |
      |/   |/   |/   |/   |
(0,0) +----+----+----+----+ (2,0)
```

Every check in the program compares a computed value with an exact one within a tolerance:
$10^{-4}$ in `f32` and $10^{-11}$ in `f64`, ten times the rounding tolerances of
[Precision](../concepts/precision.md#tolerances). A failed check stops the program with a panic.

## Opening a device

All device memory and every kernel launch belong to a CubeCL *client*, the handle of one device.
You get it from a `cubecl::Device`. On the CPU runtime that is one line:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
assert_eq!(client.name(), "cpu");
# Ok(())
# }
```

`Device::cpu()` fails with `DeviceUnavailable` if the build does not link the CPU runtime, that is,
if the `cpu` feature of `cubecl` is off. The program's `main` opens the first device that the
build links, a GPU if there is one, and calls `run` with `f64` where the device supports it and
`f32` otherwise. [Compute devices](../devices.md#choosing-a-device-at-run-time) shows how to
choose the device yourself.

## Building a mesh

A mesh starts as two host arrays. The first holds the coordinates of the nodes, two numbers per
node in the plane, always in `f64`. The second holds the cells: three node indices per triangle,
counter-clockwise. The helper `rectangle` writes both for the grid of squares:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/first_program.rs:rectangle}}
```

A [`MeshBuilder`](../api/tetraloma/struct.MeshBuilder.html) takes these arrays and uploads the
mesh to the device:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/first_program.rs:mesh}}
```

The [`CoordinateElement`](../api/tetraloma/struct.CoordinateElement.html) says how each cell's
geometry is described: here a triangle of polynomial degree 1, that is, with straight sides and
its three vertices as nodes. A cell block holds cells of one type; a mesh with triangles and
quadrilaterals would call `cell_block` twice.

The type [`PlanarMesh<F>`](../api/tetraloma/type.PlanarMesh.html) is `Mesh<F, D2, D2>`: cells of
topological dimension 2 in a space of dimension 2. `build` checks the input, numbers the vertices
and nodes, and uploads everything to the device. It stores the coordinates relative to the centre
of the mesh's bounding box, here $(1, \tfrac12)$, which limits rounding errors in `f32`. Every
method that returns coordinates adds this origin back.

## Geometric data

Each triangle $K$ is the image of the reference triangle with vertices $(0,0)$, $(1,0)$ and $(0,1)$
under a map $x = \phi_K(\xi)$. Its Jacobian $J = \partial x / \partial \xi$ is a $2 \times 2$
matrix, and $\det J$ relates areas: an integral over the mesh becomes a sum over cells and
quadrature points $\xi_q$ with weights $w_q$,

$$
\int_\Omega f \, \mathrm{d}x \approx \sum_K \sum_q w_q \, f(\phi_K(\xi_q)) \, \det J_K(\xi_q).
$$

For $f = 1$ the sum is the area of the rectangle, 2, for any rule. The program computes the
physical points $\phi_K(\xi_q)$, the Jacobians and their determinants at the points of a Gauss
rule, copies them to the host to look at them, and forms this sum:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/first_program.rs:geometry}}
```

[`QuadRule::for_grid`](../api/tetraloma/struct.QuadRule.html) makes one rule for every cell type
of the mesh, here Gauss points exact for polynomials of degree 2 on the triangle. The results are
[`BlockFields`](../api/tetraloma/struct.BlockFields.html): one field per cell block, kept on the
device. `to_host_cell_major` copies one block's field to the host in the order cell by cell, point
by point. A program that goes on computing would leave the fields on the device and pass them to
its own kernels; the copies here are for inspection only.

The Jacobian of a point is stored by columns: `j[0]` is $\partial x / \partial \xi_1$ and `j[1]` is
$\partial x / \partial \xi_2$. Every triangle here is half of a square of side $\tfrac12$, so
$\det J = \tfrac14$ at every point: the reference triangle has area $\tfrac12$ and each mesh
triangle has area $\tfrac18$. The program checks this, checks that $\det J$ is the determinant of
the stored $J$, and checks that both the quadrature sum and
[`Mesh::measure`](../api/tetraloma/struct.Mesh.html) give the area 2. `measure` forms the same sum
on the device and finishes it in `f64` on the host. [Geometric data](../tutorials/geometric-data.md)
shows the other quantities, and [Geometric factors](../concepts/geometric-factors.md) defines them.

## Writing a VTU file

[`VtkWriter`](../api/tetraloma/vtk/struct.VtkWriter.html) writes a mesh in VTK's XML format for
unstructured grids, which ParaView and VisIt read:

```rust,ignore
{{#include ../../../../tetraloma-examples/src/first_program.rs:vtu}}
```

The writer copies the mesh to the host, adds the origin back to the coordinates and writes one
file. For curved cells of degree 2 or more it writes VTK's Lagrange cells. Fields can be attached
as point or cell data; [Reading and writing files](../tutorials/read-write.md) shows how, and how
to write Gmsh, PVTU and VTKHDF files.

## Running it

From a checkout of the repository, run the program with Cargo. The optional argument is the output
directory:

```sh
cargo run --release -p tetraloma-examples --example first_program -- out
```

It prints the size of the mesh and the device, the data at the first quadrature point and the
area, and writes `out/rectangle.vtu`. Without an argument it writes to
`target/tetraloma-examples/first_program`. With `--features metal` on a Mac, or `--features cuda`
on a machine with an NVIDIA GPU, the same program runs there.

In a project of your own, call the program's `run` the same way, or copy its body into your
`main`. This test runs it on the CPU runtime in both precisions:

```rust
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();
let out_dir = std::env::temp_dir().join(format!("tetraloma-book-first-{}", std::process::id()));
std::fs::create_dir_all(&out_dir)?;
tetraloma_examples::first_program::run::<f64>(&client, &out_dir)?;
tetraloma_examples::first_program::run::<f32>(&client, &out_dir)?;
assert!(out_dir.join("rectangle.vtu").exists());
std::fs::remove_dir_all(&out_dir)?;
# Ok(())
# }
```
