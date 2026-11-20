# Typed dimensions

The topological dimension $t$ and the geometric dimension $g$ decide the shape of almost
everything you compute from a mesh: a Jacobian is $g \times t$, normals exist only when
$g = t + 1$, and inverse Jacobians only when $g = t$. tetraloma makes the pair $(t, g)$ part of
the mesh's type, so that these rules are checked by the compiler. The cell type and the
geometry degree stay run-time values, stored per block, so one mesh type covers meshes that mix
cell types.

## The mesh type

A mesh has the type
[`Mesh<F, TD, GD, C>`](../api/tetraloma/struct.Mesh.html), with four parameters:

- `F`, the scalar type of the geometry: `f32` or `f64` (the trait
  [`Real`](../api/tetraloma/trait.Real.html));
- `TD`, the topological dimension, one of the marker types `D1`, `D2`, `D3`;
- `GD`, the geometric dimension, one of `D1`, `D2`, `D3`, with `TD` at most `GD`;
- `C`, the communicator: [`Serial`](../api/tetraloma/struct.Serial.html) (the default) for a
  mesh on one process, or an MPI communicator for a distributed mesh.

The markers carry their value as a constant, `D3::N == 3`. Of the nine pairs `(TD, GD)` only
the six with $t \le g$ are valid; the trait
[`EmbeddedIn`](../api/tetraloma/trait.EmbeddedIn.html) is implemented for exactly these, and
every method of `Mesh` requires it. Four of them have names:

| Type | $t$ | $g$ | Example |
| --- | --- | --- | --- |
| `Mesh<F, D1, D1>` | 1 | 1 | a partition of an interval |
| `Mesh<F, D1, D2>` | 1 | 2 | a curve in the plane |
| `CurveMesh<F>` = `Mesh<F, D1, D3>` | 1 | 3 | a wire in space |
| `PlanarMesh<F>` = `Mesh<F, D2, D2>` | 2 | 2 | a triangulated planar domain |
| `SurfaceMesh<F>` = `Mesh<F, D2, D3>` | 2 | 3 | a triangulated sphere |
| `VolumeMesh<F>` = `Mesh<F, D3, D3>` | 3 | 3 | a tetrahedral mesh of a solid |

The topological dimension also restricts the cell types. A cell block of a `D2` mesh holds a
[`Cell2`](../api/tetraloma/enum.Cell2.html), which is a triangle or a quadrilateral; a `D3`
mesh holds tetrahedra, hexahedra, prisms or pyramids ([`Cell3`](../api/tetraloma/enum.Cell3.html)).
A tetrahedron in a surface mesh cannot be written down.

The dimensions also fix the host types of the geometric factors. A surface mesh's Jacobian is
`[[F; 3]; 2]` per point (two columns of three entries, see
[Geometric factors](geometric-factors.md#the-jacobian)) and its pseudo-inverse
`[[F; 2]; 3]`. A mix-up between a $3 \times 2$ and a $2 \times 3$ array is a type error.

## Codimension

The codimension is $g - t$. It decides which normal-like quantities exist:

- **Codimension 0** ($t = g$): planar and volume meshes, and intervals on a line. $J$ is
  square and invertible on a valid cell, and every cell has outward facet normals. The trait
  [`Codim0`](../api/tetraloma/trait.Codim0.html) marks these pairs.
- **Codimension 1** ($t = g - 1$): surfaces in 3D and curves in the plane. The cell has a
  unit normal, defined up to sign, and the mesh can be oriented. The trait
  [`Codim1`](../api/tetraloma/trait.Codim1.html) marks these pairs.
- **Codimension 2**: curves in 3D. A curve in space has a whole plane of normal directions at
  each point, so tetraloma offers no normal. Its Jacobian is the tangent, and its
  pseudo-inverse $K = J^{\mathsf T} / |J|^2$.

In every codimension the pseudo-determinant measures length, area or volume, and the
pseudo-inverse turns reference derivatives into tangential ones.

## Methods by dimension

Methods exist only for the dimensions where they make sense. Each group of methods sits in an
`impl` block with the matching trait bound:

| Method | Exists for | Why |
| --- | --- | --- |
| `jacobians`, `detj`, `pseudo_inverse_jacobians`, `measure`, `pull_back` | every mesh | defined for any $t \le g$ |
| `inverse_jacobians`, `facet_normals` | codimension 0 | $J^{-1}$ and the outward facet normals need a square $J$ |
| `normals`, `orient` | codimension 1 | a unique normal direction needs $g = t + 1$ |
| `boundary`, `facet_submesh` | $t \ge 2$ | the result is a mesh of dimension $t - 1 \ge 1$ |
| `edge_submesh` | volume meshes | a set of edges of a solid, as a `CurveMesh` |

`boundary()` returns a mesh one dimension lower: the boundary of a `VolumeMesh` is a
`SurfaceMesh`, and that of a `PlanarMesh` a `Mesh<F, D1, D2>`. A curve mesh has no
`boundary()`, because points do not form a mesh.

Calling a method where it does not exist is a compile error, not a run-time error. These lines
do not compile:

```text
let n = volume.normals(&rule)?;            // a VolumeMesh has no normals: no `Codim1<D3>` for D3
let k = surface.inverse_jacobians(&rule)?; // a SurfaceMesh has no J⁻¹: no `Codim0<D3>` for D2
let b = curve.boundary()?;                 // a curve's boundary would be points
```

Code that works for every dimension is generic over the parameters, with the same bounds. The
tutorial [Writing kernels](../tutorials/kernels.md#generic-algorithms) shows how.

The example builds a curve in 3D from two intervals, computes its length and its Jacobian (one
column of three entries), and checks the dimension constants.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
let client = cubecl::Device::cpu()?.client();

assert_eq!((CurveMesh::<f64>::TDIM, CurveMesh::<f64>::GDIM), (1, 3));
assert_eq!((SurfaceMesh::<f64>::TDIM, SurfaceMesh::<f64>::GDIM), (2, 3));

// Two straight intervals: (0, 0, 0) to (3, 4, 0), of length 5, then up to (3, 4, 12).
let seg = CoordinateElement::<D1> {
    cell: Cell1::Interval,
    degree: 1,
    family: NodeFamily::Equispaced,
};
let curve: CurveMesh<f64> = MeshBuilder::new(&client)
    .nodes(&[0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 3.0, 4.0, 12.0], None)
    .cell_block(seg, &[0, 1, 1, 2], None)
    .build()?;
assert!((curve.measure(1)? - 17.0).abs() < 1e-12);

// One column of three entries per point: the first interval's J is (3, 4, 0).
let rule = QuadRule::for_grid(&curve, 1)?;
let j: Vec<[[f64; 3]; 1]> = curve.jacobians(&rule)?.blocks[0].to_host_cell_major()?;
let want = [3.0, 4.0, 0.0];
for g in 0..3 {
    assert!((j[0][0][g] - want[g]).abs() < 1e-12);
}
# Ok(()) }
```

## AnyMesh

The dimensions of a mesh in a file are known only when the file is read. A reader therefore
returns an [`AnyMesh`](../api/tetraloma/enum.AnyMesh.html): an enum with one variant per valid
pair (`Curve1`, `Curve2`, `Curve3`, `Planar`, `Surface`, `Volume`), each holding a typed mesh.
You fix the dimensions once, near the top of your program, in one of two ways:

- convert to the type you expect with `TryFrom`, for example
  `SurfaceMesh::<f64>::try_from(any)?`. A mesh of other dimensions gives
  `GridError::DimensionMismatch`, with the expected and found pairs.
- handle every variant with the macro
  [`with_mesh!`](../api/tetraloma/macro.with_mesh.html), which runs one expression once per
  variant with a typed mesh. The expression must compile for all six types, so it may call
  only the methods that every mesh has.

`AnyMesh::as_ref` gives a borrowed view, [`AnyMeshRef`](../api/tetraloma/enum.AnyMeshRef.html),
so that `with_mesh!` and the writers never move a mesh. The tutorial
[Reading and writing files](../tutorials/read-write.md#fixing-the-dimensions) applies both to a
Gmsh file.

```rust
use tetraloma::prelude::*;
# fn main() -> Result<(), Box<dyn std::error::Error>> {
# let client = cubecl::Device::cpu()?.client();
# let seg = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
// curve: the two intervals of length 5 and 12 in 3D, as above
# let curve: CurveMesh<f64> = MeshBuilder::new(&client)
#     .nodes(&[0.0, 0.0, 0.0, 3.0, 4.0, 0.0, 3.0, 4.0, 12.0], None)
#     .cell_block(seg, &[0, 1, 1, 2], None)
#     .build()?;
let any = AnyMesh::Curve3(curve);
assert_eq!((any.tdim(), any.gdim()), (1, 3));

// One expression for every variant.
let length = tetraloma::with_mesh!(any.as_ref(), m => m.measure(1))?;
assert!((length - 17.0).abs() < 1e-12);

// Back to the typed mesh.
let curve = CurveMesh::<f64>::try_from(any)?;
assert_eq!(curve.tdim(), 1);
# Ok(()) }
```
