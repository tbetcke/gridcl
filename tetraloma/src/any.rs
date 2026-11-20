// Design: "API walkthrough", Listing B6.
//! The erased layer: `AnyMesh`, `AnyMeshRef`, `with_mesh!` and the `TryFrom` impls.

use tetraloma_device::{GridError, Real, Result};
use tetraloma_dims::{D1, D2, D3, Dim};
use tetraloma_parallel::{Comm, Serial};

use crate::Mesh;

// Design: "API walkthrough", Listing B6.
/// A mesh whose dimensions are known only at run time, as the Gmsh reader returns it.
///
/// Each variant holds the mesh of one valid pair (TD, GD). `try_into`, through the `TryFrom` impls
/// of the six mesh types, fixes the dimensions once and returns `DimensionMismatch` for another
/// variant. [`as_ref`](AnyMesh::as_ref) borrows the mesh for [`with_mesh!`](crate::with_mesh),
/// which runs generic code on any variant. [AnyMesh] explains the erased layer, and [Fixing the
/// dimensions] shows it after reading a file.
///
/// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
/// [Fixing the dimensions]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#fixing-the-dimensions
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // planar: a `PlanarMesh<f32>`, the unit square as two triangles; the tutorial "Building a mesh"
/// // shows how to make a mesh.
/// # let planar: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Triangle, 1, 1).build(&client)?;
/// let any = AnyMesh::Planar(planar);
/// assert_eq!((any.tdim(), any.gdim()), (2, 2));
/// // Fix the dimensions.
/// let planar: PlanarMesh<f32> = any.try_into()?;
/// assert_eq!(planar.cell_blocks()[0].index_map.num_local(), 2);
/// // Another mesh type is an error.
/// let r = SurfaceMesh::<f32>::try_from(AnyMesh::Planar(planar));
/// assert!(matches!(r, Err(GridError::DimensionMismatch { expected: (2, 3), found: (2, 2) })));
/// # Ok::<(), GridError>(())
/// ```
pub enum AnyMesh<F: Real, C: Comm = Serial> {
    /// A mesh of intervals on a line.
    Curve1(Mesh<F, D1, D1, C>),
    /// A mesh of intervals in the plane.
    Curve2(Mesh<F, D1, D2, C>),
    /// A mesh of intervals in space.
    Curve3(Mesh<F, D1, D3, C>),
    /// A mesh of planar cells in the plane.
    Planar(Mesh<F, D2, D2, C>),
    /// A mesh of planar cells in space.
    Surface(Mesh<F, D2, D3, C>),
    /// A mesh of volume cells in space.
    Volume(Mesh<F, D3, D3, C>),
}

// Design: "API walkthrough", Listing B6.
/// A borrowed [`AnyMesh`]: one reference per variant, so that [`with_mesh!`](crate::with_mesh) and
/// the writers never move or copy a mesh.
///
/// [`AnyMesh::as_ref`] makes one from an `AnyMesh`, and each variant wraps a reference to a typed
/// mesh. [AnyMesh] explains the erased layer.
///
/// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // planar: a `PlanarMesh<f32>`, the unit square in 2 × 2 quadrilaterals; the tutorial "Building
/// // a mesh" shows how to make a mesh.
/// # let planar: PlanarMesh<f32> = tetraloma_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
/// let cells = with_mesh!(AnyMeshRef::Planar(&planar), m => m.cell_blocks()[0].index_map.num_local());
/// assert_eq!(cells, 4);
/// # Ok::<(), GridError>(())
/// ```
pub enum AnyMeshRef<'a, F: Real, C: Comm = Serial> {
    /// A mesh of intervals on a line.
    Curve1(&'a Mesh<F, D1, D1, C>),
    /// A mesh of intervals in the plane.
    Curve2(&'a Mesh<F, D1, D2, C>),
    /// A mesh of intervals in space.
    Curve3(&'a Mesh<F, D1, D3, C>),
    /// A mesh of planar cells in the plane.
    Planar(&'a Mesh<F, D2, D2, C>),
    /// A mesh of planar cells in space.
    Surface(&'a Mesh<F, D2, D3, C>),
    /// A mesh of volume cells in space.
    Volume(&'a Mesh<F, D3, D3, C>),
}

impl<F: Real, C: Comm> AnyMesh<F, C> {
    /// The borrowed view of the mesh, for [`with_mesh!`](crate::with_mesh) and the writers.
    ///
    /// [AnyMesh] explains the erased layer.
    ///
    /// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // any: an `AnyMesh<f32>` that holds the unit cube as one hexahedron, as a reader returns a
    /// // mesh; the tutorial "Reading and writing files" shows how to read one.
    /// # let any = AnyMesh::Volume(tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?);
    /// let cells = with_mesh!(any.as_ref(), m => m.cell_blocks()[0].index_map.num_local());
    /// assert_eq!(cells, 1);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn as_ref(&self) -> AnyMeshRef<'_, F, C> {
        match self {
            AnyMesh::Curve1(m) => AnyMeshRef::Curve1(m),
            AnyMesh::Curve2(m) => AnyMeshRef::Curve2(m),
            AnyMesh::Curve3(m) => AnyMeshRef::Curve3(m),
            AnyMesh::Planar(m) => AnyMeshRef::Planar(m),
            AnyMesh::Surface(m) => AnyMeshRef::Surface(m),
            AnyMesh::Volume(m) => AnyMeshRef::Volume(m),
        }
    }

    /// The topological dimension of the mesh inside, as a number.
    ///
    /// [AnyMesh] explains the erased layer.
    ///
    /// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // any: an `AnyMesh<f32>` that holds the unit cube as one hexahedron, as a reader returns a
    /// // mesh; the tutorial "Reading and writing files" shows how to read one.
    /// # let any = AnyMesh::Volume(tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?);
    /// assert_eq!(any.tdim(), 3);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn tdim(&self) -> usize {
        crate::with_mesh!(self.as_ref(), m => m.tdim())
    }

    /// The geometric dimension of the mesh inside, as a number.
    ///
    /// [AnyMesh] explains the erased layer.
    ///
    /// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // any: an `AnyMesh<f32>` that holds the unit cube as one hexahedron, as a reader returns a
    /// // mesh; the tutorial "Reading and writing files" shows how to read one.
    /// # let any = AnyMesh::Volume(tetraloma_testing::cube_box(Cell3::Hexahedron, 1, 1).build::<f32, D3, D3>(&client)?);
    /// assert_eq!(any.gdim(), 3);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn gdim(&self) -> usize {
        crate::with_mesh!(self.as_ref(), m => m.gdim())
    }
}

// Design: "API walkthrough", Listing B6.
/// Runs an expression on the mesh inside an [`AnyMeshRef`], with the mesh's concrete type.
///
/// `with_mesh!(any.as_ref(), m => body)` matches the six variants. In each arm `m` is a
/// `&Mesh<F, TD, GD, C>` of that variant's dimensions, and `body` is compiled once per variant, so
/// it may call generic code; every arm must give the same type. [with_mesh!] shows it with generic
/// code, and [AnyMesh] explains the erased layer.
///
/// [with_mesh!]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#with_mesh
/// [AnyMesh]: https://tetracl.github.io/tetraloma/concepts/typed-dimensions.html#anymesh
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // A generic function over every mesh type: the number of local cells.
/// fn cells<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim, C: Comm>(m: &Mesh<F, TD, GD, C>) -> usize {
///     m.cell_blocks().iter().map(|b| b.index_map.num_local()).sum()
/// }
///
/// // client: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // any: an `AnyMesh<f32>` that holds the unit interval in three cells, as a reader returns a
/// // mesh; the tutorial "Reading and writing files" shows how to read one.
/// # let any = AnyMesh::Curve1(tetraloma_testing::interval_box(3, 1).build::<f32, D1, D1>(&client)?);
/// let (tdim, gdim) = with_mesh!(any.as_ref(), m => (m.tdim(), m.gdim()));
/// assert_eq!((tdim, gdim), (1, 1));
/// assert_eq!(with_mesh!(any.as_ref(), m => cells(m)), 3);
/// # Ok::<(), GridError>(())
/// ```
#[macro_export]
macro_rules! with_mesh {
    ($any:expr, $m:ident => $body:expr) => {
        match $any {
            $crate::AnyMeshRef::Curve1($m) => $body,
            $crate::AnyMeshRef::Curve2($m) => $body,
            $crate::AnyMeshRef::Curve3($m) => $body,
            $crate::AnyMeshRef::Planar($m) => $body,
            $crate::AnyMeshRef::Surface($m) => $body,
            $crate::AnyMeshRef::Volume($m) => $body,
        }
    };
}

/// `TryFrom<AnyMesh<F, C>>` for one of the six mesh types.
macro_rules! try_from_any {
    ($variant:ident, $td:ty, $gd:ty) => {
        impl<F: Real, C: Comm> TryFrom<AnyMesh<F, C>> for Mesh<F, $td, $gd, C> {
            type Error = GridError;
            fn try_from(m: AnyMesh<F, C>) -> Result<Self> {
                match m {
                    AnyMesh::$variant(mesh) => Ok(mesh),
                    other => Err(GridError::DimensionMismatch {
                        expected: (<$td as Dim>::N, <$gd as Dim>::N),
                        found: (other.tdim(), other.gdim()),
                    }),
                }
            }
        }
    };
}

try_from_any!(Curve1, D1, D1);
try_from_any!(Curve2, D1, D2);
try_from_any!(Curve3, D1, D3);
try_from_any!(Planar, D2, D2);
try_from_any!(Surface, D2, D3);
try_from_any!(Volume, D3, D3);
