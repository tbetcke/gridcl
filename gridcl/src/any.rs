//! The erased layer (Listing B6): `AnyMesh`, `AnyMeshRef`, `with_mesh!` and the `TryFrom`
//! impls.

use gridcl_device::{GridError, Real, Result};
use gridcl_dims::{D1, D2, D3, Dim};
use gridcl_parallel::{Comm, Serial};

use crate::Mesh;

/// A mesh whose dimensions are known only at run time, as file readers return it. Convert it
/// with `TryFrom` into the typed mesh, or borrow it with [`as_ref`](AnyMesh::as_ref) for
/// [`with_mesh!`](crate::with_mesh).
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let h = gridcl_testing::square_box(Cell2::Triangle, 1, 1);
/// let any = AnyMesh::Planar(h.build::<f32, D2, D2>(&client)?);
/// assert_eq!((any.tdim(), any.gdim()), (2, 2));
/// let planar = PlanarMesh::<f32>::try_from(any)?;
/// assert_eq!(planar.cell_blocks()[0].index_map.num_local(), 2);
/// # Ok::<(), GridError>(())
/// ```
pub enum AnyMesh<F: Real, C: Comm = Serial> {
    Curve1(Mesh<F, D1, D1, C>),
    Curve2(Mesh<F, D1, D2, C>),
    Curve3(Mesh<F, D1, D3, C>),
    Planar(Mesh<F, D2, D2, C>),
    Surface(Mesh<F, D2, D3, C>),
    Volume(Mesh<F, D3, D3, C>),
}

/// Borrowed view, so writers never need to move or clone a mesh.
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1);
/// let planar: PlanarMesh<f32> = h.build(&client)?;
/// let cells = with_mesh!(AnyMeshRef::Planar(&planar), m => m.cell_blocks()[0].index_map.num_local());
/// assert_eq!(cells, 4);
/// # Ok::<(), GridError>(())
/// ```
pub enum AnyMeshRef<'a, F: Real, C: Comm = Serial> {
    Curve1(&'a Mesh<F, D1, D1, C>),
    Curve2(&'a Mesh<F, D1, D2, C>),
    Curve3(&'a Mesh<F, D1, D3, C>),
    Planar(&'a Mesh<F, D2, D2, C>),
    Surface(&'a Mesh<F, D2, D3, C>),
    Volume(&'a Mesh<F, D3, D3, C>),
}

impl<F: Real, C: Comm> AnyMesh<F, C> {
    /// The borrowed view.
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

    /// The topological dimension.
    pub fn tdim(&self) -> usize {
        crate::with_mesh!(self.as_ref(), m => m.tdim())
    }

    /// The geometric dimension.
    pub fn gdim(&self) -> usize {
        crate::with_mesh!(self.as_ref(), m => m.gdim())
    }
}

/// Runs `$body` once per variant with `$m: &Mesh<F, TD, GD, C>` of that variant's types.
/// Call sites pass `any.as_ref()`.
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let h = gridcl_testing::interval_box(3, 1);
/// let any = AnyMesh::Curve1(h.build::<f32, D1, D1>(&client)?);
/// let (tdim, gdim) = with_mesh!(any.as_ref(), m => (m.tdim(), m.gdim()));
/// assert_eq!((tdim, gdim), (1, 1));
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
