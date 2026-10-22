//! The erased layer (Listing B6): `AnyMesh`, `AnyMeshRef`, `with_mesh!` and the `TryFrom`
//! impls.

use gridcl_device::{GridError, Real, Result};
use gridcl_dims::{D1, D2, D3};
use gridcl_parallel::{Comm, Serial};

use crate::Mesh;

/// A mesh whose dimensions are known only at run time.
pub enum AnyMesh<F: Real, C: Comm = Serial> {
    Curve1(Mesh<F, D1, D1, C>),
    Curve2(Mesh<F, D1, D2, C>),
    Curve3(Mesh<F, D1, D3, C>),
    Planar(Mesh<F, D2, D2, C>),
    Surface(Mesh<F, D2, D3, C>),
    Volume(Mesh<F, D3, D3, C>),
}

/// Borrowed view, so writers never need to move or clone a mesh.
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
        todo!()
    }

    /// The topological dimension.
    pub fn tdim(&self) -> usize {
        todo!()
    }

    /// The geometric dimension.
    pub fn gdim(&self) -> usize {
        todo!()
    }
}

/// Runs `$body` once per variant with `$m: &Mesh<F, TD, GD, C>` of that variant's types.
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
                let _ = m;
                todo!()
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
