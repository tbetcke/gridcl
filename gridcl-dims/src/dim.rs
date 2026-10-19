//! Type-level dimensions (docs/design.md, Listing B1).
//!
//! Marker types instead of `const N: usize` generics: stable Rust cannot express N − 1 in a
//! return type, or `[T; N]` from an associated constant in generic code. [`Dim`] is sealed, so
//! the orphan rule also stops downstream crates from adding [`EmbeddedIn`] or `Codim` impls.

use core::{fmt::Debug, hash::Hash};

mod sealed {
    pub trait Sealed {}
}

/// A dimension marker: [`D0`], [`D1`], [`D2`] or [`D3`].
///
/// Sealed: no other crate can implement it.
///
/// ```compile_fail,E0277
/// use gridcl_dims::Dim;
///
/// #[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
/// struct D4;
/// impl Dim for D4 {
///     const N: usize = 4;
///     type Lower = gridcl_dims::D3;
///     type Array<T: Copy + Default> = [T; 4];
/// }
/// ```
pub trait Dim: sealed::Sealed + Copy + Default + Debug + Eq + Hash + Send + Sync + 'static {
    /// The dimension as a number.
    const N: usize;
    /// One dimension lower. D0::Lower = D0; D0 is not a TopoDim, so it never has a boundary.
    type Lower: Dim;
    /// [T; N] on stable Rust, without generic_const_exprs.
    type Array<T: Copy + Default>: Copy + Default + AsRef<[T]> + AsMut<[T]>;
}

/// Dimension 0.
///
/// Not a topological dimension: point meshes do not exist, so a curve mesh has no boundary.
///
/// ```compile_fail,E0277
/// use gridcl_dims::{D0, TopoDim};
///
/// fn topo_dim<TD: TopoDim>() {}
/// topo_dim::<D0>();
/// ```
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct D0;
/// Dimension 1.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct D1;
/// Dimension 2.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct D2;
/// Dimension 3.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq, Hash)]
pub struct D3;

impl sealed::Sealed for D0 {}
impl sealed::Sealed for D1 {}
impl sealed::Sealed for D2 {}
impl sealed::Sealed for D3 {}

impl Dim for D0 {
    const N: usize = 0;
    type Lower = D0;
    type Array<T: Copy + Default> = [T; 0];
}
impl Dim for D1 {
    const N: usize = 1;
    type Lower = D0;
    type Array<T: Copy + Default> = [T; 1];
}
impl Dim for D2 {
    const N: usize = 2;
    type Lower = D1;
    type Array<T: Copy + Default> = [T; 2];
}
impl Dim for D3 {
    const N: usize = 3;
    type Lower = D2;
    type Array<T: Copy + Default> = [T; 3];
}

/// TD can be embedded in GD (TD <= GD). Implemented for exactly six pairs.
///
/// The host shapes of the Jacobian and its pseudo-inverse are column-major arrays, so a
/// triangle in 3D has a 3 x 2 Jacobian of two columns:
///
/// ```
/// use gridcl_dims::{D2, D3, EmbeddedIn};
///
/// let j: <D2 as EmbeddedIn<D3>>::Jacobian<f64> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// let k: <D2 as EmbeddedIn<D3>>::PseudoInverse<f64> = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
/// assert_eq!((j.len(), j[0].len()), (2, 3));
/// assert_eq!((k.len(), k[0].len()), (3, 2));
/// ```
pub trait EmbeddedIn<GD: Dim>: Dim {
    /// GD x TD Jacobian on the host: TD columns of GD entries (column-major).
    type Jacobian<T: Copy + Default>: Copy + Default;
    /// TD x GD pseudo-inverse K = (J^T J)^-1 J^T: GD columns of TD entries.
    type PseudoInverse<T: Copy + Default>: Copy + Default;
}

impl EmbeddedIn<D1> for D1 {
    type Jacobian<T: Copy + Default> = [[T; 1]; 1];
    type PseudoInverse<T: Copy + Default> = [[T; 1]; 1];
}
impl EmbeddedIn<D2> for D1 {
    type Jacobian<T: Copy + Default> = [[T; 2]; 1];
    type PseudoInverse<T: Copy + Default> = [[T; 1]; 2];
}
impl EmbeddedIn<D3> for D1 {
    type Jacobian<T: Copy + Default> = [[T; 3]; 1];
    type PseudoInverse<T: Copy + Default> = [[T; 1]; 3];
}
impl EmbeddedIn<D2> for D2 {
    type Jacobian<T: Copy + Default> = [[T; 2]; 2];
    type PseudoInverse<T: Copy + Default> = [[T; 2]; 2];
}
impl EmbeddedIn<D3> for D2 {
    type Jacobian<T: Copy + Default> = [[T; 3]; 2];
    type PseudoInverse<T: Copy + Default> = [[T; 2]; 3];
}
impl EmbeddedIn<D3> for D3 {
    type Jacobian<T: Copy + Default> = [[T; 3]; 3];
    type PseudoInverse<T: Copy + Default> = [[T; 3]; 3];
}

/// Codimension 0 (TD = GD): inverse Jacobians and facet normals exist.
pub trait Codim0<GD: Dim>: EmbeddedIn<GD> {}
impl Codim0<D1> for D1 {}
impl Codim0<D2> for D2 {}
impl Codim0<D3> for D3 {}

/// Codimension 1 (TD = GD − 1): unit normals and manifold orientation exist.
///
/// A volume mesh has no normals:
///
/// ```compile_fail,E0277
/// use gridcl_dims::{Codim1, D3};
///
/// fn codim1<TD: Codim1<GD>, GD: gridcl_dims::Dim>() {}
/// codim1::<D3, D3>();
/// ```
pub trait Codim1<GD: Dim>: EmbeddedIn<GD> {}
impl Codim1<D2> for D1 {}
impl Codim1<D3> for D2 {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn n_of_each_marker() {
        assert_eq!([D0::N, D1::N, D2::N, D3::N], [0, 1, 2, 3]);
    }

    fn lower<D: Dim>() -> usize {
        D::Lower::N
    }

    #[test]
    fn lower_chain() {
        assert_eq!(
            [lower::<D0>(), lower::<D1>(), lower::<D2>(), lower::<D3>()],
            [0, 0, 1, 2]
        );
        assert_eq!(
            core::any::type_name::<<D0 as Dim>::Lower>(),
            core::any::type_name::<D0>()
        );
        assert_eq!(
            core::any::type_name::<<<D3 as Dim>::Lower as Dim>::Lower>(),
            core::any::type_name::<D1>()
        );
    }

    fn array_len<D: Dim>() -> usize {
        let mut a = D::Array::<u8>::default();
        assert_eq!(a.as_mut().len(), a.as_ref().len());
        a.as_ref().len()
    }

    #[test]
    fn array_lengths() {
        assert_eq!(
            [
                array_len::<D0>(),
                array_len::<D1>(),
                array_len::<D2>(),
                array_len::<D3>()
            ],
            [0, 1, 2, 3]
        );
    }

    /// (columns, column length) of a column-major array.
    fn shape<const C: usize, const R: usize>(_: [[u8; R]; C]) -> (usize, usize) {
        (C, R)
    }

    /// Jacobian GD x TD: TD columns of GD entries; pseudo-inverse TD x GD: GD columns of TD.
    macro_rules! check_shapes {
        ($td:ty, $gd:ty) => {
            let j = <$td as EmbeddedIn<$gd>>::Jacobian::<u8>::default();
            let k = <$td as EmbeddedIn<$gd>>::PseudoInverse::<u8>::default();
            let (td, gd) = (<$td>::N, <$gd>::N);
            assert_eq!(shape(j), (td, gd), "Jacobian of D{td} in D{gd}");
            assert_eq!(shape(k), (gd, td), "pseudo-inverse of D{td} in D{gd}");
        };
    }

    #[test]
    fn jacobian_shapes() {
        check_shapes!(D1, D1);
        check_shapes!(D1, D2);
        check_shapes!(D1, D3);
        check_shapes!(D2, D2);
        check_shapes!(D2, D3);
        check_shapes!(D3, D3);
    }
}
