//! Type-level dimensions (docs/design.md, Listing B1).

use core::{fmt::Debug, hash::Hash};

mod sealed {
    pub trait Sealed {}
}

/// A dimension marker: [`D0`], [`D1`], [`D2`] or [`D3`]. Sealed.
pub trait Dim: sealed::Sealed + Copy + Default + Debug + Eq + Hash + Send + Sync + 'static {
    /// The dimension as a number.
    const N: usize;
    /// One dimension lower. D0::Lower = D0; D0 is not a TopoDim, so it never has a boundary.
    type Lower: Dim;
    /// [T; N] on stable Rust, without generic_const_exprs.
    type Array<T: Copy + Default>: Copy + Default + AsRef<[T]> + AsMut<[T]>;
}

/// Dimension 0.
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
pub trait EmbeddedIn<GD: Dim>: Dim {
    /// GD x TD Jacobian on the host: TD columns of GD entries (column-major).
    type Jacobian<T: Copy + Default>: Copy + Default;
    /// TD x GD pseudo-inverse K = (J^T J)^-1 J^T: GD columns of TD entries.
    type PseudoInverse<T: Copy + Default>: Copy + Default;
}

/// Codimension 0: TD = GD.
pub trait Codim0<GD: Dim>: EmbeddedIn<GD> {}

/// Codimension 1: TD = GD − 1.
pub trait Codim1<GD: Dim>: EmbeddedIn<GD> {}
