//! Field shapes (Listing B4b).

use core::marker::PhantomData;

use gridcl_dims::{Dim, EmbeddedIn};

/// The value type of a field at one point: `LEN` components, read back as `Host<T>`.
///
/// `Host<T>` must be `LEN` values of `T` in an array, possibly nested and column-major, as in
/// the four shapes below: `QuadField::to_host_cell_major` writes the components through it.
pub trait FieldShape {
    /// Components per point.
    const LEN: usize;
    /// The host value of one point.
    type Host<T: Copy + Default>: Copy + Default;
}

/// One scalar per point.
pub struct ScalarShape;
/// A vector of `GD` components per point.
pub struct VecShape<GD>(PhantomData<GD>);
/// The GD × TD Jacobian, column-major.
pub struct JacShape<TD, GD>(PhantomData<(TD, GD)>);
/// The TD × GD pseudo-inverse, column-major.
pub struct PinvShape<TD, GD>(PhantomData<(TD, GD)>);

impl FieldShape for ScalarShape {
    const LEN: usize = 1;
    type Host<T: Copy + Default> = T;
}

impl<GD: Dim> FieldShape for VecShape<GD> {
    const LEN: usize = GD::N;
    type Host<T: Copy + Default> = GD::Array<T>;
}

impl<TD: EmbeddedIn<GD>, GD: Dim> FieldShape for JacShape<TD, GD> {
    const LEN: usize = TD::N * GD::N;
    type Host<T: Copy + Default> = <TD as EmbeddedIn<GD>>::Jacobian<T>;
}

impl<TD: EmbeddedIn<GD>, GD: Dim> FieldShape for PinvShape<TD, GD> {
    const LEN: usize = TD::N * GD::N;
    type Host<T: Copy + Default> = <TD as EmbeddedIn<GD>>::PseudoInverse<T>;
}
