//! Field shapes (Listing B4b).

use core::marker::PhantomData;

use gridcl_dims::{Dim, EmbeddedIn};

/// The value type of a field at one point: `LEN` components, read back as `Host<T>`.
///
/// `Host<T>` must be `LEN` values of `T` in an array, possibly nested and column-major, as in
/// the four shapes below: `QuadField::to_host_cell_major` writes the components through it.
///
/// # Example
///
/// ```
/// use gridcl::{D2, D3, FieldShape, JacShape, PinvShape, ScalarShape, VecShape};
///
/// assert_eq!(ScalarShape::LEN, 1);
/// assert_eq!(VecShape::<D3>::LEN, 3);
/// assert_eq!(JacShape::<D2, D3>::LEN, 6);
/// assert_eq!(PinvShape::<D2, D3>::LEN, 6);
/// ```
pub trait FieldShape {
    /// Components per point.
    const LEN: usize;
    /// The host value of one point.
    type Host<T: Copy + Default>: Copy + Default;
}

/// One scalar per point.
///
/// # Example
///
/// ```
/// use gridcl::{FieldShape, ScalarShape};
///
/// // A plain value, e.g. det J.
/// let det: <ScalarShape as FieldShape>::Host<f32> = 0.25;
/// assert_eq!(ScalarShape::LEN, 1);
/// assert_eq!(det, 0.25f32);
/// ```
pub struct ScalarShape;
/// A vector of `GD` components per point.
///
/// # Example
///
/// ```
/// use gridcl::{D3, FieldShape, VecShape};
///
/// let x: <VecShape<D3> as FieldShape>::Host<f32> = [0.0, 0.5, 1.0];
/// assert_eq!(VecShape::<D3>::LEN, x.len());
/// ```
pub struct VecShape<GD>(PhantomData<GD>);
/// The GD × TD Jacobian, column-major.
///
/// # Example
///
/// ```
/// use gridcl::{D2, D3, FieldShape, JacShape};
///
/// // A surface in 3D: two columns of three entries.
/// let j: <JacShape<D2, D3> as FieldShape>::Host<f32> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// assert_eq!(JacShape::<D2, D3>::LEN, 6);
/// assert_eq!(j[1][1], 1.0);
/// ```
pub struct JacShape<TD, GD>(PhantomData<(TD, GD)>);
/// The TD × GD pseudo-inverse, column-major.
///
/// # Example
///
/// ```
/// use gridcl::{D2, D3, FieldShape, PinvShape};
///
/// // A surface in 3D: three columns of two entries.
/// let k: <PinvShape<D2, D3> as FieldShape>::Host<f32> = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
/// assert_eq!(PinvShape::<D2, D3>::LEN, 6);
/// assert_eq!(k[2], [0.0, 0.0]);
/// ```
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
