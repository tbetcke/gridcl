//! Field shapes (Listing B4b).

use core::marker::PhantomData;

use tetraloma_dims::{Dim, EmbeddedIn};

// Design: "Geometry", "Geometric factors", Listing B4b.
/// The shape of a field's value at one point: `LEN` components, read back to the host as one
/// `Host<T>`.
///
/// `Host<T>` must be `LEN` values of `T` in an array, possibly nested and column-major, as in
/// the four shapes below: [`QuadField::to_host_cell_major`](crate::QuadField::to_host_cell_major)
/// writes the components through it. [The Jacobian] and [The pseudo-inverse] give the storage
/// order of the matrices.
///
/// # Example
///
/// ```
/// use tetraloma::{D2, D3, FieldShape, JacShape, PinvShape, ScalarShape, VecShape};
///
/// assert_eq!(ScalarShape::LEN, 1);
/// assert_eq!(VecShape::<D3>::LEN, 3);
/// assert_eq!(JacShape::<D2, D3>::LEN, 6);
/// assert_eq!(PinvShape::<D2, D3>::LEN, 6);
/// ```
///
/// [The Jacobian]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-jacobian
/// [The pseudo-inverse]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-pseudo-inverse
pub trait FieldShape {
    /// Components per point.
    const LEN: usize;
    /// The host value of one point.
    type Host<T: Copy + Default>: Copy + Default;
}

/// One scalar per point, for example det J ([The determinant]); read back as `T`.
///
/// # Example
///
/// ```
/// use tetraloma::{FieldShape, ScalarShape};
///
/// // A plain value, e.g. det J.
/// let det: <ScalarShape as FieldShape>::Host<f32> = 0.25;
/// assert_eq!(ScalarShape::LEN, 1);
/// assert_eq!(det, 0.25f32);
/// ```
///
/// [The determinant]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-determinant
pub struct ScalarShape;
/// A vector of `GD` components per point, for example a physical point or a normal
/// ([Points], [Normals]); read back as `[T; GD]`.
///
/// # Example
///
/// ```
/// use tetraloma::{D3, FieldShape, VecShape};
///
/// let x: <VecShape<D3> as FieldShape>::Host<f32> = [0.0, 0.5, 1.0];
/// assert_eq!(VecShape::<D3>::LEN, x.len());
/// ```
///
/// [Points]: https://tetracl.github.io/tetraloma/tutorials/geometric-data.html#points
/// [Normals]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#normals
pub struct VecShape<GD>(PhantomData<GD>);
/// The Jacobian J, a GD × TD matrix per point, stored column-major: entry (γ, d) is component
/// d · GD + γ. Read back as `[[T; GD]; TD]`, so `j[d][γ]` is J_γd ([The Jacobian]).
///
/// # Example
///
/// ```
/// use tetraloma::{D2, D3, FieldShape, JacShape};
///
/// // A surface in 3D: two columns of three entries.
/// let j: <JacShape<D2, D3> as FieldShape>::Host<f32> = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
/// assert_eq!(JacShape::<D2, D3>::LEN, 6);
/// assert_eq!(j[1][1], 1.0);
/// ```
///
/// [The Jacobian]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-jacobian
pub struct JacShape<TD, GD>(PhantomData<(TD, GD)>);
/// The pseudo-inverse K = (JᵀJ)⁻¹Jᵀ, a TD × GD matrix per point, stored column-major: entry
/// (d, γ) is component γ · TD + d. Read back as `[[T; TD]; GD]`, so `k[γ][d]` is K_dγ
/// ([The pseudo-inverse]).
///
/// # Example
///
/// ```
/// use tetraloma::{D2, D3, FieldShape, PinvShape};
///
/// // A surface in 3D: three columns of two entries.
/// let k: <PinvShape<D2, D3> as FieldShape>::Host<f32> = [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]];
/// assert_eq!(PinvShape::<D2, D3>::LEN, 6);
/// assert_eq!(k[2], [0.0, 0.0]);
/// ```
///
/// [The pseudo-inverse]: https://tetracl.github.io/tetraloma/concepts/geometric-factors.html#the-pseudo-inverse
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
