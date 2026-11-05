//! Pull-back: reference points of physical points by Newton iteration, generic over the
//! vector size (docs/design.md, "Pull-back"; Listing B4b; ADR-0039, items 13-15).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};

use crate::{BlockFields, DeviceGrid, QuadField, ScalarShape, VecShape};

/// Options of the pull-back (ADR-0039, item 13).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let opts = PullBackOptions::default();
/// assert_eq!((opts.tolerance, opts.max_iterations), (None, 16));
/// // A fixed tolerance and more iterations for strongly curved cells.
/// let opts = PullBackOptions { tolerance: Some(1e-10), max_iterations: 32 };
/// assert_eq!(opts.max_iterations, 32);
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PullBackOptions {
    /// A lane freezes once |Δξ|∞ ≤ tolerance; `None` is 10^−13 in f64 and 10^−5 in f32.
    pub tolerance: Option<f64>,
    /// The number of iterations, always run in full: there is no early exit.
    pub max_iterations: u32,
}

impl Default for PullBackOptions {
    fn default() -> Self {
        PullBackOptions {
            tolerance: None,
            max_iterations: 16,
        }
    }
}

/// The result of a pull-back, one field per cell block, tiled like the input.
pub struct PullBack<F: Real, TD: TopoDim> {
    /// The reference point ξ per cell and point.
    pub reference: BlockFields<F, VecShape<TD>>,
    /// |Δξ|∞ of the last iteration; above the tolerance where the point has not converged.
    pub last_step: BlockFields<F, ScalarShape>,
}

/// The reference points of the physical points `x` (absolute coordinates per cell and point
/// of block `b`, as `physical_points_block` returns them), and |Δξ|∞ of the last iteration.
///
/// Iteration starts at the reference cell's vertex centroid with Δ = K(ξ)(x − X(ξ)), K the
/// pseudo-inverse: Newton in codimension 0, Gauss–Newton otherwise. A lane freezes once
/// |Δ|∞ ≤ tolerance; the loop always runs `max_iterations` (ADR-0039, item 13).
///
/// # Errors
///
/// `LengthMismatch` if `x` does not have one cell per local cell of block `b`; allocation and
/// device errors.
#[allow(clippy::type_complexity)] // Listing B4b's signature
pub fn pull_back_block<F, TD, GD, G>(
    _g: &G,
    _b: usize,
    _x: &QuadField<F, VecShape<GD>>,
    _opts: &PullBackOptions,
) -> Result<(QuadField<F, VecShape<TD>>, QuadField<F, ScalarShape>)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    todo!("pull_back_block: M9 T10b")
}
