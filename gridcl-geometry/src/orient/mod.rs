//! Manifold orientation (docs/design.md, "Manifold orientation"; ADR-0039, items 2 and 12):
//! the rule that chooses the direction of each component, and the hook that computes the
//! orientation flags of a codimension-1 mesh.

use gridcl_device::{DeviceBuffer, Real, Result};
use gridcl_dims::{Codim1, Dim, TopoDim};

use crate::DeviceGrid;

/// How `orient` chooses the direction of each connected component of a codimension-1 mesh
/// (ADR-0039, item 2). Points are absolute coordinates; the first gdim entries are used.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// // Normals away from the point (0, 0, 2); on a curve in 2D only (0, 0) is used.
/// let rule = OrientationRule::AwayFrom([0.0, 0.0, 2.0]);
/// assert_ne!(rule, OrientationRule::Outward);
/// assert_eq!(rule, OrientationRule::AwayFrom([0.0, 0.0, 2.0]));
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrientationRule {
    /// Closed components get positive enclosed signed measure (1/gdim) ∮ x·n; open components
    /// give `GridError::NeedsReferencePoint`.
    Outward,
    /// Each component gets ∫ (x − p)·n ≥ 0: normals away from p.
    AwayFrom([f64; 3]),
    /// Each component gets ∫ (x − p)·n ≤ 0: normals towards p.
    Towards([f64; 3]),
}

/// The orientation flags of every cell block of `g` under `rule`, one word per local cell
/// with bit 0 set where the cell is flipped: the manifold check, parity per interior facet,
/// label propagation and the direction of each component (docs/design.md, "Manifold
/// orientation"; ADR-0039, item 12). Needs the facet entities and the (tdim − 1, tdim)
/// connectivity. Collective. For `Mesh::orient`; not part of the stable API (tests:
/// gridcl's `orient` and `orient_threads`).
///
/// # Errors
///
/// `NonManifold`, `NonOrientable`, `NeedsReferencePoint` (agreed across ranks);
/// `EntitiesNotCreated`, `ConnectivityNotCreated`; allocation, device and communication
/// errors.
#[doc(hidden)]
pub fn orientation_flags<F, TD, GD, G>(
    _g: &G,
    _rule: OrientationRule,
) -> Result<Vec<DeviceBuffer<u32>>>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    todo!("orientation_flags: M9 T7-T9")
}
