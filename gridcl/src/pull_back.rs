//! `Mesh::pull_back`: `gridcl_geometry::pull_back_block` over the cell blocks (ADR-0039,
//! item 13).

use gridcl_device::{Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{BlockFields, PullBack, PullBackOptions, VecShape};
use gridcl_parallel::Comm;

use crate::Mesh;

/// `Mesh::pull_back`.
pub(crate) fn pull_back<F, TD, GD, C>(
    _mesh: &Mesh<F, TD, GD, C>,
    _x: &BlockFields<F, VecShape<GD>>,
    _opts: &PullBackOptions,
) -> Result<PullBack<F, TD>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    todo!("pull_back: M9 T10b")
}
