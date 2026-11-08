//! `Mesh::pull_back`: `gridcl_geometry::pull_back_block` over the cell blocks (ADR-0039,
//! item 13).

use gridcl_device::{GridError, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{BlockFields, PullBack, PullBackOptions, VecShape, pull_back_block};
use gridcl_parallel::Comm;

use crate::Mesh;

/// `Mesh::pull_back`.
pub(crate) fn pull_back<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    x: &BlockFields<F, VecShape<GD>>,
    opts: &PullBackOptions,
) -> Result<PullBack<F, TD>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let num_blocks = mesh.topology.cell_blocks().len();
    if x.blocks.len() != num_blocks {
        return Err(GridError::LengthMismatch {
            expected: num_blocks,
            found: x.blocks.len(),
        });
    }
    let (reference, last_step) = x
        .blocks
        .iter()
        .enumerate()
        .map(|(b, x)| pull_back_block(mesh, b, x, opts))
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .unzip();
    Ok(PullBack {
        reference: BlockFields { blocks: reference },
        last_step: BlockFields { blocks: last_step },
    })
}
