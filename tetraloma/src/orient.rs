//! `Mesh::orient` (docs/design.md, "Manifold orientation"; ADR-0039, items 5 and 12): creates
//! the facet entities and the (tdim − 1, tdim) connectivity, computes the flags with
//! `tetraloma_geometry::orientation_flags` and stores them in the cell blocks.

use tetraloma_device::{Real, Result};
use tetraloma_dims::{Codim1, Dim, TopoDim};
use tetraloma_geometry::OrientationRule;
use tetraloma_parallel::Comm;

use crate::Mesh;

/// `Mesh::orient`.
pub(crate) fn orient<F, TD, GD, C>(
    mesh: &mut Mesh<F, TD, GD, C>,
    rule: OrientationRule,
) -> Result<()>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let tdim = TD::N;
    mesh.create_entities(tdim - 1)?;
    mesh.create_connectivity(tdim - 1, tdim)?;
    let flags = tetraloma_geometry::orientation_flags(&*mesh, rule)?;
    for (b, f) in flags.into_iter().enumerate() {
        mesh.topology.set_cell_flags(b, f)?;
    }
    Ok(())
}
