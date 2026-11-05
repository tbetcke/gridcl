//! Submeshes: boundary, facet, edge and cell submeshes with their `EntityMap`
//! (docs/design.md, "Tags, regions and submeshes"; ADR-0039, items 1 and 3-11).

use gridcl_device::{Real, Result};
use gridcl_dims::{D3, Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::Comm;

use crate::{CurveMesh, EntityMap, Mesh, MeshTags};

/// `Mesh::boundary`.
#[allow(clippy::type_complexity)] // Listing B4's signature
pub(crate) fn boundary<F, TD, GD, C>(
    _mesh: &Mesh<F, TD, GD, C>,
) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    todo!("boundary: M9 T4b")
}

/// `Mesh::facet_submesh`.
#[allow(clippy::type_complexity)] // Listing B4's signature
pub(crate) fn facet_submesh<F, TD, GD, C>(
    _mesh: &Mesh<F, TD, GD, C>,
    _facets: &MeshTags,
) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap)>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    todo!("facet_submesh: M9 T4b")
}

/// `Mesh::edge_submesh`.
pub(crate) fn edge_submesh<F: Real, C: Comm>(
    _mesh: &Mesh<F, D3, D3, C>,
    _edges: &MeshTags,
) -> Result<(CurveMesh<F, C>, EntityMap)> {
    todo!("edge_submesh: M9 T6")
}

/// `Mesh::cell_submesh`.
pub(crate) fn cell_submesh<F, TD, GD, C>(
    _mesh: &Mesh<F, TD, GD, C>,
    _cells: &MeshTags,
) -> Result<(Mesh<F, TD, GD, C>, EntityMap)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    todo!("cell_submesh: M9 T6")
}
