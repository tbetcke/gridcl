//! Manifold orientation (docs/design.md, "Manifold orientation"; ADR-0039, items 2 and 12):
//! the rule that chooses the direction of each component, and the hook that computes the
//! orientation flags of a codimension-1 mesh.

mod direction;
mod parity;
mod propagate;

use tetraloma_device::{DeviceBuffer, DeviceStrided, Real, Result};
use tetraloma_dims::{Codim1, Dim, TopoDim};
use tetraloma_topology::{facet_cells_of_parts, facet_rows};

use crate::DeviceGrid;
use parity::{Neighbours, SideExchange};
pub use propagate::OrientationComponents;

// Design: "Manifold orientation"; ADR-0039, item 2.
/// How `Mesh::orient` chooses the direction of the normals on each connected component of a
/// surface in 3D or a curve in 2D.
///
/// `orient` first makes the normals of neighbouring cells agree; each connected component then
/// has two consistent orientations, and the rule picks one. The points are absolute
/// coordinates; on a curve in 2D only their first two entries are used. [Manifold orientation]
/// explains the steps, and [Orientation] shows `orient` on a surface.
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // Normals away from the point (0, 0, 2); on a curve in 2D only (0, 0) is used.
/// let rule = OrientationRule::AwayFrom([0.0, 0.0, 2.0]);
/// assert_ne!(rule, OrientationRule::Outward);
/// assert_eq!(rule, OrientationRule::AwayFrom([0.0, 0.0, 2.0]));
/// ```
///
/// [Manifold orientation]: https://tetracl.github.io/tetraloma/concepts/orientation.html#manifold-orientation
/// [Orientation]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#orientation
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OrientationRule {
    /// Normals out of the enclosed region: each closed component gets a positive enclosed
    /// signed measure (1/gdim) ∮ x·n. A component with a boundary encloses nothing and gives
    /// `GridError::NeedsReferencePoint`.
    Outward,
    /// Normals away from the point p: each component gets ∫ (x − p)·n ≥ 0.
    AwayFrom([f64; 3]),
    /// Normals towards the point p: each component gets ∫ (x − p)·n ≤ 0.
    Towards([f64; 3]),
}

/// The orientation flags of every cell block of `g` under `rule`, one word per local cell
/// with bit 0 set where the cell is flipped: the manifold check, parity per interior facet,
/// label propagation and the direction of each component (docs/design.md, "Manifold
/// orientation"; ADR-0039, item 12). Needs the facet entities and the (tdim − 1, tdim)
/// connectivity. Collective. For `Mesh::orient`; not part of the stable API (tests:
/// tetraloma's `orient` and `orient_threads`).
///
/// # Errors
///
/// `NonManifold`, `NonOrientable`, `NeedsReferencePoint` (agreed across ranks);
/// `EntitiesNotCreated`, `ConnectivityNotCreated`; allocation, device and communication
/// errors.
#[doc(hidden)]
pub fn orientation_flags<F, TD, GD, G>(
    g: &G,
    rule: OrientationRule,
) -> Result<Vec<DeviceBuffer<u32>>>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    G: DeviceGrid<F = F, TD = TD, GD = GD>,
{
    with_neighbours(g, |exchange, counts, n| {
        let components =
            propagate::propagate(g.client(), g.comm(), g.cell_blocks(), &n, counts, exchange)?;
        direction::flags(g, rule, &components, &n)
    })
}

/// The neighbour table of `g` (rule 12): per cell block, per cell and local facet,
/// `(neighbour << 1) | parity` with the parity of the shared facet (1 where the two cells
/// induce the same direction on it), or `u32::MAX` where the facet has not two cells
/// globally. Neighbours below the number of local cells are flat local cells; a neighbour
/// `num_cells + f` is the remote side of facet f, given for owned cells only. Runs the manifold
/// check first. Collective. Not part of the stable API (ADR-0038; test: tetraloma-geometry's
/// `orient`).
///
/// # Errors
///
/// As [`orientation_flags`], up to `NonManifold`.
#[doc(hidden)]
pub fn orientation_neighbours<G>(g: &G) -> Result<Vec<DeviceStrided<u32>>>
where
    G: DeviceGrid,
    G::TD: Codim1<G::GD>,
{
    with_neighbours(g, |_, _, n| Ok(n.table))
}

/// Steps 1 to 3 of `orient` on `g`: the manifold check, the neighbour table and label
/// propagation, giving each local cell the smallest cell origin id of its component and its
/// flip relative to that cell. Collective. Not part of the stable API (ADR-0038; test:
/// tetraloma-geometry's `orient`).
///
/// # Errors
///
/// As [`orientation_flags`], up to `NonOrientable`.
#[doc(hidden)]
pub fn orientation_components<G>(g: &G) -> Result<OrientationComponents>
where
    G: DeviceGrid,
    G::TD: Codim1<G::GD>,
{
    with_neighbours(g, |exchange, counts, n| {
        propagate::propagate(g.client(), g.comm(), g.cell_blocks(), &n, counts, exchange)
    })
}

/// Steps 1 and 2 of `orient` on `g`: the manifold check and the neighbour table, handed to
/// `then` with the side exchange it was built with and the owned cells per facet.
fn with_neighbours<G, T>(
    g: &G,
    then: impl FnOnce(&SideExchange<'_>, &DeviceBuffer<u32>, Neighbours) -> Result<T>,
) -> Result<T>
where
    G: DeviceGrid,
    G::TD: Codim1<G::GD>,
{
    let tdim = <G::TD as tetraloma_dims::Dim>::N;
    let client = g.client();
    let facet_blocks = g.entity_blocks(tdim - 1)?;
    let up = parity::upward(g.connectivity(parity::facet_to_cell(tdim))?);
    let rows = facet_rows(g.cell_blocks(), |key| g.connectivity(key))?;
    let facet_cells = facet_cells_of_parts(client, g.cell_blocks(), facet_blocks, &rows)?;
    parity::manifold_check(g.comm(), &facet_cells.counts)?;
    let perms: Vec<_> = (0..g.cell_blocks().len())
        .map(|b| g.permutation_info(b))
        .collect();
    let exchange = SideExchange::new(client, g.comm(), facet_blocks, 1)?;
    let neighbours = parity::neighbours(
        client,
        g.cell_blocks(),
        &perms,
        &rows,
        up,
        &facet_cells.counts,
        &exchange,
    )?;
    then(&exchange, &facet_cells.counts, neighbours)
}
