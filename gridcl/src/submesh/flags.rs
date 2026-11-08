//! The orientation flags of codimension-1 facet children (ADR-0039, item 8): bit 0 where the
//! child's geometric normal at the facet midpoint points into its parent cell.
//!
//! The child map is the parent map composed with the facet's affine map A, so its Jacobian is
//! J A, and its geometric normal J A e0 × J A e1 (or the rotated J A e0 in 2D) is cof(J) t with
//! t = A e0 × A e1 (or (A e0)_y, −(A e0)_x). The parent's outward normal is
//! J^−T n_ref = cof(J) n_ref / det J. Both t and n_ref are normal to the reference facet, so
//! t = λ n_ref, and the dot product of the two normals is λ |cof(J) n_ref|² / det J: its sign
//! is sign(λ) · sign(det J). The kernel therefore reads det J of the parent at the facet
//! midpoint (one Gauss point per facet) and a comptime per-(cell type, facet) table of
//! sign(λ); this is the sign of the dot product of rule 8, without its rounding.

use cubecl::prelude::*;
use gridcl_device::{DeviceBuffer, Real, Result, debug_assert_bindings};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{FacetQuadRule, detj_block};
use gridcl_parallel::Comm;

use super::topology::ChildTopology;
use super::{fit, plan};
use crate::Mesh;

/// The comptime table of the flags kernel for one parent cell type: per local facet f, the
/// first point of f in the facet rule and whether λ_f < 0 (module docs).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct FacetTable {
    facets: Vec<(u32, bool)>,
}

/// `flags[i] = (λ_f < 0) ^ (det J < 0)` for the children i < len whose parent cell c lies in
/// the chunk's cells `[cell_begin, cell_begin + chunk_cells)`, f being the local facet: det J
/// is read at the chunk's tiled point q of f (`t`) of local cell c − cell_begin. Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn flags_kernel<F: Float>(
    cells: &[u32],
    facets: &[u32],
    detj: &[F],
    flags: &mut [u32],
    len: u32,
    cell_begin: u32,
    chunk_cells: u32,
    num_points: u32,
    tile: u32,
    items_per_unit: u32,
    #[comptime] t: FacetTable,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let c = cells[i];
            if c >= cell_begin && c - cell_begin < chunk_cells {
                let lc = c - cell_begin;
                let tl = lc / tile;
                let lane = lc - tl * tile;
                let f = facets[i];
                #[unroll]
                for g in 0..comptime!(t.facets.len()) {
                    if f == comptime!(g as u32) {
                        let q = comptime!(t.facets[g].0);
                        let flip = comptime!(u32::from(t.facets[g].1));
                        let d = detj[((tl * num_points + q) * tile + lane) as usize];
                        flags[i] = flip ^ u32::cast_from(d < F::from_int(0));
                    }
                }
            }
        }
    }
}

/// Per facet f of `cell`: the first point of f in the facet rule, and whether λ_f < 0, λ_f
/// being the sign of t · n_ref (module docs).
fn facet_table(cell: CellType, offsets: &[usize]) -> Result<FacetTable> {
    let tdim = gridcl_refcell::tdim(cell);
    let refs = gridcl_refcell::vertices(cell);
    let normals = gridcl_refcell::reference_facet_normals(cell);
    let facets = gridcl_refcell::sub_entities(cell, tdim - 1)
        .iter()
        .enumerate()
        .map(|(f, verts)| {
            let edge = |k: usize| -> Vec<f64> {
                (0..tdim)
                    .map(|d| refs[verts[k]][d] - refs[verts[0]][d])
                    .collect()
            };
            let a = edge(1);
            let t = if tdim == 2 {
                vec![a[1], -a[0]]
            } else {
                let b = edge(2);
                vec![
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ]
            };
            let dot: f64 = t
                .iter()
                .zip(&normals[f * tdim..(f + 1) * tdim])
                .map(|(x, n)| x * n)
                .sum();
            Ok((fit(offsets[f])?, dot < 0.0))
        })
        .collect::<Result<_>>()?;
    Ok(FacetTable { facets })
}

/// The flags of every child block of the facet children `child` of `mesh`: bit 0 as in the
/// module docs where TD = GD (codimension-1 children); zeros for the codimension-2 boundary of
/// a surface in 3D (ADR-0039, item 8).
pub(crate) fn facet_flags<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    child: &ChildTopology<TD::Lower>,
) -> Result<Vec<DeviceBuffer<u32>>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
    TD::Lower: TopoDim,
{
    let client = &mesh.client;
    let out: Vec<DeviceBuffer<u32>> = child
        .facet_blocks
        .iter()
        .map(|(cells, _)| DeviceBuffer::zeros(client, cells.len()))
        .collect::<Result<_>>()?;
    if TD::N != GD::N || out.iter().all(|f| f.len() == 0) {
        return Ok(out);
    }
    let cells: Vec<TD::Cell> = mesh.topology.cell_blocks().iter().map(|b| b.cell).collect();
    let rule = FacetQuadRule::<F, TD>::gauss(client, &cells, 1)?;
    let mut cell_begin = 0usize;
    for (b, block) in mesh.topology.cell_blocks().iter().enumerate() {
        let num_local = block.index_map.num_local();
        let present = child.present.iter().any(|p| p[b]);
        if num_local > 0 && present {
            let set = rule.get(block.cell)?;
            let table = facet_table(block.cell.into(), set.facet_offsets())?;
            let detj = detj_block(mesh, b, set.points())?;
            let (num_points, tile) = (detj.num_points(), detj.tile());
            for chunk in detj.chunks() {
                fit(chunk.data.len())?;
                for ((cells, facets), flags) in child.facet_blocks.iter().zip(&out) {
                    let n = cells.len();
                    if n == 0 {
                        continue;
                    }
                    let launch = plan(client, n)?;
                    debug_assert_bindings(client, 4);
                    // SAFETY: children i < n read cells[i] and facets[i] (n words each) and
                    // write flags[i] (n words). Those with c in the chunk's cells and a local
                    // facet in the table read det J at ((tl · num_points + q) · tile + lane)
                    // with tl · tile + lane = c − first < chunk cells and q < num_points,
                    // inside the chunk's padded tiles, whose length fits u32 (checked above).
                    unsafe {
                        flags_kernel::launch_unchecked::<F>(
                            client,
                            launch.cube_count,
                            launch.cube_dim,
                            cells.arg(),
                            facets.arg(),
                            chunk.data.arg(),
                            flags.arg(),
                            fit(n)?,
                            fit(cell_begin + chunk.first_cell)?,
                            fit(chunk.num_cells)?,
                            fit(num_points)?,
                            fit(tile)?,
                            launch.items_per_unit,
                            table.clone(),
                        );
                    }
                }
            }
        }
        cell_begin += num_local;
    }
    Ok(out)
}
