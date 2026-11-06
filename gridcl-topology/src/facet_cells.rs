//! Facet classification for submeshes and `orient` (ADR-0039, items 11 and 17): the number of
//! owned cells of every local facet over all ranks, which works in every `GhostMode`.
//!
//! Each rank counts its owned cells per local facet (a u32-atomic histogram of the owned
//! cells' downward rows), and each facet block's scatterer sums the counts at the facet owner
//! (`reverse(Add)`) and returns them to every holder (`forward`). A cell is owned by exactly
//! one rank, so the sum is the global number of cells that contain the facet, whatever ghost
//! layer the ranks hold. A facet with one cell globally has that cell on every rank that holds
//! the facet (as an owned or a ghost cell), so its local cell and local facet index are found
//! by a scatter from the local cells without conflicts.

use cubecl::prelude::*;
use gridcl_device::{
    ConnKey, DeviceBuffer, DeviceStrided, GridError, KernelKind, LaunchPlan, Result,
    debug_assert_bindings, histogram,
};
use gridcl_dims::TopoDim;
use gridcl_parallel::{CommDyn, ReverseOp};

use crate::{CellBlock, Connectivity, EntityBlock, Topology};

/// The cells of every local facet (entities of dimension tdim − 1), in flat facet order over
/// the facet blocks. For gridcl's submeshes and `orient`; not part of the stable API.
#[doc(hidden)]
pub struct FacetCells {
    /// The number of owned cells of each local facet, summed over all ranks: 1 on the
    /// boundary, 2 inside a manifold, more where the mesh is not a manifold.
    pub counts: DeviceBuffer<u32>,
    /// For a facet with one cell globally: the flat local index of that cell, which every
    /// holder of the facet holds; `u32::MAX` for other facets.
    pub cell: DeviceBuffer<u32>,
    /// For a facet with one cell globally: its local facet index in that cell; `u32::MAX` for
    /// other facets.
    pub local_facet: DeviceBuffer<u32>,
}

/// `out[base + i] = min(rows[i], last)` for the `len` entries of a block's owned rows: the
/// flat facet of each (owned cell, local facet) pair, clamped like the topology kernels.
#[cube(launch_unchecked)]
fn owned_entries_kernel(
    rows: &[u32],
    out: &mut [u32],
    len: u32,
    base: u32,
    last: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            out[base as usize + i] = min(rows[i], last);
        }
    }
}

/// `out[i] = values[i] − shift`, or `u32::MAX` for values below `shift`, so that the facets of
/// earlier blocks fall outside a histogram of this block's facets.
#[cube(launch_unchecked)]
fn shift_kernel(values: &[u32], out: &mut [u32], len: u32, shift: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let v = values[i];
            if v < shift {
                out[i] = u32::MAX;
            } else {
                out[i] = v - shift;
            }
        }
    }
}

/// `dst[base + i] = src[i]` for i < len.
#[cube(launch_unchecked)]
fn place_kernel(src: &[u32], dst: &mut [u32], len: u32, base: u32, items_per_unit: u32) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            dst[base as usize + i] = src[i];
        }
    }
}

/// For each entry i < len of a block's local rows (k per cell) whose facet f has count 1:
/// `cell[f] = cell_offset + i / k`, `local_facet[f] = i mod k`. Such a facet lies in one
/// local cell only, so no two entries write the same f.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn single_cell_kernel(
    rows: &[u32],
    counts: &[u32],
    cell: &mut [u32],
    local_facet: &mut [u32],
    len: u32,
    k: u32,
    cell_offset: u32,
    last: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let f = min(rows[i], last) as usize;
            if counts[f] == 1 {
                cell[f] = cell_offset + i as u32 / k;
                local_facet[f] = i as u32 % k;
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// A scalar launch over `len` u32 items.
fn plan(client: &Client, len: usize) -> Result<LaunchPlan> {
    LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)
}

/// The [`FacetCells`] of `topology`, whose facet entities exist: owned cells are counted per
/// local facet over their downward (tdim, tdim − 1) rows, combined by a reverse `Add` and
/// returned by a forward scatter over each facet block's `IndexMap` (ADR-0039, item 11).
/// Collective. Not part of the stable API (tests: gridcl-topology's `facet_cells`).
///
/// For tdim = 1 the facets are the vertices (the `Point` block) and the rows are the cells'
/// vertex rows. A topology of dimension 0 has no facets: `EntitiesNotCreated(0)`.
///
/// # Errors
///
/// `EntitiesNotCreated(tdim − 1)` before `create_entities(tdim − 1)`; `TooLarge` if the
/// entries do not fit u32; allocation, device and communication errors.
#[doc(hidden)]
pub fn facet_cell_counts<TD: TopoDim>(
    comm: &dyn CommDyn,
    topology: &Topology<TD>,
) -> Result<FacetCells> {
    let _ = comm; // the scatterers communicate over their maps' communicators
    let tdim = TD::N;
    let Some(dim) = tdim.checked_sub(1) else {
        return Err(GridError::EntitiesNotCreated(0));
    };
    let facet_blocks = topology.entity_blocks(dim)?;
    let rows = facet_rows(topology.cell_blocks(), |key| topology.connectivity(key))?;
    facet_cells_of_parts(
        topology.client(),
        topology.cell_blocks(),
        facet_blocks,
        &rows,
    )
}

/// The downward (tdim, tdim − 1) rows of each cell block, or their vertex rows for tdim = 1,
/// from a connectivity accessor such as `Topology::connectivity` or `DeviceGrid`'s.
///
/// # Errors
///
/// The accessor's `ConnectivityNotCreated` before the facet entities exist;
/// `EntitiesNotCreated(0)` for tdim = 0.
#[doc(hidden)]
pub fn facet_rows<'a, TD: TopoDim>(
    cell_blocks: &'a [CellBlock<TD>],
    connectivity: impl FnOnce(ConnKey) -> Result<&'a Connectivity>,
) -> Result<Vec<&'a DeviceStrided<u32>>> {
    let tdim = TD::N;
    match tdim {
        0 => Err(GridError::EntitiesNotCreated(0)),
        1 => Ok(cell_blocks.iter().map(|b| &b.vertices).collect()),
        _ => {
            let key = ConnKey {
                from: tdim as u8,
                to: (tdim - 1) as u8,
            };
            match connectivity(key)? {
                Connectivity::Downward(rows) => Ok(rows.iter().collect()),
                Connectivity::Upward(_) => unreachable!("(tdim, tdim - 1) is downward"),
            }
        }
    }
}

/// [`facet_cell_counts`] from the parts of a topology or a `DeviceGrid`: its client, cell
/// blocks, facet blocks (entities of dimension tdim − 1) and the cells' facet rows
/// ([`facet_rows`]). Collective. For gridcl-geometry's `orient`, which sees a mesh as a
/// `DeviceGrid`; not part of the stable API (ADR-0038; tests: gridcl-topology's
/// `facet_cells`, through [`facet_cell_counts`], and gridcl's `orient`).
///
/// # Errors
///
/// As [`facet_cell_counts`]; `LengthMismatch` if `rows` has not one entry per cell block.
#[doc(hidden)]
pub fn facet_cells_of_parts<TD: TopoDim>(
    client: &Client,
    cell_blocks: &[CellBlock<TD>],
    facet_blocks: &[EntityBlock],
    rows: &[&DeviceStrided<u32>],
) -> Result<FacetCells> {
    if rows.len() != cell_blocks.len() {
        return Err(GridError::LengthMismatch {
            expected: cell_blocks.len(),
            found: rows.len(),
        });
    }
    let client = client.clone();
    let num_facets = facet_blocks
        .iter()
        .try_fold(0usize, |acc, b| acc.checked_add(b.index_map.num_local()))
        .ok_or(GridError::TooLarge)?;
    let last = fit(num_facets)?.saturating_sub(1);

    // The flat facet of every (owned cell, local facet) pair.
    let owned_len = cell_blocks
        .iter()
        .zip(rows)
        .try_fold(0usize, |acc, (b, r)| {
            b.index_map
                .num_owned()
                .checked_mul(r.stride as usize)
                .and_then(|n| acc.checked_add(n))
        })
        .ok_or(GridError::TooLarge)?;
    let entries = DeviceBuffer::<u32>::uninit(&client, owned_len)?;
    let mut base = 0usize;
    for (block, r) in cell_blocks.iter().zip(rows) {
        let len = block.index_map.num_owned() * r.stride as usize;
        if len > 0 {
            let launch = plan(&client, len)?;
            debug_assert_bindings(&client, 2);
            // SAFETY: the kernel reads rows[i] for i < len ≤ rows.len() (the owned cells come
            // first) and writes out[base + i] < base + len ≤ owned_len, its length.
            unsafe {
                owned_entries_kernel::launch_unchecked(
                    &client,
                    launch.cube_count,
                    launch.cube_dim,
                    r.data.arg(),
                    entries.arg(),
                    fit(len)?,
                    fit(base)?,
                    last,
                    launch.items_per_unit,
                );
            }
        }
        base += len;
    }

    // Per facet block: the local counts of its facets, summed at the owners and returned.
    let counts = DeviceBuffer::<u32>::zeros(&client, num_facets)?;
    for block in facet_blocks {
        let n = block.index_map.num_local();
        if n == 0 && block.index_map.size_global() == 0 {
            continue;
        }
        let values = if block.offset == 0 || owned_len == 0 {
            entries.share()
        } else {
            let shifted = DeviceBuffer::<u32>::uninit(&client, owned_len)?;
            let launch = plan(&client, owned_len)?;
            debug_assert_bindings(&client, 2);
            // SAFETY: the kernel reads values[i] and writes out[i] for i < len = owned_len,
            // both buffers' length.
            unsafe {
                shift_kernel::launch_unchecked(
                    &client,
                    launch.cube_count,
                    launch.cube_dim,
                    entries.arg(),
                    shifted.arg(),
                    fit(owned_len)?,
                    block.offset,
                    launch.items_per_unit,
                );
            }
            shifted
        };
        let mut local = histogram(&values, fit(n)?)?;
        // Collective over the block's map: every rank takes part, even with no facets.
        let scatterer = block.index_map.scatterer(&client, 1)?;
        scatterer.reverse(&mut local, ReverseOp::Add)?;
        scatterer.forward(&mut local)?;
        if n > 0 {
            let launch = plan(&client, n)?;
            debug_assert_bindings(&client, 2);
            // SAFETY: the kernel reads src[i] for i < n = src.len() and writes
            // dst[offset + i] < offset + n ≤ num_facets, dst's length (the blocks tile the
            // flat facets).
            unsafe {
                place_kernel::launch_unchecked(
                    &client,
                    launch.cube_count,
                    launch.cube_dim,
                    local.arg(),
                    counts.arg(),
                    fit(n)?,
                    block.offset,
                    launch.items_per_unit,
                );
            }
        }
    }

    // The single local cell of each facet with one cell globally.
    let mut cell = DeviceBuffer::<u32>::uninit(&client, num_facets)?;
    let mut local_facet = DeviceBuffer::<u32>::uninit(&client, num_facets)?;
    gridcl_device::fill(&mut cell, u32::MAX)?;
    gridcl_device::fill(&mut local_facet, u32::MAX)?;
    let mut cell_offset = 0usize;
    for (block, r) in cell_blocks.iter().zip(rows) {
        let len = r.data.len();
        if len > 0 && num_facets > 0 {
            let launch = plan(&client, len)?;
            debug_assert_bindings(&client, 4);
            // SAFETY: the kernel reads rows[i] for i < len = rows.len() and counts[f], and
            // writes cell[f] and local_facet[f], for f = min(rows[i], last) < num_facets,
            // their length.
            unsafe {
                single_cell_kernel::launch_unchecked(
                    &client,
                    launch.cube_count,
                    launch.cube_dim,
                    r.data.arg(),
                    counts.arg(),
                    cell.arg(),
                    local_facet.arg(),
                    fit(len)?,
                    r.stride.max(1),
                    fit(cell_offset)?,
                    last,
                    launch.items_per_unit,
                );
            }
        }
        cell_offset += block.index_map.num_local();
    }
    fit(cell_offset)?;
    Ok(FacetCells {
        counts,
        cell,
        local_facet,
    })
}
