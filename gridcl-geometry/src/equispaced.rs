//! GLL → equispaced conversion of the geometry nodes (docs/design.md, "Reference cells and
//! conventions", node families; ADR-0024, item 7).
//!
//! 1. The occurrences of the nodes of all GLL blocks, in flat (block, cell, local node) order,
//!    are keyed by node index (Scalar kernel) and sorted stably (`radix_sort`), so
//!    `unique_by_key` gives each node its first occurrence (its one allowed count read).
//! 2. Per GLL block, the physical-points factor kernel (GeomFactors, Listing B5) evaluates the
//!    block's geometry at the equispaced lattice `lagrange_points(cell, p, Equispaced)`, whose
//!    point j is the cell's local node j (A.5 order). It adds a zero buffer in place of the
//!    origin, so the values stay relative to the origin like the L-vector.
//! 3. A Scalar kernel per field chunk copies the value of each node's first occurrence into a
//!    copy of the L-vector. Each node is written once: no atomics, bitwise deterministic.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, SortKeys, debug_assert_bindings,
    radix_sort, unique_by_key,
};
use gridcl_dims::{CellType, Dim};
use gridcl_refcell::{NodeFamily, lagrange_points};

use crate::layout::{Factor, launch_factor};
use crate::names::element_error;
use crate::{DeviceGrid, PointSet, QuadField, VecShape};

/// The occurrences of one GLL block: its local cells' dofmap rows, flat indices
/// `start..start + len`, `len = cells · k`.
struct Occurrences {
    block: usize,
    start: usize,
    len: usize,
    k: usize,
}

/// Key and payload of the occurrences i < `len` of one block: `keys[start + i] = dofmap[i]`
/// (the node) and `payload[start + i] = start + i` (the flat occurrence).
#[cube(launch_unchecked)]
fn occurrences_kernel(
    dofmap: &[u32],
    keys: &mut [u32],
    payload: &mut [u32],
    len: u32,
    start: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            keys[start as usize + i] = dofmap[i];
            payload[start as usize + i] = start + i as u32;
        }
    }
}

/// For every node group u < `count` whose first occurrence lies in `[start, end)`, the
/// occurrences of the cells of one field chunk: `out[node · G + g]` = the chunk's value at that
/// occurrence.
///   first  the first flat occurrence of each group (`Unique::first`)
///   nodes  the node of each group (plane 0 of `Unique::keys`)
///   field  X at the lattice, tiled: component g at point q of chunk cell c at
///          ((c / tile · k + q) · G + g) · tile + c − (c / tile) · tile
///   out    the L-vector [node][G]
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn select_kernel<F: Float>(
    first: &[u32],
    nodes: &[u32],
    field: &[F],
    out: &mut [F],
    count: u32,
    start: u32,
    end: u32,
    num_nodes: u32,
    k: u32,
    tile: u32,
    items_per_unit: u32,
    #[comptime] gdim: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let u = begin + it;
        if u < count as usize {
            let f = first[u];
            let node = nodes[u];
            if f >= start && f < end && node < num_nodes {
                // v − (v / d) · d, not v % d (CubeCL 0.11 folds `x % 1` on the CPU runtime).
                let r = (f - start) as usize;
                let c = r / k as usize;
                let q = r - c * k as usize;
                let t = c / tile as usize;
                let lane = c - t * tile as usize;
                #[unroll]
                for g in 0..gdim {
                    out[node as usize * gdim + g] =
                        field[((t * k as usize + q) * gdim + g) * tile as usize + lane];
                }
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

/// The L-vector of `grid` with the numbering, length and origin of
/// [`nodes`](DeviceGrid::nodes), where every node of a GLL block holds the equispaced
/// interpolant of its cell's geometry. Equispaced blocks are copied unchanged.
///
/// A node shared by several cells takes its value from its first occurrence in flat
/// (block, cell, local node) order, so the result is bitwise deterministic.
///
/// The geometry is evaluated by the physical-points kernel at the equispaced lattice of each
/// GLL element, relative to the origin (no round trip through the absolute coordinates). Only
/// local cells are evaluated, so on a distributed grid a ghost node that no local GLL cell
/// touches keeps its GLL value.
///
/// The copy into the L-vector runs once per field chunk, and each run scans all unique nodes,
/// so its work is (unique nodes) × (chunks). A block has one chunk unless its field exceeds
/// the device's largest allocation.
///
/// # Errors
///
/// `LengthMismatch` if a GLL block's dofmap has fewer rows than local cells; `TooLarge` if
/// the occurrences or a field chunk do not fit u32 indices; allocation and device errors.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
/// use gridcl_geometry::equispaced_nodes;
/// use gridcl_refcell::lagrange_points;
///
/// let client = gridcl_testing::test_client();
/// // One GLL interval of degree 4 on [0, 2]: its nodes are the GLL points, in A.5 order.
/// let gll = lagrange_points(CellType::Interval, 4, NodeFamily::Gll).unwrap();
/// let coords: Vec<f64> = gll.iter().map(|x| 2.0 * x).collect();
/// let e = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 4, family: NodeFamily::Gll };
/// let mesh: Mesh<f32, D1, D1> =
///     MeshBuilder::new(&client).nodes(&coords, None).cell_block(e, &[0, 1, 2, 3, 4], None).build()?;
/// // The interval is affine, so its equispaced nodes are 0, 2, 0.5, 1, 1.5, relative to the
/// // origin 1.
/// let x = equispaced_nodes(&mesh)?.to_host()?;
/// for (got, want) in x.iter().zip([-1.0, 1.0, -0.5, 0.0, 0.5]) {
///     assert!((got - want).abs() < 1e-6, "{got} != {want}");
/// }
/// # Ok::<(), GridError>(())
/// ```
pub fn equispaced_nodes<G: DeviceGrid>(grid: &G) -> Result<DeviceBuffer<G::F>> {
    let client = grid.client();
    let nodes = grid.nodes();
    let gdim = G::GD::N;
    let out = nodes.deep_clone()?;

    // The GLL blocks' occurrences, in flat (block, cell, local node) order.
    let mut blocks = Vec::new();
    let mut total = 0usize;
    for (b, cell_block) in grid.cell_blocks().iter().enumerate() {
        let gb = grid.geometry_block(b);
        if gb.element.family != NodeFamily::Gll {
            continue;
        }
        let k = gb.nodes_per_cell();
        let len = cell_block
            .index_map
            .num_local()
            .checked_mul(k)
            .ok_or(GridError::TooLarge)?;
        if len > gb.dofmap.data.len() {
            return Err(GridError::LengthMismatch {
                expected: len,
                found: gb.dofmap.data.len(),
            });
        }
        if len == 0 {
            continue;
        }
        blocks.push(Occurrences {
            block: b,
            start: total,
            len,
            k,
        });
        total = total.checked_add(len).ok_or(GridError::TooLarge)?;
    }
    if total == 0 {
        return Ok(out);
    }
    fit(total)?;
    let num_nodes = fit(nodes.len() / gdim)?;
    fit(out.len())?;

    // 1. Each node's first occurrence.
    let mut keys = SortKeys::uninit(client, total, 1)?;
    let payload = DeviceBuffer::<u32>::uninit(client, total)?;
    for occ in &blocks {
        let len = occ.len;
        let dofmap = &grid.geometry_block(occ.block).dofmap.data;
        let plan = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
        debug_assert_bindings(client, 3);
        // SAFETY: the kernel touches i < len only: it reads dofmap[i] with len ≤ dofmap.len()
        // (checked above), and writes keys and payload at start + i < total; keys has one
        // plane of pad_len(total) ≥ total words, payload total elements.
        unsafe {
            occurrences_kernel::launch_unchecked(
                client,
                plan.cube_count,
                plan.cube_dim,
                dofmap.arg(),
                keys.words_mut().arg(),
                payload.arg(),
                fit(len)?,
                fit(occ.start)?,
                plan.items_per_unit,
            );
        }
    }
    let (sorted, payload) = radix_sort(&keys, &payload)?;
    let unique = unique_by_key(&sorted, &payload)?; // the count read
    let count = unique.first.len();

    // 2. and 3. The interpolant at the lattice, block by block, and its first occurrences.
    let zero_origin = DeviceBuffer::<G::F>::zeros(client, gdim)?;
    let mut lattices: Vec<PointSet<G::F, G::TD>> = Vec::new();
    for occ in &blocks {
        let element = grid.geometry_block(occ.block).element;
        // The node count grows with the degree, so (cell, k) identifies the lattice.
        let index = match lattices
            .iter()
            .position(|ps| ps.cell() == element.cell && ps.len() == occ.k)
        {
            Some(i) => i,
            None => {
                let cell: CellType = element.cell.into();
                let points = lagrange_points(cell, element.degree, NodeFamily::Equispaced)
                    .map_err(element_error)?;
                lattices.push(PointSet::lattice(client, element.cell, points)?);
                lattices.len() - 1
            }
        };
        let x: QuadField<G::F, VecShape<G::GD>> = launch_factor(
            grid,
            occ.block,
            &lattices[index],
            Factor::Points,
            Some(&zero_origin),
        )?;
        let tile = x.tile();
        for chunk in x.chunks().iter().filter(|c| c.num_cells > 0) {
            let start = chunk
                .first_cell
                .checked_mul(occ.k)
                .and_then(|r| occ.start.checked_add(r))
                .ok_or(GridError::TooLarge)?;
            let end = chunk
                .num_cells
                .checked_mul(occ.k)
                .and_then(|r| start.checked_add(r))
                .ok_or(GridError::TooLarge)?;
            fit(chunk.data.len())?;
            let plan = LaunchPlan::elementwise(
                client,
                KernelKind::Scalar,
                count,
                size_of::<G::F>(),
                1,
                1,
            )?;
            debug_assert_bindings(client, 4);
            // SAFETY: the kernel touches groups u < count only, reading first[u] and nodes[u]
            // (plane 0 of the unique keys, pad_len(count) words). For a first occurrence f in
            // [start, end), r = f − start < chunk.num_cells · k, so chunk cell c < num_cells
            // and point q < k; the field index ((c / tile · k + q) · G + g) · tile + c mod tile
            // lies below the chunk's whole tiles · k · G values. It writes out[node · G + g]
            // only for node < num_nodes = nodes.len() / G, within `out`.
            unsafe {
                select_kernel::launch_unchecked::<G::F>(
                    client,
                    plan.cube_count,
                    plan.cube_dim,
                    unique.first.arg(),
                    unique.keys.words().arg(),
                    chunk.data.arg(),
                    out.arg(),
                    fit(count)?,
                    fit(start)?,
                    fit(end)?,
                    num_nodes,
                    fit(occ.k)?,
                    fit(tile)?,
                    plan.items_per_unit,
                    gdim,
                );
            }
        }
    }
    Ok(out)
}
