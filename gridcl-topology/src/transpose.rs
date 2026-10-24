//! The cell-fastest copy of the cell–vertex connectivity ("Where vectors apply"; ADR-0015,
//! item 8). Above vector size 1, TopologyKeys and PermutationInfo read their cells as vectors
//! of consecutive cells, so they need the connectivity as planes `[local vertex][padded cell]`:
//! word j of cell c at j · padded + c, with padded = pad_len(cells). Padding cells (c ≥ cells)
//! copy the last cell, as E-vectors do, so every lane reads a real cell.
//!
//! The copy is transient: built right before each candidate launch at N > 1 and dropped after
//! it. It is an indirect copy, so it is scalar and plans as `KernelKind::Scalar`; its time
//! counts towards the family, because the `bench` hooks time the whole sequence.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, KernelKind, LaunchPlan, Result, debug_assert_bindings,
    pad_len,
};

/// Cell-major rows to vertex planes: `out[j · padded + c] = rows[min(c, cells − 1) · nv + j]`
/// for every padded cell c < padded. Scalar: one cell per item, so each CPU thread writes a
/// contiguous run of every plane.
#[cube(launch_unchecked)]
fn transpose_vertices_kernel(
    rows: &[u32],
    out: &mut [u32],
    num_cells: u32,
    padded: u32,
    items_per_unit: u32,
    #[comptime] nv: usize,
) {
    let last = num_cells - 1;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < padded as usize {
            let src = min(c as u32, last) as usize * nv;
            #[unroll]
            for j in 0..nv {
                out[j * padded as usize + c] = rows[src + j];
            }
        }
    }
}

/// Cell-major rows to planes of their vertices' origin-id words: the lo word of vertex j of
/// cell c at j · padded + c and, unless `fast`, the hi word at (nv + j) · padded + c. Rows are
/// read as in [`transpose_vertices_kernel`]; each vertex index is clamped to `last_vertex`
/// before it reads `ids`, as the cell-major PermutationInfo kernel clamps it.
///   ids  interleaved (hi, lo) origin id words per local vertex
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn transpose_ids_kernel(
    rows: &[u32],
    ids: &[u32],
    out: &mut [u32],
    num_cells: u32,
    last_vertex: u32,
    padded: u32,
    items_per_unit: u32,
    #[comptime] nv: usize,
    #[comptime] fast: bool,
) {
    let last = num_cells - 1;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let c = first + it;
        if c < padded as usize {
            let src = min(c as u32, last) as usize * nv;
            #[unroll]
            for j in 0..nv {
                let vertex = min(rows[src + j], last_vertex) as usize;
                out[j * padded as usize + c] = ids[2 * vertex + 1];
                if comptime!(!fast) {
                    out[(nv + j) * padded as usize + c] = ids[2 * vertex];
                }
            }
        }
    }
}

/// The plan of a copy of `cells` cells: one unit per padded cell. The candidate kernel plans
/// after it, and its vector size decides whether the copy runs; at size 1 the plan is made but
/// not launched. `record_plans` therefore lists it directly before the candidate's plan.
///
/// # Errors
///
/// `TooLarge` as for [`LaunchPlan::elementwise`].
pub(crate) fn plan(client: &Client, cells: usize) -> Result<LaunchPlan> {
    LaunchPlan::elementwise(client, KernelKind::Scalar, cells, size_of::<u32>(), 1, 1)
}

/// Checked u32 sizes of a copy: (cells, padded, words of `planes` planes).
fn sizes(cells: usize, planes: usize) -> Result<(u32, u32, usize)> {
    let fit = |x: usize| u32::try_from(x).map_err(|_| GridError::TooLarge);
    let padded = pad_len(cells)?;
    let words = planes.checked_mul(padded).ok_or(GridError::TooLarge)?;
    fit(words)?;
    Ok((fit(cells)?, fit(padded)?, words))
}

/// The vertex planes of the `cells` cells of `rows` (stride nv), launched with `plan` from
/// [`plan`]: nv · pad_len(cells) words.
///
/// # Errors
///
/// `TooLarge` if the copy's length overflows u32; allocation and device errors.
pub(crate) fn vertices(
    plan: &LaunchPlan,
    rows: &DeviceStrided<u32>,
    cells: usize,
) -> Result<DeviceBuffer<u32>> {
    let client = rows.data.client();
    let nv = rows.stride as usize;
    let (num_cells, padded, words) = sizes(cells, nv)?;
    let out = DeviceBuffer::<u32>::uninit(client, words)?;
    if cells == 0 {
        return Ok(out);
    }
    debug_assert_eq!(plan.num_vectors, padded, "a plan of another cell count");
    debug_assert!(rows.data.len() >= cells * nv, "fewer rows than cells");
    debug_assert_bindings(client, 2);
    // SAFETY: the kernel visits c < padded = plan.num_vectors (vector size 1). It reads `rows`
    // at min(c, cells − 1) · nv + j with j < nv, within the cells · nv entries of the block,
    // and writes `out` at j · padded + c < nv · padded = words, its length; words fits in u32.
    unsafe {
        transpose_vertices_kernel::launch_unchecked(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            rows.data.arg(),
            out.arg(),
            num_cells,
            padded,
            plan.items_per_unit,
            nv,
        );
    }
    Ok(out)
}

/// The origin-id planes of the `cells` cells of `rows` (stride nv), launched with `plan` from
/// [`plan`]: nv lo planes, then (unless `fast`) nv hi planes, of pad_len(cells) words each.
/// Vertex indices above `last_vertex` read the ids of `last_vertex`.
///
/// # Errors
///
/// `TooLarge` if the copy's length overflows u32; allocation and device errors.
pub(crate) fn origin_ids(
    plan: &LaunchPlan,
    rows: &DeviceStrided<u32>,
    cells: usize,
    ids: &DeviceBuffer<u32>,
    last_vertex: u32,
    fast: bool,
) -> Result<DeviceBuffer<u32>> {
    let client = rows.data.client();
    let nv = rows.stride as usize;
    let planes = if fast { nv } else { 2 * nv };
    let (num_cells, padded, words) = sizes(cells, planes)?;
    let out = DeviceBuffer::<u32>::uninit(client, words)?;
    if cells == 0 {
        return Ok(out);
    }
    debug_assert_eq!(plan.num_vectors, padded, "a plan of another cell count");
    debug_assert!(rows.data.len() >= cells * nv, "fewer rows than cells");
    debug_assert_bindings(client, 3);
    // SAFETY: rows and the bounds of c as in `vertices`. It reads `ids` at 2 · vertex and
    // 2 · vertex + 1 with vertex = min(row entry, last_vertex): the callers pass
    // last_vertex = ids.len() / 2 − 1, or 0 with no vertices, when it reads words 0 and 1 of an
    // allocation of at least VECTOR_PAD words. It writes `out` at p · padded + c with
    // p < planes, below words = planes · padded, its length; words fits in u32.
    unsafe {
        transpose_ids_kernel::launch_unchecked(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            rows.data.arg(),
            ids.arg(),
            out.arg(),
            num_cells,
            last_vertex,
            padded,
            plan.items_per_unit,
            nv,
            fast,
        );
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encode_origin_ids;

    /// `cells` rows of `nv` pseudo-random vertex indices below `num_vertices`.
    fn rows(cells: usize, nv: usize, num_vertices: u32) -> Vec<u32> {
        let mut x = 0x9e37_79b9_7f4a_7c15u64 ^ cells as u64;
        (0..cells * nv)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x % u64::from(num_vertices)) as u32
            })
            .collect()
    }

    fn upload(client: &Client, rows: &[u32], nv: usize) -> DeviceStrided<u32> {
        DeviceStrided {
            data: DeviceBuffer::from_host(client, rows).unwrap(),
            stride: nv as u32,
        }
    }

    /// The copy read back cell-major equals the input rows, and every padding cell repeats the
    /// last cell; `word` maps a vertex index to the copied word.
    fn check_planes(got: &[u32], rows: &[u32], nv: usize, cells: usize, word: impl Fn(u32) -> u32) {
        let padded = pad_len(cells).unwrap();
        assert_eq!(got.len(), nv * padded);
        if cells == 0 {
            return;
        }
        let back: Vec<u32> = (0..cells)
            .flat_map(|c| (0..nv).map(move |j| (c, j)))
            .map(|(c, j)| got[j * padded + c])
            .collect();
        let want: Vec<u32> = rows.iter().map(|&v| word(v)).collect();
        assert!(back == want, "{cells} cells: the copy does not round-trip");
        for c in cells..padded {
            for j in 0..nv {
                assert_eq!(
                    got[j * padded + c],
                    got[j * padded + cells - 1],
                    "{cells} cells: padding cell {c}"
                );
            }
        }
    }

    #[test]
    fn transpose_round_trips() {
        let client = gridcl_testing::test_client();
        for (cells, nv) in [(0, 3), (1, 4), (15, 8), (16, 5), (17, 6), (1_000_003, 3)] {
            let host = rows(cells, nv, 1000);
            let block = upload(&client, &host, nv);
            let p = plan(&client, cells).unwrap();
            assert_eq!(p.vector_size, 1);
            let got = vertices(&p, &block, cells).unwrap().to_host().unwrap();
            check_planes(&got, &host, nv, cells, |v| v);
        }
    }

    #[test]
    fn origin_id_planes_round_trip_on_both_id_paths() {
        let client = gridcl_testing::test_client();
        let num_vertices = 50u32;
        for (cells, nv) in [(0, 3), (1, 4), (15, 8), (16, 5), (17, 6)] {
            // Rows reach past the last vertex: those indices read the last vertex's ids.
            let host = rows(cells, nv, num_vertices + 3);
            let block = upload(&client, &host, nv);
            let last = num_vertices - 1;
            let clamp = move |v: u32| v.min(last) as usize;
            for fast in [true, false] {
                let ids: Vec<i64> = (0..i64::from(num_vertices))
                    .map(|i| if fast { 7 * i } else { (i - 20) << 33 | i })
                    .collect();
                let words = encode_origin_ids(&ids);
                let id_buffer = DeviceBuffer::from_host(&client, &words).unwrap();
                let p = plan(&client, cells).unwrap();
                let got = origin_ids(&p, &block, cells, &id_buffer, last, fast)
                    .unwrap()
                    .to_host()
                    .unwrap();
                let padded = pad_len(cells).unwrap();
                let planes = if fast { 1 } else { 2 };
                assert_eq!(got.len(), planes * nv * padded);
                let (lo, hi) = got.split_at(nv * padded);
                check_planes(lo, &host, nv, cells, |v| words[2 * clamp(v) + 1]);
                if !fast {
                    check_planes(hi, &host, nv, cells, |v| words[2 * clamp(v)]);
                }
            }
        }
    }
}
