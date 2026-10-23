//! Entity creation, step 2: the TopologyKeys kernel, generic over the vector size. Each key is
//! the sub-entity's local vertex ids sorted by a sorting network; the payload is the emission
//! index, and padding positions write an all-ones sentinel key.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, KernelKind, LaunchPlan, Result, SortKeys, VECTOR_PAD,
    debug_assert_bindings,
};
use gridcl_dims::CellType;

use crate::emission::{EmissionPlan, sub_entity_vertices};

/// The comptime tables of one launch: the cell's vertex count and the local vertex lists of
/// its sub-entities of the emitted type, all of `words` vertices.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct KeyTables {
    pub nv: usize,
    pub entities: Vec<Vec<usize>>,
    pub words: usize,
}

/// Exchanges `a` and `b` lane by lane where b < a: one comparator of a sorting network.
#[cube]
fn sort2<N: Size>(a: &mut Vector<u32, N>, b: &mut Vector<u32, N>) {
    let lo = min(*a, *b);
    let hi = max(*a, *b);
    *a = lo;
    *b = hi;
}

/// Keys and payloads of every cell vector `v < num_vectors` (Listing B5 loop) of one block:
/// sub-entity i of cell c goes to emission index start + i · padded_cells + c. Lanes of cells
/// c ≥ num_cells write the all-ones sentinel key; they read the last cell, so reads stay in
/// bounds. Vertex indices are clamped to `last_vertex` lane by lane (an arithmetic `min`), as
/// the head and PermutationInfo kernels clamp them before reading origin ids, so all three
/// treat an index past the local vertices as the last vertex, and no real key reaches the
/// sentinel while `last_vertex` < u32::MAX.
///   vertices  [cell][nv] local vertex indices
///   keys      planar words: word w of key e at w · plane + e
///   payload   one emission index per key
#[cube(launch_unchecked)]
fn topology_keys_kernel<N: Size>(
    vertices: &[u32],
    keys: &mut [Vector<u32, N>],
    payload: &mut [Vector<u32, N>],
    num_cells: u32,
    last_vertex: u32,
    padded_cells: u32,
    start: u32,
    plane: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] t: KeyTables,
) {
    let lanes = N::value();
    let nv = t.nv;
    let last = num_cells - 1;
    let all_ones = Vector::<u32, N>::broadcast(u32::MAX);
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let v = first + it;
        if v < num_vectors as usize {
            // The cell index of each lane.
            let mut cell = Vector::<u32, N>::broadcast(0u32);
            #[unroll]
            for lane in 0..lanes {
                cell.insert(lane, (v * lanes + lane) as u32);
            }
            let real = cell.less_than(&Vector::<u32, N>::broadcast(num_cells));
            #[unroll]
            for i in 0..comptime!(t.entities.len()) {
                let mut w = Array::<Vector<u32, N>>::new(comptime!(t.words));
                #[unroll]
                for j in 0..comptime!(t.words) {
                    let local = comptime!(t.entities[i][j]);
                    let mut x = Vector::<u32, N>::broadcast(0u32);
                    #[unroll]
                    for lane in 0..lanes {
                        let c = min((v * lanes + lane) as u32, last) as usize;
                        x.insert(lane, min(vertices[c * nv + local], last_vertex));
                    }
                    w[j] = x;
                }
                // Sorting networks of 2, 3 and 4 values.
                if comptime!(t.words == 2) {
                    let (mut a, mut b) = (w[0], w[1]);
                    sort2::<N>(&mut a, &mut b);
                    w[0] = a;
                    w[1] = b;
                } else if comptime!(t.words == 3) {
                    let (mut a, mut b, mut c) = (w[0], w[1], w[2]);
                    sort2::<N>(&mut a, &mut b);
                    sort2::<N>(&mut b, &mut c);
                    sort2::<N>(&mut a, &mut b);
                    w[0] = a;
                    w[1] = b;
                    w[2] = c;
                } else {
                    let (mut a, mut b, mut c, mut d) = (w[0], w[1], w[2], w[3]);
                    sort2::<N>(&mut a, &mut b);
                    sort2::<N>(&mut c, &mut d);
                    sort2::<N>(&mut a, &mut c);
                    sort2::<N>(&mut b, &mut d);
                    sort2::<N>(&mut b, &mut c);
                    w[0] = a;
                    w[1] = b;
                    w[2] = c;
                    w[3] = d;
                }
                // Emission vector of sub-entity i: (start + i · padded_cells) / N + v.
                let base = (start as usize + comptime!(i) * padded_cells as usize) / lanes + v;
                #[unroll]
                for j in 0..comptime!(t.words) {
                    keys[j * (plane as usize / lanes) + base] = select_many(real, w[j], all_ones);
                }
                let e =
                    Vector::<u32, N>::broadcast(start + comptime!(i as u32) * padded_cells) + cell;
                payload[base] = e;
            }
        }
    }
}

/// The keys and payloads of `plan`: one launch per cell block of the plan. `blocks` gives the
/// cell type and vertex rows of every cell block, indexed like the plan's ranges' `block`;
/// vertex indices above `last_vertex` count as `last_vertex`.
///
/// # Errors
///
/// Allocation and device errors; `TooLarge` beyond u32 indices.
pub(crate) fn emit_keys(
    client: &Client,
    blocks: &[(CellType, &DeviceStrided<u32>)],
    dim: usize,
    plan: &EmissionPlan,
    last_vertex: u32,
) -> Result<(SortKeys, DeviceBuffer<u32>)> {
    let words = gridcl_refcell::vertices(plan.entity).len();
    let mut keys = SortKeys::uninit(client, plan.total, words)?;
    let payload = DeviceBuffer::<u32>::uninit(client, plan.total)?;
    let plane =
        u32::try_from(gridcl_device::pad_len(plan.total)?).map_err(|_| GridError::TooLarge)?;
    for range in &plan.ranges {
        if range.cells == 0 {
            continue;
        }
        let (cell, vertices) = blocks[range.block];
        let tables = KeyTables {
            nv: gridcl_refcell::vertices(cell).len(),
            entities: sub_entity_vertices(cell, dim, &range.local),
            words,
        };
        let fit = |x: usize| u32::try_from(x).map_err(|_| GridError::TooLarge);
        let launch = LaunchPlan::elementwise(
            client,
            KernelKind::TopologyKeys,
            range.cells,
            size_of::<u32>(),
            VECTOR_PAD,
            2 * words + 2,
        )?;
        let n = launch.vector_size;
        debug_assert_bindings(client, 3);
        // SAFETY: the plan covers pad_len(cells) = padded cells, so the kernel's vectors
        // v < num_vectors index cells below `padded`. It reads `vertices` at c · nv + j with
        // c ≤ cells − 1, within the block's rows. Sub-entity i writes emission vectors
        // (start + i · padded) / n + v < end / n ≤ total / n, and key words at
        // w · plane / n + that index < words · plane / n, within `keys` (words planes of
        // `plane` = pad_len(total) words) and `payload` (pad_len(total) words); start, padded and
        // plane are multiples of VECTOR_PAD, which n divides.
        unsafe {
            topology_keys_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                n,
                vertices.data.arg(),
                keys.words_mut().vector_arg(n),
                payload.vector_arg(n),
                fit(range.cells)?,
                last_vertex,
                fit(range.padded)?,
                fit(range.start)?,
                plane,
                launch.num_vectors,
                launch.items_per_unit,
                tables,
            );
        }
    }
    Ok((keys, payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::emission::emission_plans;
    use CellType::*;

    /// Host reference: the keys (key-major) and payloads of `plan`.
    fn reference(
        blocks: &[(CellType, Vec<u32>)],
        dim: usize,
        plan: &EmissionPlan,
    ) -> (Vec<u32>, Vec<u32>) {
        let words = gridcl_refcell::vertices(plan.entity).len();
        let mut keys = vec![u32::MAX; plan.total * words];
        let mut payload = vec![0u32; plan.total];
        for r in &plan.ranges {
            let (cell, rows) = &blocks[r.block];
            let nv = gridcl_refcell::vertices(*cell).len();
            let lists = sub_entity_vertices(*cell, dim, &r.local);
            for (i, list) in lists.iter().enumerate() {
                for c in 0..r.padded {
                    let e = r.start + i * r.padded + c;
                    payload[e] = e as u32;
                    if c < r.cells {
                        let mut key: Vec<u32> = list.iter().map(|&l| rows[c * nv + l]).collect();
                        key.sort_unstable();
                        keys[e * words..(e + 1) * words].copy_from_slice(&key);
                    }
                }
            }
        }
        (keys, payload)
    }

    fn check(blocks: &[(CellType, Vec<u32>)], dim: usize) {
        let client = gridcl_testing::test_client();
        let strided: Vec<DeviceStrided<u32>> = blocks
            .iter()
            .map(|(cell, rows)| DeviceStrided {
                data: DeviceBuffer::from_host(&client, rows).unwrap(),
                stride: gridcl_refcell::vertices(*cell).len() as u32,
            })
            .collect();
        let refs: Vec<(CellType, &DeviceStrided<u32>)> =
            blocks.iter().map(|(c, _)| *c).zip(strided.iter()).collect();
        let counts: Vec<(CellType, usize)> = blocks
            .iter()
            .map(|(c, rows)| (*c, rows.len() / gridcl_refcell::vertices(*c).len()))
            .collect();
        for plan in emission_plans(&counts, dim).unwrap() {
            let (keys, payload) = emit_keys(&client, &refs, dim, &plan, 999).unwrap();
            let (want_keys, want_payload) = reference(blocks, dim, &plan);
            assert_eq!(keys.to_host().unwrap(), want_keys, "{:?} keys", plan.entity);
            assert_eq!(
                payload.to_host().unwrap(),
                want_payload,
                "{:?} payload",
                plan.entity
            );
        }
    }

    /// `n` cells of `cell` with pseudo-random distinct vertices out of 1000.
    fn rows(cell: CellType, n: usize, seed: u64) -> Vec<u32> {
        let nv = gridcl_refcell::vertices(cell).len();
        let mut x = seed | 1;
        let mut out = Vec::new();
        for _ in 0..n {
            let mut row: Vec<u32> = Vec::new();
            while row.len() < nv {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let v = (x % 1000) as u32;
                if !row.contains(&v) {
                    row.push(v);
                }
            }
            out.extend(row);
        }
        out
    }

    #[test]
    fn keys_match_the_host_reference() {
        check(
            &[
                (Triangle, rows(Triangle, 21, 1)),
                (Quadrilateral, rows(Quadrilateral, 16, 2)),
            ],
            1,
        );
        check(
            &[
                (Tetrahedron, rows(Tetrahedron, 5, 3)),
                (Hexahedron, rows(Hexahedron, 17, 4)),
                (Prism, rows(Prism, 0, 5)),
                (Pyramid, rows(Pyramid, 33, 6)),
            ],
            2,
        );
        check(&[(Prism, rows(Prism, 40, 7))], 1);
    }

    #[test]
    fn every_vector_size_gives_the_same_keys() {
        use gridcl_device::VectorSizeOverride;
        use gridcl_device::testing::{record_plans, with_vector_size};
        let client = gridcl_testing::test_client();
        let widest = client.io_optimized_vector_sizes(4).next().unwrap_or(1);
        for n in [1, 2, 4, 8, 16]
            .into_iter()
            .filter(|&n| n == 1 || n <= widest)
        {
            let ((), plans) = record_plans(|| {
                with_vector_size(VectorSizeOverride::Fixed(n), || {
                    check(
                        &[
                            (Hexahedron, rows(Hexahedron, 17, 8)),
                            (Pyramid, rows(Pyramid, 15, 9)),
                        ],
                        2,
                    );
                })
            });
            let p = plans
                .iter()
                .find(|(k, _)| *k == KernelKind::TopologyKeys)
                .unwrap();
            assert_eq!(p.1.vector_size, n);
        }
    }
}
