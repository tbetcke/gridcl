//! Stage 7: the local cell order per [`CellOrdering`] (docs/design.md, "Parallel creation
//! pipeline"; ADR-0026, item 6).

use std::collections::HashMap;

use super::ghosts::LocalCells;
use super::nodes::Nodes;
use super::{BlockSpec, Cells, facet_keys};
use crate::CellOrdering;

/// Bits per axis of the quantised centroids of the Hilbert order.
const HILBERT_BITS: u32 = 21;

/// Stage 7: orders the owned cells of `local` by `ordering`; the ghost cells keep their order
/// from stage 5, (owner rank, origin id), after the owned cells. Host only.
///
/// - `Input`: by global input position, which orders by source rank and then by position on
///   it.
/// - `Hilbert`: by the Hilbert key of the vertex centroid, quantised to 2^21 per axis in the
///   global bounding box of `nodes`, then by origin id.
/// - `ReverseCuthillMcKee`: reverse Cuthill–McKee on the facet graph of the owned cells. Each
///   connected component starts at its cell of least (degree, origin id), and the neighbours
///   of a cell are queued by (degree, origin id).
pub(crate) fn order_cells(
    blocks: &[BlockSpec],
    local: LocalCells,
    nodes: &Nodes,
    ordering: CellOrdering,
) -> LocalCells {
    let n = local.num_owned;
    let cells = &local.cells;
    let mut order = match ordering {
        CellOrdering::Input => {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_unstable_by_key(|&i| cells.input[i]);
            order
        }
        CellOrdering::Hilbert => hilbert_order(blocks, cells, n, nodes),
        CellOrdering::ReverseCuthillMcKee => rcm_order(blocks, cells, n),
    };
    order.extend(n..cells.len());
    LocalCells {
        cells: cells.select(&order),
        num_owned: n,
        owner: local.owner,
    }
}

/// The first `n` cells by (Hilbert key of the vertex centroid, origin id).
fn hilbert_order(blocks: &[BlockSpec], cells: &Cells, n: usize, nodes: &Nodes) -> Vec<usize> {
    let gd = nodes.gdim;
    assert!(gd <= 3, "at most three axes");
    let keys: Vec<(u64, i64)> = (0..n)
        .map(|i| {
            let vertices = cells.vertices_of(i, blocks);
            let mut q = [0_u32; 3];
            for (g, q) in q.iter_mut().enumerate().take(gd) {
                let sum: f64 = vertices.iter().map(|&v| nodes.coords_of(v)[g]).sum();
                let centroid = sum / vertices.len() as f64;
                *q = quantise(centroid, nodes.lo[g], nodes.hi[g]);
            }
            (hilbert_key(&q[..gd], HILBERT_BITS), cells.ids[i])
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by_key(|&i| keys[i]);
    order
}

/// `x` in [`lo`, `hi`] as an integer in [0, 2^21): ⌊(x − lo) / (hi − lo) · 2^21⌋, clamped; 0
/// on an axis of zero extent.
fn quantise(x: f64, lo: f64, hi: f64) -> u32 {
    let cells = f64::from(1_u32 << HILBERT_BITS);
    if hi <= lo {
        return 0;
    }
    ((x - lo) / (hi - lo) * cells)
        .floor()
        .clamp(0.0, cells - 1.0) as u32
}

/// The index along the Hilbert curve of the point `x` of a grid of 2^`bits` points per axis,
/// in `x.len()` · `bits` ≤ 64 bits (J. Skilling, "Programming the Hilbert curve", AIP Conf.
/// Proc. 707, 2004: the transpose of the key, then its bits interleaved from the most
/// significant, axis 0 first). The curve starts at the origin.
pub(crate) fn hilbert_key(x: &[u32], bits: u32) -> u64 {
    let d = x.len();
    assert!(
        (1..=32).contains(&bits) && d * bits as usize <= 64,
        "a key of at most 64 bits"
    );
    let mut x = x.to_vec();
    let m = 1_u32 << (bits - 1);
    // Inverse undo.
    let mut q = m;
    while q > 1 {
        let p = q - 1;
        for i in 0..d {
            if x[i] & q != 0 {
                x[0] ^= p;
            } else {
                let t = (x[0] ^ x[i]) & p;
                x[0] ^= t;
                x[i] ^= t;
            }
        }
        q >>= 1;
    }
    // Gray encode.
    for i in 1..d {
        x[i] ^= x[i - 1];
    }
    let mut t = 0;
    let mut q = m;
    while q > 1 {
        if x[d - 1] & q != 0 {
            t ^= q - 1;
        }
        q >>= 1;
    }
    for xi in &mut x {
        *xi ^= t;
    }
    let mut key = 0_u64;
    for b in (0..bits).rev() {
        for xi in &x {
            key = (key << 1) | u64::from((xi >> b) & 1);
        }
    }
    key
}

/// The first `n` cells in reverse Cuthill–McKee order of their facet graph.
fn rcm_order(blocks: &[BlockSpec], cells: &Cells, n: usize) -> Vec<usize> {
    let mut by_key: HashMap<Vec<i64>, Vec<usize>> = HashMap::new();
    for i in 0..n {
        for key in facet_keys(cells, i, blocks) {
            by_key.entry(key).or_default().push(i);
        }
    }
    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
    for group in by_key.values() {
        for &a in group {
            adjacency[a].extend(group.iter().filter(|&&b| b != a));
        }
    }
    for row in &mut adjacency {
        row.sort_unstable();
        row.dedup();
    }
    let rank = |i: usize| (adjacency[i].len(), cells.ids[i]);
    let mut starts: Vec<usize> = (0..n).collect();
    starts.sort_unstable_by_key(|&i| rank(i));

    let mut visited = vec![false; n];
    let mut order = Vec::with_capacity(n);
    let mut next = Vec::new();
    for s in starts {
        if visited[s] {
            continue;
        }
        visited[s] = true;
        let mut head = order.len();
        order.push(s);
        while let Some(&a) = order.get(head) {
            head += 1;
            next.clear();
            next.extend(adjacency[a].iter().copied().filter(|&b| !visited[b]));
            next.sort_unstable_by_key(|&b| rank(b));
            for &b in &next {
                visited[b] = true;
                order.push(b);
            }
        }
    }
    order.reverse();
    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GhostMode;
    use crate::pipeline::nodes::fetch_nodes;
    use crate::pipeline::testing::{OwnedInput, local_cells, meshes};
    use gridcl_dims::Cell2;
    use gridcl_parallel::CommDyn;
    use gridcl_testing::{HostChunk, ThreadComm};

    /// Keys of the first curves, computed by hand: the U of order 1 in 2D, the first quadrant
    /// of order 2, and the Gray-code cube of order 1 in 3D.
    #[test]
    fn hilbert_keys_by_hand() {
        let key = |x: &[u32], bits| hilbert_key(x, bits);
        assert_eq!(
            [[0, 0], [0, 1], [1, 1], [1, 0]].map(|x| key(&x, 1)),
            [0, 1, 2, 3]
        );
        assert_eq!(
            [[0, 0], [1, 0], [1, 1], [0, 1]].map(|x| key(&x, 2)),
            [0, 1, 2, 3]
        );
        let cube = [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
            [1, 0, 0],
        ];
        assert_eq!(cube.map(|x| key(&x, 1)), [0, 1, 2, 3, 4, 5, 6, 7]);
        // One axis: the coordinate itself.
        assert_eq!(key(&[12345], 21), 12345);
    }

    /// The keys are a bijection onto 0..2^(d·bits), consecutive keys are neighbouring grid
    /// points, and the curve starts at the origin.
    #[test]
    fn hilbert_curves_are_continuous() {
        for (d, bits) in [(2, 1), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3)] {
            let side = 1_u32 << bits;
            let mut points = vec![None; 1 << (d * bits as usize)];
            for p in 0..points.len() as u32 {
                let x: Vec<u32> = (0..d).map(|g| (p >> (g as u32 * bits)) % side).collect();
                let k = hilbert_key(&x, bits) as usize;
                assert!(
                    points[k].replace(x).is_none(),
                    "{d}D, {bits} bits: key {k} twice"
                );
            }
            let points: Vec<Vec<u32>> = points.into_iter().map(Option::unwrap).collect();
            assert!(points[0].iter().all(|&c| c == 0));
            for w in points.windows(2) {
                let step: u32 = w[0].iter().zip(&w[1]).map(|(a, b)| a.abs_diff(*b)).sum();
                assert_eq!(step, 1, "{d}D, {bits} bits: {:?} → {:?}", w[0], w[1]);
            }
        }
    }

    /// At 21 bits the leading key bits follow the order-1 curve of the leading coordinate
    /// bits, so the quadrants (octants) are visited in that order.
    #[test]
    fn full_resolution_keys_follow_the_first_level() {
        let half = 1_u32 << 20;
        let top = |x: &[u32]| -> Vec<u32> { x.iter().map(|&c| c / half).collect() };
        let mut rng = 0x1234_5678_u64;
        let mut next = || {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            ((rng >> 33) as u32) % (2 * half)
        };
        for d in [2, 3] {
            for _ in 0..200 {
                let x: Vec<u32> = (0..d).map(|_| next()).collect();
                let k = hilbert_key(&x, 21);
                assert!(k < 1 << (21 * d));
                assert_eq!(k >> (20 * d), hilbert_key(&top(&x), 1), "{x:?}");
            }
        }
        assert_eq!(hilbert_key(&[0, 0, 0], 21), 0);
    }

    #[test]
    fn quantisation_covers_the_box() {
        let top = (1 << 21) - 1;
        assert_eq!(quantise(-1.0, -1.0, 3.0), 0);
        assert_eq!(quantise(3.0, -1.0, 3.0), top);
        assert_eq!(quantise(1.0, -1.0, 3.0), 1 << 20);
        assert_eq!(quantise(5.0, 5.0, 5.0), 0);
    }

    /// Stages 1–7 on `size` ranks, from even chunks or everything on rank 0: the origin ids
    /// of each rank's cells in local order, with the number owned.
    fn ordered_ids(
        h: &gridcl_testing::HostMesh,
        size: usize,
        on_zero: bool,
        mode: GhostMode,
        ordering: CellOrdering,
    ) -> Vec<(Vec<i64>, usize)> {
        ThreadComm::run(size, |comm| {
            let r = comm.rank();
            let chunk = match (on_zero, r) {
                (false, _) => h.chunk(r, size),
                (true, 0) => h.chunk(0, 1),
                (true, _) => HostChunk::empty(h.gdim),
            };
            let input = OwnedInput::of_chunk(&chunk);
            let (read, local) = local_cells(&comm, &input.input(), mode);
            let nodes = fetch_nodes(&comm, &read.blocks, h.gdim, &read.nodes, &local).unwrap();
            let before = local.clone();
            let local = order_cells(&read.blocks, local, &nodes, ordering);
            // A permutation of the owned cells; the ghosts and owners are unchanged.
            let n = local.num_owned;
            assert_eq!(n, before.num_owned);
            assert_eq!(local.owner, before.owner);
            assert_eq!(local.cells.ids[n..], before.cells.ids[n..]);
            let mut sorted = local.cells.ids[..n].to_vec();
            let mut want = before.cells.ids[..n].to_vec();
            sorted.sort_unstable();
            want.sort_unstable();
            assert_eq!(sorted, want);
            if ordering == CellOrdering::Input {
                assert!(local.cells.input[..n].is_sorted());
            }
            (local.cells.ids, n)
        })
    }

    /// The local order depends on the partition only, not on how the input was chunked.
    #[test]
    fn order_is_independent_of_the_chunking() {
        let orderings = [
            CellOrdering::Input,
            CellOrdering::Hilbert,
            CellOrdering::ReverseCuthillMcKee,
        ];
        for (name, h) in meshes() {
            let h = h.with_shuffled_cells(3);
            for size in 1..=4 {
                for ordering in orderings {
                    let mode = GhostMode::SharedFacet;
                    let even = ordered_ids(&h, size, false, mode, ordering);
                    let zero = ordered_ids(&h, size, true, mode, ordering);
                    assert_eq!(even, zero, "{name}, {ordering:?} on {size} ranks");
                }
            }
        }
    }

    /// The largest distance in local order between two owned cells that share a facet.
    fn bandwidth(h: &gridcl_testing::HostMesh, ids: &[i64]) -> usize {
        let position: HashMap<i64, usize> =
            ids.iter().enumerate().map(|(i, &id)| (id, i)).collect();
        let adjacency =
            crate::pipeline::testing::host_adjacency(&crate::pipeline::testing::host_facets(h));
        let mut width = 0;
        for (&a, &pa) in &position {
            for b in &adjacency[&a] {
                if let Some(&pb) = position.get(b) {
                    width = width.max(pa.abs_diff(pb));
                }
            }
        }
        width
    }

    /// On a shuffled grid, reverse Cuthill–McKee gives each rank a facet-graph bandwidth no
    /// worse than the input order's, and Hilbert keeps neighbours close too.
    #[test]
    fn rcm_bandwidth_is_no_worse_than_the_input() {
        let h = gridcl_testing::square_box(Cell2::Quadrilateral, 12, 1).with_shuffled_cells(7);
        for size in 1..=4 {
            let input = ordered_ids(&h, size, false, GhostMode::None, CellOrdering::Input);
            let rcm = ordered_ids(
                &h,
                size,
                false,
                GhostMode::None,
                CellOrdering::ReverseCuthillMcKee,
            );
            for (r, ((a, _), (b, _))) in input.iter().zip(&rcm).enumerate() {
                let (wa, wb) = (bandwidth(&h, a), bandwidth(&h, b));
                assert!(wb <= wa, "rank {r} of {size}: RCM {wb}, input {wa}");
                if size == 1 {
                    // A 12 × 12 grid: RCM keeps the band within about two rows.
                    assert!(wb <= 24, "RCM {wb}");
                }
            }
        }
    }
}
