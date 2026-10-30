//! `DistGraph` and `Partitioner` (Listing S3), and the crate-private graph checks: `Block`'s
//! shape checks and the full validation of every external partitioner (ADR-0029, item 9).

use std::borrow::Cow;

use gridcl_parallel::{CommDyn, CommExt, GridError, Result};

/// A distributed CSR graph in the ParMETIS and PT-Scotch convention: rank r holds the global
/// vertices `vtxdist[r]..vtxdist[r + 1]`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DistGraph {
    /// The first global vertex of every rank, and the total; `nranks + 1` entries.
    pub vtxdist: Vec<i64>,
    /// Row offsets of the local vertices; `num_local + 1` entries.
    pub xadj: Vec<i64>,
    /// Global ids of the neighbours (global cell ids).
    pub adjncy: Vec<i64>,
    /// Vertex weights, from the cost model per (cell type, degree); `None` for unit weights.
    pub vwgt: Option<Vec<i64>>,
}

/// A graph partitioner.
pub trait Partitioner {
    /// The partitioner's name, for logs.
    fn name(&self) -> &str;
    /// Destination rank of every local graph vertex (cell). Collective.
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>>;
}

/// The agreed result of a collective step (ADR-0026, item 15): every rank learns the lowest
/// failing rank. A failing rank returns its own error, every other rank `RemoteFailure` with
/// the lowest failing rank. Collective.
pub(crate) fn agree<T>(comm: &dyn CommDyn, local: Result<T>) -> Result<T> {
    let lowest = comm.min(if local.is_err() {
        comm.rank() as u64
    } else {
        u64::MAX
    });
    match local {
        Ok(_) if lowest != u64::MAX => Err(GridError::RemoteFailure {
            rank: lowest as usize,
        }),
        other => other,
    }
}

/// `Block`'s local checks (ADR-0026): `nparts` in `1..=i32::MAX`, the lengths and monotonicity
/// of `vtxdist` and `xadj`, and the length of `vwgt`. Returns this rank's global vertex range.
/// Not collective.
///
/// # Errors
///
/// `LengthMismatch` if `nparts` is 0 (expected 1), if `vtxdist` does not have
/// `comm.size() + 1` non-decreasing entries from 0, if `xadj` does not have one more entry than
/// this rank's vertices, starting at 0, not decreasing and ending at `adjncy.len()`, or if
/// `vwgt` does not have one entry per vertex; `TooLarge` if `nparts` exceeds `i32::MAX`.
pub(crate) fn check_shape(comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<(i64, i64)> {
    if nparts == 0 {
        return Err(GridError::LengthMismatch {
            expected: 1,
            found: 0,
        });
    }
    if i32::try_from(nparts).is_err() {
        return Err(GridError::TooLarge);
    }
    let p = comm.size();
    if g.vtxdist.len() != p + 1 {
        return Err(GridError::LengthMismatch {
            expected: p + 1,
            found: g.vtxdist.len(),
        });
    }
    if g.vtxdist[0] != 0 || g.vtxdist.windows(2).any(|w| w[1] < w[0]) {
        return Err(GridError::LengthMismatch {
            expected: 0,
            found: g.vtxdist[0].unsigned_abs() as usize,
        });
    }
    let (begin, end) = (g.vtxdist[comm.rank()], g.vtxdist[comm.rank() + 1]);
    let n = usize::try_from(end - begin).map_err(|_| GridError::TooLarge)?;
    if g.xadj.len() != n + 1 {
        return Err(GridError::LengthMismatch {
            expected: n + 1,
            found: g.xadj.len(),
        });
    }
    let last = g.xadj[n];
    if g.xadj[0] != 0
        || g.xadj.windows(2).any(|w| w[1] < w[0])
        || usize::try_from(last).ok() != Some(g.adjncy.len())
    {
        return Err(GridError::LengthMismatch {
            expected: g.adjncy.len(),
            found: last.unsigned_abs() as usize,
        });
    }
    if let Some(w) = &g.vwgt
        && w.len() != n
    {
        return Err(GridError::LengthMismatch {
            expected: n,
            found: w.len(),
        });
    }
    Ok((begin, end))
}

/// The largest value of a signed integer type of `bits` bits (1 to 64), for example a library's
/// `SCOTCH_NUM_BITS` or `IDX_BITS`.
#[allow(dead_code)] // Until the external partitioners (M7 T3–T9) use it.
pub(crate) const fn signed_max(bits: u32) -> i64 {
    assert!(bits != 0 && bits <= 64);
    i64::MAX >> (64 - bits)
}

/// A graph that passed [`validate`]; the same on every rank except the vertex range.
#[allow(dead_code)] // Until the external partitioners (M7 T3–T9) use it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Checked {
    /// This rank's global vertex range `begin..end`.
    pub begin: i64,
    /// The end of this rank's vertex range.
    pub end: i64,
    /// The global vertex count N, `vtxdist[nranks]`.
    pub num_vertices: i64,
    /// The global number of `adjncy` entries (each undirected edge twice).
    pub num_edges: i64,
    /// Whether any rank has `vwgt`; ranks with `None` then count as unit weights.
    pub weighted: bool,
}

/// This rank's vertex weights: `vwgt`, or unit weights when `weighted` but `vwgt` is `None`;
/// `None` when the graph is unweighted on every rank.
#[allow(dead_code)] // Until the external partitioners (M7 T3–T9) use it.
pub(crate) fn local_weights(g: &DistGraph, weighted: bool) -> Option<Cow<'_, [i64]>> {
    match &g.vwgt {
        Some(w) => Some(Cow::Borrowed(w)),
        None if weighted => Some(Cow::Owned(vec![1; g.xadj.len().saturating_sub(1)])),
        None => None,
    }
}

/// The validation every external partitioner runs before any library call (ADR-0029, item 9),
/// for a library whose index type holds values up to `max`; `name` is the partitioner's name.
/// Collective; the result is agreed (ADR-0026, item 15).
///
/// Checks, in order (each step agreed before the next):
///
/// 1. [`check_shape`] (`Block`'s checks);
/// 2. `vtxdist` agrees with the vertex counts of the ranks (`LengthMismatch`), and N, the edge
///    count (all `adjncy` entries), the sum of the weights (unit weights where `vwgt` is
///    `None`) and `nparts` are at most `max` (`TooLarge`);
/// 3. every `adjncy` entry in `[0, N)` (`IndexOutOfRange`, `usize::MAX` for a negative entry),
///    no self-loop (`PartitionerFailed { name, code: -2 }`), every weight non-negative
///    (`TooLarge`: outside the library's range `[0, max]`), and symmetry: the owner of each
///    edge's far end confirms the reverse edge (`PartitionerFailed { name, code: -3 }`).
///
/// # Errors
///
/// As above, plus the errors of `CommExt::all_to_all_v`. Ranks without a local error return
/// `RemoteFailure { rank }` with the lowest failing rank.
#[allow(dead_code)] // Until the external partitioners (M7 T3–T9) use it.
pub(crate) fn validate(
    comm: &dyn CommDyn,
    g: &DistGraph,
    nparts: usize,
    name: &'static str,
    max: i64,
) -> Result<Checked> {
    let (begin, end) = agree(comm, check_shape(comm, g, nparts))?;
    let checked = agree(comm, check_totals(comm, g, nparts, max, begin, end))?;
    let local = check_entries(g, &checked, name);
    // Every rank takes part in the exchange; a rank with a local error asks nothing.
    let (queries, counts) = match local {
        Ok(()) => reverse_queries(&g.vtxdist, begin, g),
        Err(_) => (Vec::new(), vec![0; comm.size()]),
    };
    let answers = comm.all_to_all_v(&queries, &counts);
    let local = local
        .and(answers)
        .and_then(|(pairs, _)| confirm_reverse(g, begin, &pairs, name));
    agree(comm, local.map(|()| checked))
}

/// Step 2 of [`validate`]: `vtxdist` against the vertex counts of the ranks, and the global
/// totals. Collective (one `all_to_all_v`), not agreed.
fn check_totals(
    comm: &dyn CommDyn,
    g: &DistGraph,
    nparts: usize,
    max: i64,
    begin: i64,
    end: i64,
) -> Result<Checked> {
    let p = comm.size();
    let weight_sum = match &g.vwgt {
        // Negative weights are reported by `check_entries`.
        Some(w) => w.iter().map(|&x| x.max(0) as u128).sum::<u128>(),
        None => (end - begin) as u128,
    };
    let mine = [
        (end - begin) as u64,
        g.adjncy.len() as u64,
        u64::try_from(weight_sum).unwrap_or(u64::MAX),
        u64::from(g.vwgt.is_some()),
    ];
    let send: Vec<u64> = (0..p).flat_map(|_| mine).collect();
    let (all, _) = comm.all_to_all_v(&send, &vec![mine.len(); p])?;
    let (counts, rest) = all.as_chunks::<4>();
    debug_assert!(rest.is_empty());
    for (r, c) in counts.iter().enumerate() {
        let declared = g.vtxdist[r + 1] - g.vtxdist[r];
        if declared as u64 != c[0] {
            return Err(GridError::LengthMismatch {
                expected: c[0] as usize,
                found: declared as usize,
            });
        }
    }
    let total = |k: usize| counts.iter().map(|c| u128::from(c[k])).sum::<u128>();
    let num_vertices = g.vtxdist[p];
    let limit = u128::try_from(max).map_err(|_| GridError::TooLarge)?;
    let num_edges = total(1);
    if num_vertices as u128 > limit || num_edges > limit || total(2) > limit {
        return Err(GridError::TooLarge);
    }
    if nparts as u128 > limit {
        return Err(GridError::TooLarge);
    }
    Ok(Checked {
        begin,
        end,
        num_vertices,
        num_edges: num_edges as i64,
        weighted: total(3) > 0,
    })
}

/// Step 3 of [`validate`], the local part: `adjncy` in `[0, N)`, no self-loops, non-negative
/// weights.
fn check_entries(g: &DistGraph, checked: &Checked, name: &'static str) -> Result<()> {
    let n = checked.num_vertices;
    for (i, row) in g.xadj.windows(2).enumerate() {
        let u = checked.begin + i as i64;
        for &v in &g.adjncy[row[0] as usize..row[1] as usize] {
            if !(0..n).contains(&v) {
                return Err(GridError::IndexOutOfRange {
                    index: usize::try_from(v).unwrap_or(usize::MAX),
                    len: n as usize,
                });
            }
            if v == u {
                return Err(GridError::PartitionerFailed { name, code: -2 });
            }
        }
    }
    if g.vwgt.as_ref().is_some_and(|w| w.iter().any(|&x| x < 0)) {
        return Err(GridError::TooLarge);
    }
    Ok(())
}

/// The rank that owns global vertex `v` (`vtxdist[r] <= v < vtxdist[r + 1]`; empty ranks own
/// nothing). `v` must lie in `[0, vtxdist[nranks])`.
pub(crate) fn owner(vtxdist: &[i64], v: i64) -> usize {
    vtxdist.partition_point(|&x| x <= v) - 1
}

/// The pair (far, near) of every local edge near → far, grouped by the owner of `far`, and the
/// number of values for each rank (two per edge). `adjncy` must be in range.
fn reverse_queries(vtxdist: &[i64], begin: i64, g: &DistGraph) -> (Vec<i64>, Vec<usize>) {
    let p = vtxdist.len() - 1;
    let mut counts = vec![0_usize; p];
    for &v in &g.adjncy {
        counts[owner(vtxdist, v)] += 2;
    }
    let mut next: Vec<usize> = counts
        .iter()
        .scan(0, |acc, &c| {
            let start = *acc;
            *acc += c;
            Some(start)
        })
        .collect();
    let mut pairs = vec![0_i64; 2 * g.adjncy.len()];
    for (i, row) in g.xadj.windows(2).enumerate() {
        let u = begin + i as i64;
        for &v in &g.adjncy[row[0] as usize..row[1] as usize] {
            let at = &mut next[owner(vtxdist, v)];
            pairs[*at] = v;
            pairs[*at + 1] = u;
            *at += 2;
        }
    }
    (pairs, counts)
}

/// `PartitionerFailed { name, code: -3 }` unless, for every received pair (far, near), `far`
/// is a local vertex whose row contains `near`.
fn confirm_reverse(g: &DistGraph, begin: i64, pairs: &[i64], name: &'static str) -> Result<()> {
    let mut sorted = g.adjncy.clone();
    for row in g.xadj.windows(2) {
        sorted[row[0] as usize..row[1] as usize].sort_unstable();
    }
    let n = g.xadj.len() - 1;
    let asymmetric = GridError::PartitionerFailed { name, code: -3 };
    for pair in pairs.as_chunks::<2>().0 {
        let [far, near] = *pair;
        let Some(i) = usize::try_from(far - begin).ok().filter(|&i| i < n) else {
            return Err(asymmetric);
        };
        let row = &sorted[g.xadj[i] as usize..g.xadj[i + 1] as usize];
        if row.binary_search(&near).is_err() {
            return Err(asymmetric);
        }
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use gridcl_parallel::Serial;
    use proptest::prelude::*;

    const NAME: &str = "test";

    /// The graph of undirected `edges` on `n` vertices, both directions in every row, rows
    /// in input order.
    fn symmetric(n: usize, edges: &[(usize, usize)], vwgt: Option<Vec<i64>>) -> DistGraph {
        let mut rows = vec![Vec::new(); n];
        for &(u, v) in edges {
            rows[u].push(v as i64);
            rows[v].push(u as i64);
        }
        from_rows(&rows, vwgt)
    }

    fn from_rows(rows: &[Vec<i64>], vwgt: Option<Vec<i64>>) -> DistGraph {
        let mut xadj = vec![0_i64];
        for r in rows {
            xadj.push(xadj.last().unwrap() + r.len() as i64);
        }
        DistGraph {
            vtxdist: vec![0, rows.len() as i64],
            xadj,
            adjncy: rows.concat(),
            vwgt,
        }
    }

    fn check(g: &DistGraph, nparts: usize, max: i64) -> Result<Checked> {
        validate(&Serial, g, nparts, NAME, max)
    }

    /// The rows of a graph on one rank.
    fn rows_of(g: &DistGraph) -> Vec<Vec<i64>> {
        g.xadj
            .windows(2)
            .map(|w| g.adjncy[w[0] as usize..w[1] as usize].to_vec())
            .collect()
    }

    fn code(e: GridError) -> Option<i32> {
        match e {
            GridError::PartitionerFailed { name: NAME, code } => Some(code),
            _ => None,
        }
    }

    #[test]
    fn signed_max_of_library_widths() {
        assert_eq!(signed_max(32), i64::from(i32::MAX));
        assert_eq!(signed_max(64), i64::MAX);
        assert_eq!(signed_max(8), 127);
        assert_eq!(signed_max(1), 0);
    }

    #[test]
    fn valid_graphs_pass_with_their_totals() {
        let path = symmetric(4, &[(0, 1), (1, 2), (2, 3)], Some(vec![3, 0, 5, 1]));
        assert_eq!(
            check(&path, 2, i64::MAX).unwrap(),
            Checked {
                begin: 0,
                end: 4,
                num_vertices: 4,
                num_edges: 6,
                weighted: true,
            }
        );
        let empty = DistGraph {
            vtxdist: vec![0, 0],
            xadj: vec![0],
            adjncy: vec![],
            vwgt: None,
        };
        let c = check(&empty, 1, 1).unwrap();
        assert_eq!((c.num_vertices, c.num_edges, c.weighted), (0, 0, false));
        // Isolated vertices and duplicate edges are allowed.
        let multi = symmetric(5, &[(0, 3), (3, 0)], None);
        assert_eq!(check(&multi, 3, 5).unwrap().num_edges, 4);
        assert!(matches!(
            check(&multi, 3, 4).unwrap_err(),
            GridError::TooLarge
        ));
    }

    #[test]
    fn shape_errors_are_blocks() {
        let g = symmetric(3, &[(0, 1)], None);
        let mismatch = |e: GridError| matches!(e, GridError::LengthMismatch { .. });
        assert!(mismatch(check(&g, 0, i64::MAX).unwrap_err()));
        assert!(matches!(
            check(&g, 1 << 31, i64::MAX).unwrap_err(),
            GridError::TooLarge
        ));
        let short = DistGraph {
            vwgt: Some(vec![1]),
            ..g.clone()
        };
        assert!(mismatch(check(&short, 2, i64::MAX).unwrap_err()));
        let unsorted = DistGraph {
            xadj: vec![0, 2, 1, 2],
            ..g
        };
        assert!(mismatch(check(&unsorted, 2, i64::MAX).unwrap_err()));
    }

    #[test]
    fn each_invalid_graph_has_its_error() {
        let g = symmetric(3, &[(0, 1), (1, 2)], Some(vec![1, 1, 1]));
        let with = |rows: Vec<Vec<i64>>| from_rows(&rows, g.vwgt.clone());
        let mut rows = rows_of(&g);
        rows[0][0] = 3;
        assert!(matches!(
            check(&with(rows), 2, i64::MAX).unwrap_err(),
            GridError::IndexOutOfRange { index: 3, len: 3 }
        ));
        let mut rows = rows_of(&g);
        rows[2][0] = -1;
        assert!(matches!(
            check(&with(rows), 2, i64::MAX).unwrap_err(),
            GridError::IndexOutOfRange {
                index: usize::MAX,
                len: 3
            }
        ));
        let mut rows = rows_of(&g);
        rows[1].push(1);
        assert_eq!(code(check(&with(rows), 2, i64::MAX).unwrap_err()), Some(-2));
        let mut rows = rows_of(&g);
        rows[0].push(2);
        assert_eq!(code(check(&with(rows), 2, i64::MAX).unwrap_err()), Some(-3));
        let negative = DistGraph {
            vwgt: Some(vec![1, -1, 1]),
            ..g.clone()
        };
        assert!(matches!(
            check(&negative, 2, i64::MAX).unwrap_err(),
            GridError::TooLarge
        ));
    }

    #[test]
    fn totals_must_fit_the_index_type() {
        // N = 4, 6 edge entries, weight sum 9.
        let g = symmetric(4, &[(0, 1), (1, 2), (2, 3)], Some(vec![3, 0, 5, 1]));
        assert!(check(&g, 2, 9).is_ok());
        assert!(matches!(check(&g, 2, 8).unwrap_err(), GridError::TooLarge));
        let unit = DistGraph { vwgt: None, ..g };
        assert!(check(&unit, 2, 6).is_ok());
        assert!(matches!(
            check(&unit, 2, 5).unwrap_err(),
            GridError::TooLarge
        ));
        // N alone: no edges.
        let isolated = symmetric(4, &[], None);
        assert!(check(&isolated, 2, 4).is_ok());
        assert!(matches!(
            check(&isolated, 2, 3).unwrap_err(),
            GridError::TooLarge
        ));
        // nparts too.
        assert!(matches!(
            check(&isolated, 5, 4).unwrap_err(),
            GridError::TooLarge
        ));
        // Weights whose sum overflows i64.
        let huge = symmetric(2, &[], Some(vec![i64::MAX, i64::MAX]));
        assert!(matches!(
            check(&huge, 1, i64::MAX).unwrap_err(),
            GridError::TooLarge
        ));
    }

    #[test]
    fn local_weights_fill_unit_weights() {
        let g = symmetric(3, &[], None);
        assert_eq!(local_weights(&g, false), None);
        assert_eq!(local_weights(&g, true).unwrap().as_ref(), [1, 1, 1]);
        let w = DistGraph {
            vwgt: Some(vec![4, 5, 6]),
            ..g
        };
        assert_eq!(local_weights(&w, false).unwrap().as_ref(), [4, 5, 6]);
    }

    /// The rank-local graphs of `g` (on one rank) split by `vtxdist`.
    fn split(g: &DistGraph, vtxdist: &[i64]) -> Vec<DistGraph> {
        let rows = rows_of(g);
        vtxdist
            .windows(2)
            .map(|w| {
                let mut part = from_rows(&rows[w[0] as usize..w[1] as usize], None);
                part.vtxdist = vtxdist.to_vec();
                part.vwgt = g
                    .vwgt
                    .as_ref()
                    .map(|v| v[w[0] as usize..w[1] as usize].to_vec());
                part
            })
            .collect()
    }

    /// The symmetry check over simulated ranks: queries routed by hand.
    fn symmetric_on_ranks(parts: &[DistGraph]) -> Result<()> {
        let p = parts.len();
        let mut inbox = vec![Vec::new(); p];
        for (r, g) in parts.iter().enumerate() {
            let (pairs, counts) = reverse_queries(&g.vtxdist, g.vtxdist[r], g);
            let mut at = 0;
            for (dest, c) in counts.into_iter().enumerate() {
                inbox[dest].extend_from_slice(&pairs[at..at + c]);
                at += c;
            }
        }
        for (r, g) in parts.iter().enumerate() {
            confirm_reverse(g, g.vtxdist[r], &inbox[r], NAME)?;
        }
        Ok(())
    }

    #[test]
    fn owners_skip_empty_ranks() {
        let vtxdist = [0, 2, 2, 2, 5, 5];
        let owners: Vec<usize> = (0..5).map(|v| owner(&vtxdist, v)).collect();
        assert_eq!(owners, [0, 0, 3, 3, 3]);
    }

    #[test]
    fn symmetry_is_checked_across_ranks() {
        let g = symmetric(6, &[(0, 5), (1, 2), (2, 3), (4, 5), (0, 3)], None);
        for vtxdist in [
            vec![0, 6],
            vec![0, 3, 6],
            vec![0, 0, 4, 4, 6],
            vec![0, 1, 2, 6, 6],
        ] {
            assert!(symmetric_on_ranks(&split(&g, &vtxdist)).is_ok());
        }
        let mut rows = rows_of(&g);
        rows[1].push(4);
        let bad = from_rows(&rows, None);
        for vtxdist in [vec![0, 6], vec![0, 3, 6], vec![0, 0, 4, 4, 6]] {
            let e = symmetric_on_ranks(&split(&bad, &vtxdist)).unwrap_err();
            assert_eq!(code(e), Some(-3));
        }
    }

    /// splitmix64: a small deterministic generator for the property tests.
    pub(crate) struct Rng(pub u64);

    impl Rng {
        pub(crate) fn below(&mut self, n: u64) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            (z ^ (z >> 31)) % n
        }
    }

    /// A random symmetric graph without self-loops on `n` vertices with about `degree · n / 2`
    /// undirected edges, weighted or not.
    pub(crate) fn random_graph(n: usize, degree: usize, weighted: bool, seed: u64) -> DistGraph {
        let mut rng = Rng(seed);
        let mut edges = Vec::new();
        if n >= 2 {
            for _ in 0..degree * n / 2 {
                let u = rng.below(n as u64) as usize;
                let v = rng.below(n as u64 - 1) as usize;
                edges.push((u, if v >= u { v + 1 } else { v }));
            }
        }
        let vwgt = weighted.then(|| (0..n).map(|_| rng.below(217) as i64).collect());
        symmetric(n, &edges, vwgt)
    }

    /// Mostly small sizes, now and then up to 10^5.
    pub(crate) fn size() -> impl Strategy<Value = usize> {
        prop_oneof![8 => 0_usize..64, 3 => 0_usize..5_000, 1 => 0_usize..=100_000]
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn random_symmetric_graphs_pass_and_perturbations_fail(
            n in size(),
            degree in 0_usize..8,
            weighted in any::<bool>(),
            seed in any::<u64>(),
            kind in 0_u8..5,
        ) {
            let g = random_graph(n, degree, weighted, seed);
            let c = check(&g, 3, i64::MAX).unwrap();
            prop_assert_eq!(c.num_vertices, n as i64);
            prop_assert_eq!(c.num_edges, g.adjncy.len() as i64);
            prop_assert_eq!(c.weighted, weighted);
            // Totals just above the limit.
            let w = g.vwgt.as_ref().map_or(n as i64, |w| w.iter().sum());
            let largest = (n as i64).max(c.num_edges).max(w).max(3);
            prop_assert!(check(&g, 3, largest).is_ok());
            prop_assert!(matches!(check(&g, 3, largest - 1), Err(GridError::TooLarge)));
            // One perturbation of each kind.
            prop_assume!(n >= 2);
            let mut rng = Rng(!seed);
            let mut rows = rows_of(&g);
            let u = rng.below(n as u64) as usize;
            let mut vwgt = g.vwgt.clone();
            match kind {
                0 => rows[u].push(n as i64 + rng.below(3) as i64),
                1 => rows[u].push(-1 - rng.below(3) as i64),
                2 => rows[u].push(u as i64),
                3 => {
                    // A one-way edge to a vertex not adjacent to u.
                    let free = (0..n as i64).find(|&v| v != u as i64 && !rows[u].contains(&v));
                    prop_assume!(free.is_some());
                    rows[u].push(free.unwrap());
                }
                _ => vwgt = Some((0..n as i64).map(|v| if v == u as i64 { -1 } else { 1 }).collect()),
            }
            let e = check(&from_rows(&rows, vwgt), 3, i64::MAX).unwrap_err();
            match kind {
                0 => {
                    let out = matches!(e, GridError::IndexOutOfRange { len, .. } if len == n);
                    prop_assert!(out);
                }
                1 => {
                    let negative =
                        matches!(e, GridError::IndexOutOfRange { index: usize::MAX, len } if len == n);
                    prop_assert!(negative);
                }
                2 => prop_assert_eq!(code(e), Some(-2)),
                3 => prop_assert_eq!(code(e), Some(-3)),
                _ => prop_assert!(matches!(e, GridError::TooLarge)),
            }
        }

        #[test]
        fn symmetry_holds_on_any_split(
            n in 0_usize..300,
            degree in 0_usize..6,
            seed in any::<u64>(),
            cuts in proptest::collection::vec(0_usize..=300, 0..5),
        ) {
            let g = random_graph(n, degree, false, seed);
            let mut cuts: Vec<i64> = cuts.into_iter().map(|c| c.min(n) as i64).collect();
            cuts.sort_unstable();
            let vtxdist: Vec<i64> = [0].into_iter().chain(cuts).chain([n as i64]).collect();
            prop_assert!(symmetric_on_ranks(&split(&g, &vtxdist)).is_ok());
            prop_assume!(n >= 2);
            let mut rows = rows_of(&g);
            let u = (seed % n as u64) as usize;
            let free = (0..n as i64).find(|&v| v != u as i64 && !rows[u].contains(&v));
            prop_assume!(free.is_some());
            rows[u].push(free.unwrap());
            let e = symmetric_on_ranks(&split(&from_rows(&rows, None), &vtxdist)).unwrap_err();
            prop_assert_eq!(code(e), Some(-3));
        }
    }
}
