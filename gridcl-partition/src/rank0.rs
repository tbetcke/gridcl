//! The rank-0 path of `Metis` and `Scotch` (ADR-0029, item 7): the graph gathered on rank 0 by
//! `all_to_all_v` with counts to rank 0 only, the destinations scattered back the same way, and
//! the checked narrowing to a library's index type.
#![allow(dead_code)] // Until `Metis` and `Scotch` (M7 T3, T4) use it.

use gridcl_parallel::{CommDyn, CommExt, GridError, Result};

use crate::graph::{Checked, DistGraph, agree, local_weights};

/// The whole graph in CSR form, on rank 0.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Csr {
    /// Row offsets; N + 1 entries.
    pub xadj: Vec<i64>,
    /// Global neighbour ids.
    pub adjncy: Vec<i64>,
    /// Vertex weights (unit weights on ranks with `vwgt: None`); `None` if no rank has weights.
    pub vwgt: Option<Vec<i64>>,
}

/// Counts that send `len` values to rank 0 only.
fn to_root(comm: &dyn CommDyn, len: usize) -> Vec<usize> {
    let mut counts = vec![0; comm.size()];
    counts[0] = len;
    counts
}

/// `MessageTooLarge` if `count` elements exceed an MPI count (`i32::MAX`).
fn fits_message(count: i64) -> Result<()> {
    if i32::try_from(count).is_err() {
        return Err(GridError::MessageTooLarge {
            count: usize::try_from(count).unwrap_or(usize::MAX),
        });
    }
    Ok(())
}

/// The graph `g`, which passed [`validate`](crate::graph::validate) as `checked`, gathered on
/// rank 0: `Some` there, `None` on the other ranks. Collective; the result is agreed.
///
/// # Errors
///
/// `MessageTooLarge` on every rank if N or the edge count exceeds `i32::MAX` (rank 0 would
/// receive more than one MPI message holds; every rank knows the totals, so no rank enters the
/// exchange); the errors of `CommExt::all_to_all_v`. Ranks without a local error return
/// `RemoteFailure`.
pub(crate) fn gather_to_root(
    comm: &dyn CommDyn,
    g: &DistGraph,
    checked: &Checked,
) -> Result<Option<Csr>> {
    fits_message(checked.num_vertices)?;
    fits_message(checked.num_edges)?;
    let degrees: Vec<i64> = g.xadj.windows(2).map(|w| w[1] - w[0]).collect();
    // Every exchange runs on every rank; the errors are combined afterwards.
    let degrees = comm.all_to_all_v(&degrees, &to_root(comm, degrees.len()));
    let adjncy = comm.all_to_all_v(&g.adjncy, &to_root(comm, g.adjncy.len()));
    let vwgt =
        local_weights(g, checked.weighted).map(|w| comm.all_to_all_v(&w, &to_root(comm, w.len())));
    let local = (|| {
        let (degrees, _) = degrees?;
        let (adjncy, _) = adjncy?;
        let vwgt = vwgt.transpose()?.map(|(w, _)| w);
        if comm.rank() != 0 {
            return Ok(None);
        }
        let mut xadj = Vec::with_capacity(degrees.len() + 1);
        xadj.push(0_i64);
        for d in degrees {
            xadj.push(xadj[xadj.len() - 1] + d);
        }
        Ok(Some(Csr { xadj, adjncy, vwgt }))
    })();
    agree(comm, local)
}

/// The destinations of rank 0 (`parts`: one per global vertex, or the error of the library
/// call; `None` on the other ranks) sent back to the owning ranks by `vtxdist`; returns this
/// rank's destinations. Collective; the result is agreed.
///
/// # Errors
///
/// The error in `parts` on rank 0; `LengthMismatch` on rank 0 if `parts` is `None` there or
/// does not have `vtxdist[nranks]` entries; the errors of `CommExt::all_to_all_v`. Ranks without
/// a local error return `RemoteFailure`.
pub(crate) fn scatter_parts(
    comm: &dyn CommDyn,
    vtxdist: &[i64],
    parts: Option<Result<Vec<i32>>>,
) -> Result<Vec<i32>> {
    let p = comm.size();
    let total = vtxdist[p];
    let root = if comm.rank() == 0 {
        match parts {
            Some(Ok(parts)) if parts.len() as i64 == total => Ok(parts),
            Some(Ok(parts)) => Err(GridError::LengthMismatch {
                expected: total as usize,
                found: parts.len(),
            }),
            Some(Err(e)) => Err(e),
            None => Err(GridError::LengthMismatch {
                expected: total as usize,
                found: 0,
            }),
        }
    } else {
        Ok(Vec::new())
    };
    // A failing rank 0 sends nothing; the other ranks then receive nothing.
    let (send, counts) = match &root {
        Ok(parts) if comm.rank() == 0 => (
            parts.as_slice(),
            vtxdist.windows(2).map(|w| (w[1] - w[0]) as usize).collect(),
        ),
        _ => (&[][..], vec![0; p]),
    };
    let received = comm.all_to_all_v(send, &counts);
    agree(comm, root.and(received).map(|(mine, _)| mine))
}

/// `values` narrowed to the library's index type `T`; `TooLarge` if a value does not fit.
/// Not collective.
pub(crate) fn narrow<T: TryFrom<i64>>(values: &[i64]) -> Result<Vec<T>> {
    values
        .iter()
        .map(|&v| T::try_from(v).map_err(|_| GridError::TooLarge))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::tests::{random_graph, size};
    use crate::graph::validate;
    use gridcl_parallel::Serial;
    use proptest::prelude::*;

    fn round_trip(g: &DistGraph) -> Vec<i32> {
        let checked = validate(&Serial, g, 2, "test", i64::from(i32::MAX)).unwrap();
        let csr = gather_to_root(&Serial, g, &checked).unwrap().unwrap();
        assert_eq!(csr.xadj, g.xadj);
        assert_eq!(csr.adjncy, g.adjncy);
        assert_eq!(csr.vwgt, g.vwgt);
        // Destinations: the vertex id mod 2.
        let parts = (0..csr.xadj.len() as i32 - 1).map(|v| v % 2).collect();
        scatter_parts(&Serial, &g.vtxdist, Some(Ok(parts))).unwrap()
    }

    #[test]
    fn gather_and_scatter_round_trip_on_one_rank() {
        let g = random_graph(7, 3, true, 1);
        assert_eq!(round_trip(&g), [0, 1, 0, 1, 0, 1, 0]);
        let unweighted = random_graph(5, 2, false, 2);
        assert_eq!(round_trip(&unweighted), [0, 1, 0, 1, 0]);
    }

    #[test]
    fn empty_graph_round_trips() {
        let g = DistGraph {
            vtxdist: vec![0, 0],
            xadj: vec![0],
            adjncy: vec![],
            vwgt: None,
        };
        assert!(round_trip(&g).is_empty());
    }

    #[test]
    fn scatter_reports_rank_zero_errors() {
        let vtxdist = [0, 3];
        assert!(matches!(
            scatter_parts(&Serial, &vtxdist, Some(Ok(vec![0, 1]))).unwrap_err(),
            GridError::LengthMismatch {
                expected: 3,
                found: 2
            }
        ));
        assert!(matches!(
            scatter_parts(&Serial, &vtxdist, None).unwrap_err(),
            GridError::LengthMismatch { .. }
        ));
        let failed = Err(GridError::PartitionerFailed {
            name: "test",
            code: 7,
        });
        assert!(matches!(
            scatter_parts(&Serial, &vtxdist, Some(failed)).unwrap_err(),
            GridError::PartitionerFailed { code: 7, .. }
        ));
    }

    #[test]
    fn gather_refuses_more_than_one_message_holds() {
        let g = random_graph(2, 1, false, 3);
        let checked = Checked {
            num_edges: 1 << 31,
            ..validate(&Serial, &g, 1, "test", i64::MAX).unwrap()
        };
        assert!(matches!(
            gather_to_root(&Serial, &g, &checked).unwrap_err(),
            GridError::MessageTooLarge { count } if count == 1 << 31
        ));
    }

    #[test]
    fn narrowing_checks_every_value() {
        assert_eq!(
            narrow::<i32>(&[0, -5, i64::from(i32::MAX)]).unwrap(),
            [0, -5, i32::MAX]
        );
        assert!(matches!(
            narrow::<i32>(&[1, i64::from(i32::MAX) + 1]).unwrap_err(),
            GridError::TooLarge
        ));
        assert!(matches!(
            narrow::<i32>(&[i64::from(i32::MIN) - 1]).unwrap_err(),
            GridError::TooLarge
        ));
        assert_eq!(
            narrow::<i64>(&[i64::MIN, i64::MAX]).unwrap(),
            [i64::MIN, i64::MAX]
        );
        assert!(narrow::<i32>(&[]).unwrap().is_empty());
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(256))]

        #[test]
        fn narrowing_round_trips_iff_in_range(
            values in proptest::collection::vec(
                prop_oneof![
                    4 => i64::from(i32::MIN)..=i64::from(i32::MAX),
                    1 => any::<i64>(),
                ],
                0..64,
            ),
        ) {
            let fits = values.iter().all(|&v| i32::try_from(v).is_ok());
            match narrow::<i32>(&values) {
                Ok(n) => {
                    prop_assert!(fits);
                    prop_assert_eq!(n.into_iter().map(i64::from).collect::<Vec<_>>(), values);
                }
                Err(e) => {
                    prop_assert!(!fits);
                    prop_assert!(matches!(e, GridError::TooLarge));
                }
            }
        }

        #[test]
        fn random_graphs_round_trip(
            n in size(),
            degree in 0_usize..8,
            weighted in any::<bool>(),
            seed in any::<u64>(),
        ) {
            let g = random_graph(n, degree, weighted, seed);
            let parts = round_trip(&g);
            prop_assert_eq!(parts.len(), n);
            prop_assert!(parts.iter().enumerate().all(|(v, &d)| d == v as i32 % 2));
        }
    }
}
