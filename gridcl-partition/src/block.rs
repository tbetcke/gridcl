//! The built-in block partitioner (ADR-0026, item 2).

use gridcl_parallel::{CommDyn, Result};

use crate::graph::{agree, check_shape};
use crate::{DistGraph, Partitioner};

/// Built-in block partitioner: global graph vertex i goes to rank ⌊i · nparts / N⌋, with
/// N = `vtxdist[nranks]` (ADR-0026).
///
/// The parts differ in size by at most one. The edges are not read.
///
/// # Example
///
/// ```
/// use gridcl_parallel::Serial;
/// use gridcl_partition::{Block, DistGraph, Partitioner};
///
/// // A path of five vertices.
/// let g = DistGraph {
///     vtxdist: vec![0, 5],
///     xadj: vec![0, 1, 3, 5, 7, 8],
///     adjncy: vec![1, 0, 2, 1, 3, 2, 4, 3],
///     vwgt: None,
/// };
/// assert_eq!(Block.partition(&Serial, &g, 2)?, [0, 0, 0, 1, 1]);
/// # Ok::<(), gridcl_parallel::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Block;

impl Partitioner for Block {
    fn name(&self) -> &str {
        "block"
    }
    /// # Errors
    ///
    /// Agreed over the ranks (ADR-0026, item 15): `LengthMismatch` if `nparts` is 0
    /// (expected 1), if `vtxdist` does not have `comm.size() + 1` non-decreasing entries from
    /// 0, if `xadj` does not have one more entry than this rank's vertices, starting at 0, not
    /// decreasing and ending at `adjncy.len()`, or if `vwgt` does not have one entry per
    /// vertex; `TooLarge` if `nparts` exceeds `i32::MAX`. Other ranks return `RemoteFailure`.
    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        let (begin, end) = agree(comm, check_shape(comm, g, nparts))?;
        let total = g.vtxdist[comm.size()] as u128;
        // Exact in u128: i < 2^63 and nparts < 2^31.
        Ok((begin..end)
            .map(|i| (i as u128 * nparts as u128 / total) as i32)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_parallel::{GridError, Serial};

    fn graph(n: i64) -> DistGraph {
        DistGraph {
            vtxdist: vec![0, n],
            xadj: vec![0; n as usize + 1],
            adjncy: vec![],
            vwgt: None,
        }
    }

    #[test]
    fn splits_into_contiguous_balanced_parts() {
        assert_eq!(Block.name(), "block");
        assert_eq!(
            Block.partition(&Serial, &graph(7), 3).unwrap(),
            [0, 0, 0, 1, 1, 2, 2]
        );
        assert_eq!(Block.partition(&Serial, &graph(2), 4).unwrap(), [0, 2]);
        assert!(Block.partition(&Serial, &graph(0), 3).unwrap().is_empty());
        for n in 0..40 {
            for parts in 1..9 {
                let p = Block.partition(&Serial, &graph(n), parts).unwrap();
                let mut sizes = vec![0; parts];
                for &d in &p {
                    sizes[d as usize] += 1;
                }
                assert!(p.windows(2).all(|w| w[0] <= w[1]));
                let (lo, hi) = (sizes.iter().min().unwrap(), sizes.iter().max().unwrap());
                assert!(hi - lo <= 1, "n = {n}, parts = {parts}: {sizes:?}");
            }
        }
    }

    #[test]
    fn inconsistent_graphs_are_rejected() {
        let bad = |g: DistGraph, parts: usize| Block.partition(&Serial, &g, parts).unwrap_err();
        let mismatch = |e: GridError| matches!(e, GridError::LengthMismatch { .. });
        assert!(mismatch(bad(graph(3), 0)));
        assert!(matches!(bad(graph(3), 1 << 31), GridError::TooLarge));
        assert!(mismatch(bad(
            DistGraph {
                vtxdist: vec![0, 2, 3],
                ..graph(3)
            },
            2
        )));
        assert!(mismatch(bad(
            DistGraph {
                vtxdist: vec![1, 3],
                ..graph(3)
            },
            2
        )));
        assert!(mismatch(bad(
            DistGraph {
                xadj: vec![0, 0, 0],
                ..graph(3)
            },
            2
        )));
        assert!(mismatch(bad(
            DistGraph {
                xadj: vec![0, 2, 1, 2],
                adjncy: vec![0, 1],
                ..graph(3)
            },
            2
        )));
        assert!(mismatch(bad(
            DistGraph {
                xadj: vec![0, 1, 1, 1],
                ..graph(3)
            },
            2
        )));
        assert!(mismatch(bad(
            DistGraph {
                vwgt: Some(vec![1, 1]),
                ..graph(3)
            },
            2
        )));
        let weighted = DistGraph {
            vwgt: Some(vec![1, 2, 3]),
            ..graph(3)
        };
        assert_eq!(Block.partition(&Serial, &weighted, 2).unwrap(), [0, 0, 1]);
    }
}
