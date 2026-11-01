//! Partition quality for the M7 acceptance tests (docs/plans/M7.md, T0; ADR-0029, item 11):
//! the edge cut and the weighted imbalance of a partition of a distributed graph, the slab
//! bound on the cut, the two acceptance meshes, a hash of a partition, and a serial facet dual
//! graph of a host mesh.

use std::collections::HashMap;

use gridcl::{CommDyn, CommExt, DistGraph, Result};
use gridcl_dims::{Cell3, CellType};
use gridcl_refcell::{sub_entities, tdim, vertices};

use crate::HostMesh;

/// The kind of an acceptance box.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxKind {
    /// `cube_box(Cell3::Tetrahedron, n, p)`: two triangles per grid square of an x-plane.
    Tetrahedra,
    /// `hex_prism_pyramid_box(n, p)`: one quadrilateral per grid square of an x-plane.
    HexPrismPyramid,
}

/// A box of n³ hexahedral subcells for the partitioner tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PartitionMesh {
    /// The name in logs.
    pub name: &'static str,
    /// The cell types.
    pub kind: BoxKind,
    /// Subcells per direction.
    pub n: usize,
}

impl PartitionMesh {
    /// The host mesh at degree `p`.
    pub fn host(&self, p: u32) -> HostMesh {
        match self.kind {
            BoxKind::Tetrahedra => crate::cube_box(Cell3::Tetrahedron, self.n, p),
            BoxKind::HexPrismPyramid => crate::hex_prism_pyramid_box(self.n, p),
        }
    }
}

/// The two meshes of the M7 acceptance: the tetrahedral box with n = 56 (1 053 696 cells) and
/// the hexahedron/prism/pyramid box with n = 41 (208 444 cells), both at p = 1.
///
/// # Example
///
/// ```
/// let [tets, mixed] = gridcl_testing::partition::acceptance_meshes();
/// assert_eq!(6 * tets.n.pow(3), 1_053_696);
/// assert_eq!(mixed.n, 41);
/// ```
pub fn acceptance_meshes() -> [PartitionMesh; 2] {
    [
        PartitionMesh {
            name: "tetrahedra n=56",
            kind: BoxKind::Tetrahedra,
            n: 56,
        },
        PartitionMesh {
            name: "hex/prism/pyramid n=41",
            kind: BoxKind::HexPrismPyramid,
            n: 41,
        },
    ]
}

/// The edge cut of k = `nparts` slabs separated by k − 1 grid planes normal to x: 2(k − 1)n²
/// for tetrahedra and (k − 1)n² for the hexahedron/prism/pyramid box (ADR-0029, item 11).
///
/// # Example
///
/// ```
/// use gridcl_testing::partition::{BoxKind, PartitionMesh, slab_cut_bound};
///
/// let m = PartitionMesh { name: "", kind: BoxKind::Tetrahedra, n: 4 };
/// assert_eq!(slab_cut_bound(&m, 3), 64);
/// assert_eq!(slab_cut_bound(&m, 1), 0);
/// ```
pub fn slab_cut_bound(mesh: &PartitionMesh, nparts: usize) -> i64 {
    let per_square = match mesh.kind {
        BoxKind::Tetrahedra => 2,
        BoxKind::HexPrismPyramid => 1,
    };
    let n = mesh.n as i64;
    per_square * (nparts as i64 - 1).max(0) * n * n
}

/// The number of undirected edges of the symmetric graph `g` whose ends lie in different parts,
/// each counted once; `part` holds the part of every local vertex. The parts of remote
/// neighbours come from their owners. Collective.
///
/// # Errors
///
/// Errors of `CommExt::all_to_all_v`.
pub fn edge_cut(comm: &dyn CommDyn, g: &DistGraph, part: &[i32]) -> Result<i64> {
    let size = comm.size();
    let first = g.vtxdist[comm.rank()];
    let owner = |v: i64| g.vtxdist.partition_point(|&x| x <= v) - 1;
    // Edges (u, v) with u < v, by the owner of v.
    let mut by_owner: Vec<Vec<(usize, i64)>> = vec![Vec::new(); size];
    for u in 0..part.len() {
        let gu = first + u as i64;
        for &v in &g.adjncy[g.xadj[u] as usize..g.xadj[u + 1] as usize] {
            if v > gu {
                by_owner[owner(v)].push((u, v));
            }
        }
    }
    let ask: Vec<i64> = by_owner.iter().flatten().map(|&(_, v)| v).collect();
    let counts: Vec<usize> = by_owner.iter().map(Vec::len).collect();
    let (asked, asked_counts) = comm.all_to_all_v(&ask, &counts)?;
    let answers: Vec<i32> = asked.iter().map(|&v| part[(v - first) as usize]).collect();
    let (theirs, _) = comm.all_to_all_v(&answers, &asked_counts)?;
    let cut = by_owner
        .iter()
        .flatten()
        .zip(&theirs)
        .filter(|&(&(u, _), &pv)| part[u] != pv)
        .count();
    Ok(comm.sum(cut as i64))
}

/// max_k W_k · nparts / W − 1, with W_k the summed vertex weight of part k (weight 1 without
/// `vwgt`) and W the total (ADR-0029, item 11); 0 for an empty graph. Collective.
///
/// # Errors
///
/// Errors of `CommExt::all_to_all_v`.
///
/// # Panics
///
/// If a part is outside `0..nparts`.
pub fn weighted_imbalance(
    comm: &dyn CommDyn,
    g: &DistGraph,
    part: &[i32],
    nparts: usize,
) -> Result<f64> {
    let mut local = vec![0_i64; nparts];
    for (i, &k) in part.iter().enumerate() {
        let k = usize::try_from(k).expect("a non-negative part");
        assert!(k < nparts, "part {k} of {nparts}");
        local[k] += g.vwgt.as_ref().map_or(1, |w| w[i]);
    }
    // Every rank's sums on rank 0, then rank 0's maximum and total to every rank.
    let mut counts = vec![0; comm.size()];
    counts[0] = nparts;
    let (all, _) = comm.all_to_all_v(&local, &counts)?;
    let mut sums = vec![0_i64; nparts];
    for chunk in all.chunks_exact(nparts.max(1)) {
        for (s, x) in sums.iter_mut().zip(chunk) {
            *s += x;
        }
    }
    let max = comm.max(sums.iter().copied().max().unwrap_or(0));
    let total = comm.sum(local.iter().sum::<i64>());
    if total == 0 {
        return Ok(0.0);
    }
    Ok(max as f64 * nparts as f64 / total as f64 - 1.0)
}

/// A hash of the partition over all ranks, equal on every rank: FNV-1a of each rank's parts,
/// combined in rank order. Equal partitions on equal rank counts give equal hashes. Collective.
///
/// # Example
///
/// ```
/// use gridcl::Serial;
/// use gridcl_testing::partition::parts_hash;
///
/// assert_eq!(parts_hash(&Serial, &[0, 1, 1]), parts_hash(&Serial, &[0, 1, 1]));
/// assert_ne!(parts_hash(&Serial, &[0, 1, 1]), parts_hash(&Serial, &[1, 0, 1]));
/// ```
pub fn parts_hash(comm: &dyn CommDyn, part: &[i32]) -> u64 {
    let local = fnv1a(part.iter().flat_map(|k| k.to_le_bytes()));
    fnv1a(
        comm.all_gather(local)
            .into_iter()
            .flat_map(u64::to_le_bytes),
    )
}

/// FNV-1a of `bytes`.
fn fnv1a(bytes: impl IntoIterator<Item = u8>) -> u64 {
    bytes.into_iter().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3)
    })
}

/// The facet dual graph of `h` on one rank: vertices are the cells in block order, and two
/// cells are joined when they share a facet (equal sorted vertex nodes); rows are sorted;
/// `vwgt` is the node count of every cell (the cost model of ADR-0029, item 2).
///
/// # Example
///
/// ```
/// let h = gridcl_testing::interval_box(3, 2);
/// let g = gridcl_testing::partition::host_dual_graph(&h);
/// assert_eq!(g.vtxdist, [0, 3]);
/// assert_eq!(g.adjncy, [1, 0, 2, 1]);
/// assert_eq!(g.vwgt, Some(vec![3, 3, 3]));
/// ```
pub fn host_dual_graph(h: &HostMesh) -> DistGraph {
    let mut by_facet: HashMap<Vec<u32>, Vec<i64>> = HashMap::new();
    let mut vwgt = Vec::new();
    for block in &h.blocks {
        let k = block.nodes_per_cell();
        let facets = sub_entities(block.cell, tdim(block.cell) - 1);
        for row in block.cells.chunks_exact(k) {
            let cell = vwgt.len() as i64;
            vwgt.push(k as i64);
            for facet in facets {
                let mut key: Vec<u32> = facet.iter().map(|&v| row[v]).collect();
                key.sort_unstable();
                by_facet.entry(key).or_default().push(cell);
            }
        }
    }
    let mut rows: Vec<Vec<i64>> = vec![Vec::new(); vwgt.len()];
    for cells in by_facet.values() {
        for &a in cells {
            rows[a as usize].extend(cells.iter().filter(|&&b| b != a));
        }
    }
    let mut xadj = vec![0_i64];
    let mut adjncy = Vec::new();
    for mut row in rows {
        row.sort_unstable();
        row.dedup();
        adjncy.extend(row);
        xadj.push(adjncy.len() as i64);
    }
    DistGraph {
        vtxdist: vec![0, vwgt.len() as i64],
        xadj,
        adjncy,
        vwgt: Some(vwgt),
    }
}

/// The slab partition of `h`, a box of `n`³ subcells, into `nparts` parts: the cell whose
/// vertex centroid lies in subcell column i goes to part ⌊i · nparts / n⌋. Cells in block order.
pub fn slab_parts(h: &HostMesh, n: usize, nparts: usize) -> Vec<i32> {
    let mut parts = Vec::with_capacity(h.num_cells());
    for block in &h.blocks {
        let k = block.nodes_per_cell();
        let nv = vertices(block.cell).len();
        for row in block.cells.chunks_exact(k) {
            let cx = row[..nv]
                .iter()
                .map(|&v| h.nodes[v as usize * h.gdim])
                .sum::<f64>()
                / nv as f64;
            let i = ((cx * n as f64) as usize).min(n - 1);
            parts.push((i * nparts / n) as i32);
        }
    }
    parts
}

/// The number of cells of type `cell` in `h`.
pub fn num_cells_of(h: &HostMesh, cell: CellType) -> usize {
    h.blocks
        .iter()
        .filter(|b| b.cell == cell)
        .map(|b| b.num_cells())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ThreadComm;
    use gridcl::Serial;

    /// A path 0 – 1 – … – (n − 1) split over `counts.len()` ranks; rank r's share.
    fn path(rank: usize, counts: &[i64], weights: bool) -> DistGraph {
        let mut vtxdist = vec![0_i64];
        for c in counts {
            vtxdist.push(vtxdist.last().unwrap() + c);
        }
        let n = vtxdist[counts.len()];
        let (begin, end) = (vtxdist[rank], vtxdist[rank + 1]);
        let mut xadj = vec![0_i64];
        let mut adjncy = Vec::new();
        for v in begin..end {
            if v > 0 {
                adjncy.push(v - 1);
            }
            if v + 1 < n {
                adjncy.push(v + 1);
            }
            xadj.push(adjncy.len() as i64);
        }
        DistGraph {
            vtxdist,
            xadj,
            adjncy,
            vwgt: weights.then(|| (begin..end).map(|v| v % 3 + 1).collect()),
        }
    }

    #[test]
    fn edge_cut_and_imbalance_of_a_path() {
        // 9 vertices, parts 0 0 1 1 1 0 2 2 2: cuts at 1–2, 4–5, 5–6.
        let all = [0, 0, 1, 1, 1, 0, 2, 2, 2];
        // Weights v % 3 + 1: 1 2 3 1 2 3 1 2 3; parts 0: 1+2+3 = 6, 1: 3+1+2 = 6, 2: 6.
        for counts in [vec![9], vec![4, 5], vec![2, 0, 7], vec![3, 3, 2, 1]] {
            let out = ThreadComm::run(counts.len(), |comm| {
                let g = path(comm.rank(), &counts, true);
                let first = g.vtxdist[comm.rank()] as usize;
                let part = &all[first..first + g.xadj.len() - 1];
                let unweighted = DistGraph {
                    vwgt: None,
                    ..g.clone()
                };
                (
                    edge_cut(&comm, &g, part).unwrap(),
                    weighted_imbalance(&comm, &g, part, 3).unwrap(),
                    weighted_imbalance(&comm, &unweighted, part, 3).unwrap(),
                    weighted_imbalance(&comm, &g, part, 4).unwrap(),
                    parts_hash(&comm, part),
                )
            });
            for (cut, w, u, four, _) in &out {
                assert_eq!(*cut, 3, "{counts:?}");
                assert!(w.abs() < 1e-15, "{counts:?}: {w}");
                // Unit weights: sizes 3, 3, 3.
                assert!(u.abs() < 1e-15, "{counts:?}: {u}");
                // Four parts, one empty: 6 · 4 / 18 − 1.
                assert!((four - 1.0 / 3.0).abs() < 1e-15, "{counts:?}: {four}");
            }
            assert!(out.iter().all(|o| o.4 == out[0].4));
        }
        // Everything in one part: no cut, imbalance nparts − 1.
        let g = path(0, &[5], false);
        assert_eq!(edge_cut(&Serial, &g, &[1; 5]).unwrap(), 0);
        assert!((weighted_imbalance(&Serial, &g, &[1; 5], 2).unwrap() - 1.0).abs() < 1e-15);
        let empty = path(0, &[0], true);
        assert_eq!(edge_cut(&Serial, &empty, &[]).unwrap(), 0);
        assert_eq!(weighted_imbalance(&Serial, &empty, &[], 2).unwrap(), 0.0);
    }

    #[test]
    fn host_dual_graph_is_symmetric_with_node_count_weights() {
        let h = crate::hex_prism_pyramid_box(3, 2);
        let g = host_dual_graph(&h);
        let n = h.num_cells();
        assert_eq!(g.xadj.len(), n + 1);
        let row = |u: usize| &g.adjncy[g.xadj[u] as usize..g.xadj[u + 1] as usize];
        for u in 0..n {
            for &v in row(u) {
                assert_ne!(v, u as i64);
                assert!(row(v as usize).contains(&(u as i64)));
            }
        }
        let w = g.vwgt.as_ref().unwrap();
        // P2: hexahedron 27, prism 18, pyramid 14 nodes.
        let want = 27 * num_cells_of(&h, CellType::Hexahedron)
            + 18 * num_cells_of(&h, CellType::Prism)
            + 14 * num_cells_of(&h, CellType::Pyramid);
        assert_eq!(w.iter().sum::<i64>(), want as i64);
    }

    /// The slab partition cuts exactly the bound.
    #[test]
    fn slab_cut_bound_is_the_cut_of_the_slab_partition() {
        for kind in [BoxKind::Tetrahedra, BoxKind::HexPrismPyramid] {
            for n in [3, 4, 6] {
                let mesh = PartitionMesh { name: "", kind, n };
                let h = mesh.host(1);
                let g = host_dual_graph(&h);
                for nparts in 1..=n {
                    let part = slab_parts(&h, n, nparts);
                    let cut = edge_cut(&Serial, &g, &part).unwrap();
                    assert_eq!(
                        cut,
                        slab_cut_bound(&mesh, nparts),
                        "{kind:?}, n = {n}, {nparts} parts"
                    );
                }
            }
        }
    }
}
