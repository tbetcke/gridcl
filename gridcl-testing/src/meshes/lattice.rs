//! Lattice and node helpers shared by the generators: nodes sit on a global integer lattice of
//! resolution 2·n·p and are deduplicated by lattice key, so shared high-order nodes are unique.
//!
//! A cell is given by the lattice keys of its vertices. Its Lagrange nodes are the equispaced
//! reference points of `gridcl_refcell::lagrange_points`, scaled by p to integers, and mapped
//! affinely through the vertices in exact integer arithmetic, so equal points get equal keys.

use std::collections::HashMap;
use std::hash::Hash;

use gridcl_dims::CellType;
use gridcl_refcell::NodeFamily;

use super::{HostBlock, HostMesh};

/// A lattice point: integer coordinates, unused trailing coordinates 0.
pub(super) type Key = [i64; 3];

/// The vertices whose differences to vertex 0 are the axes of the affine map of `cell`
/// (Appendix A.1): x = v0 + Σ_k r_k (v_{a_k} − v0).
fn axis_vertices(cell: CellType) -> &'static [usize] {
    match cell {
        CellType::Point => &[],
        CellType::Interval => &[1],
        CellType::Triangle | CellType::Quadrilateral => &[1, 2],
        CellType::Tetrahedron | CellType::Prism => &[1, 2, 3],
        CellType::Hexahedron | CellType::Pyramid => &[1, 2, 4],
    }
}

/// The axes v_{a_k} − v0 of the affine map through `verts`.
fn axes(cell: CellType, verts: &[Key]) -> Vec<Key> {
    axis_vertices(cell)
        .iter()
        .map(|&a| core::array::from_fn(|g| verts[a][g] - verts[0][g]))
        .collect()
}

/// The determinant of the square matrix of `axes` restricted to the first `axes.len()`
/// coordinates: the sign of det J of the affine map, when tdim = gdim.
fn axis_det(axes: &[Key]) -> i128 {
    let m = |k: usize, g: usize| i128::from(axes[k][g]);
    match axes.len() {
        1 => m(0, 0),
        2 => m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0),
        3 => {
            m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
        }
        n => unreachable!("no affine map with {n} axes"),
    }
}

/// Reorders the vertices of a full-dimensional affine cell so that det J > 0, keeping the tensor
/// order of quadrilateral faces: it swaps the axis vertices 1 and 2 (and 4 and 5 of a prism).
///
/// # Panics
///
/// If the cell is degenerate, or if it is a hexahedron or quadrilateral with det J < 0 (the
/// generators make those positive by construction).
pub(super) fn orient(cell: CellType, verts: &mut [Key]) {
    let det = axis_det(&axes(cell, verts));
    assert!(det != 0, "degenerate {cell:?} cell {verts:?}");
    if det > 0 {
        return;
    }
    match cell {
        CellType::Triangle | CellType::Tetrahedron | CellType::Pyramid => verts.swap(1, 2),
        CellType::Prism => {
            verts.swap(1, 2);
            verts.swap(4, 5);
        }
        _ => panic!("{cell:?} cell {verts:?} is negatively oriented"),
    }
    debug_assert!(axis_det(&axes(cell, verts)) > 0);
}

/// One cell block being generated.
struct BlockBuilder {
    cell: CellType,
    degree: u32,
    /// The reference Lagrange points times the degree, [node][tdim] as integers.
    ref_points: Vec<Key>,
    cells: Vec<u32>,
}

/// Collects cells, deduplicates their nodes by key and produces a [`HostMesh`].
///
/// Cells are given by the lattice keys ([`Key`]) of their vertices; their nodes are
/// deduplicated by a global key `K`. For the boxes `K` is the lattice key itself; the curved
/// generators map face-local lattice keys to a canonical global key ([`Assembler::cell_with`]).
pub(super) struct Assembler<K = Key> {
    gdim: usize,
    resolution: i64,
    index: HashMap<K, u32>,
    keys: Vec<K>,
    blocks: Vec<BlockBuilder>,
}

impl Assembler<Key> {
    /// An empty mesh of dimension `gdim` on the lattice with `resolution` points per unit length.
    pub(super) fn new(gdim: usize, resolution: i64) -> Self {
        assert!(resolution > 0);
        Assembler {
            resolution,
            ..Assembler::with_keys(gdim)
        }
    }

    /// Adds a cell of type `cell` and degree `degree` with the vertex keys `verts` (A.1 order).
    /// The cell must be an affine image of the reference cell; its nodes are the images of the
    /// reference Lagrange points.
    ///
    /// # Panics
    ///
    /// If `verts` has the wrong length, if a node falls off the lattice or if the cell is not
    /// affine (a node's vertex image differs from the given vertex).
    pub(super) fn cell(&mut self, cell: CellType, degree: u32, verts: &[Key]) {
        self.cell_with(cell, degree, verts, |k| k);
    }

    /// The mesh: node coordinates key / resolution, blocks in [`CellType`] order, index-based
    /// node ids and cell ids numbered consecutively over the blocks.
    pub(super) fn finish(self) -> HostMesh {
        let res = self.resolution as f64;
        self.finish_with(|k| k.map(|x| x as f64 / res))
    }
}

impl<K: Copy + Eq + Hash> Assembler<K> {
    /// An empty mesh of dimension `gdim` whose nodes are deduplicated by keys of type `K`.
    pub(super) fn with_keys(gdim: usize) -> Self {
        assert!((1..=3).contains(&gdim));
        Assembler {
            gdim,
            resolution: 1,
            index: HashMap::new(),
            keys: Vec::new(),
            blocks: Vec::new(),
        }
    }

    /// The node at `key`, created on first use: nodes are numbered in order of first use.
    fn node(&mut self, key: K) -> u32 {
        if let Some(&i) = self.index.get(&key) {
            return i;
        }
        let i = u32::try_from(self.keys.len()).expect("more than u32::MAX nodes");
        self.index.insert(key, i);
        self.keys.push(key);
        i
    }

    /// The block of `cell` at `degree`, created on first use.
    fn block(&mut self, cell: CellType, degree: u32) -> &mut BlockBuilder {
        let b = match self.blocks.iter().position(|b| b.cell == cell) {
            Some(b) => b,
            None => {
                let td = gridcl_refcell::tdim(cell);
                let points = gridcl_refcell::lagrange_points(cell, degree, NodeFamily::Equispaced)
                    .expect("equispaced Lagrange points");
                let scale = f64::from(degree);
                let ref_points = points
                    .chunks_exact(td)
                    .map(|r| {
                        core::array::from_fn(|k| {
                            if k >= td {
                                return 0;
                            }
                            let s = r[k] * scale;
                            let i = s.round();
                            assert!((s - i).abs() < 1e-9, "{cell:?} point off the lattice");
                            i as i64
                        })
                    })
                    .collect();
                self.blocks.push(BlockBuilder {
                    cell,
                    degree,
                    ref_points,
                    cells: Vec::new(),
                });
                self.blocks.len() - 1
            }
        };
        let block = &mut self.blocks[b];
        assert_eq!(block.degree, degree, "one degree per cell type");
        block
    }

    /// [`Assembler::cell`] with the node keys mapped by `canon` to the global key of the node.
    pub(super) fn cell_with(
        &mut self,
        cell: CellType,
        degree: u32,
        verts: &[Key],
        canon: impl Fn(Key) -> K,
    ) {
        let nv = gridcl_refcell::vertices(cell).len();
        assert_eq!(verts.len(), nv, "{cell:?} has {nv} vertices");
        let axes = axes(cell, verts);
        let p = i64::from(degree);
        let ref_points = core::mem::take(&mut self.block(cell, degree).ref_points);
        let mut row = Vec::with_capacity(ref_points.len());
        for (i, r) in ref_points.iter().enumerate() {
            let key: Key = core::array::from_fn(|g| {
                let sum: i64 = axes.iter().zip(r).map(|(a, &rk)| rk * a[g]).sum();
                assert!(sum % p == 0, "{cell:?} node off the lattice");
                verts[0][g] + sum / p
            });
            if i < nv {
                assert_eq!(key, verts[i], "{cell:?} cell {verts:?} is not affine");
            }
            row.push(self.node(canon(key)));
        }
        let block = self.block(cell, degree);
        block.ref_points = ref_points;
        block.cells.extend(row);
    }

    /// The mesh with node coordinates `coords(key)` (the first gdim of them), blocks in
    /// [`CellType`] order, index-based node ids and cell ids numbered consecutively over the
    /// blocks.
    pub(super) fn finish_with(mut self, coords: impl Fn(&K) -> [f64; 3]) -> HostMesh {
        let gdim = self.gdim;
        let nodes = self
            .keys
            .iter()
            .flat_map(|k| {
                let x = coords(k);
                (0..gdim).map(move |g| x[g])
            })
            .collect();
        let node_ids = Some((0..self.keys.len() as i64).collect());
        self.blocks.sort_by_key(|b| b.cell);
        let mut next_id = 0i64;
        let blocks = self
            .blocks
            .into_iter()
            .map(|b| {
                let k = b.ref_points.len();
                let n = (b.cells.len() / k) as i64;
                let ids = (next_id..next_id + n).collect();
                next_id += n;
                HostBlock {
                    cell: b.cell,
                    degree: b.degree,
                    cells: b.cells,
                    cell_ids: Some(ids),
                }
            })
            .collect();
        HostMesh {
            gdim: self.gdim,
            nodes,
            node_ids,
            blocks,
        }
    }
}

/// A bucket grid over a point set for nearest-distance queries.
pub(super) struct PointGrid<'a> {
    gdim: usize,
    points: &'a [f64],
    /// Buckets per unit length.
    scale: f64,
    buckets: HashMap<Key, Vec<usize>>,
}

impl<'a> PointGrid<'a> {
    /// The grid of `points` ([point][gdim]), about one point per bucket in the unit box.
    pub(super) fn new(gdim: usize, points: &'a [f64]) -> Self {
        let n = points.len() / gdim;
        let scale = (n as f64).powf(1.0 / gdim as f64).ceil().max(1.0);
        let mut grid = PointGrid {
            gdim,
            points,
            scale,
            buckets: HashMap::new(),
        };
        for i in 0..n {
            let key = grid.bucket(&points[i * gdim..(i + 1) * gdim]);
            grid.buckets.entry(key).or_default().push(i);
        }
        grid
    }

    fn bucket(&self, x: &[f64]) -> Key {
        core::array::from_fn(|g| {
            if g < self.gdim {
                (x[g] * self.scale).floor() as i64
            } else {
                0
            }
        })
    }

    /// min(cap, the distance from `x` to the nearest point).
    pub(super) fn distance(&self, x: &[f64], cap: f64) -> f64 {
        let centre = self.bucket(x);
        let mut best = cap;
        let span = |g: usize, r: i64| if g < self.gdim { -r..=r } else { 0..=0 };
        for r in 0i64.. {
            // Rings 0..r have been searched; points in the buckets of ring r or beyond are at
            // least (r − 1) / scale away.
            if best <= (r - 1) as f64 / self.scale {
                return best;
            }
            for a in span(0, r) {
                for b in span(1, r) {
                    for c in span(2, r) {
                        if a.abs().max(b.abs()).max(c.abs()) != r {
                            continue;
                        }
                        let key = [centre[0] + a, centre[1] + b, centre[2] + c];
                        for &i in self.buckets.get(&key).into_iter().flatten() {
                            let y = &self.points[i * self.gdim..(i + 1) * self.gdim];
                            let d2: f64 = x.iter().zip(y).map(|(u, v)| (u - v) * (u - v)).sum();
                            best = best.min(d2.sqrt());
                        }
                    }
                }
            }
        }
        unreachable!("the ring search ends once r exceeds cap · scale")
    }
}

/// SplitMix64: a small deterministic generator for shuffles.
#[derive(Clone, Debug)]
pub(super) struct SplitMix64(u64);

impl SplitMix64 {
    pub(super) fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub(super) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in 0..bound (bound > 0).
    pub(super) fn below(&mut self, bound: u64) -> u64 {
        self.next_u64() % bound
    }
}
