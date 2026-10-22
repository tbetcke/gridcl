//! Test meshes with origin ids at any degree p (docs/design.md, "Testing and validation"):
//! structured boxes of every cell type and the mixed box, curved sphere surfaces, a mixed
//! triangle–quadrilateral surface and curved balls.
//!
//! Generators return a [`HostMesh`] of host arrays; [`HostMesh::build`] turns it into a
//! `Mesh` through `MeshBuilder`.

mod ball;
mod boxes;
mod lattice;
mod sphere;
mod surface;

pub use ball::ball;
pub use boxes::{cube_box, interval_box, mixed_box, square_box, triangle_strip};
pub use sphere::sphere;
pub use surface::mixed_surface;

use std::collections::HashSet;

use cubecl::prelude::Client;
use gridcl::{
    CoordinateElement, Dim, EmbeddedIn, GridError, Mesh, MeshBuilder, NodeFamily, Real, Result,
    TopoDim,
};
use gridcl_dims::CellType;

use lattice::{PointGrid, SplitMix64};

/// A mesh as host arrays, in the form `MeshBuilder` takes.
#[derive(Clone, Debug, PartialEq)]
pub struct HostMesh {
    /// The geometric dimension.
    pub gdim: usize,
    /// Node coordinates, [node][gdim].
    pub nodes: Vec<f64>,
    /// Node origin ids; `None` means the node index.
    pub node_ids: Option<Vec<i64>>,
    /// The cell blocks, at most one per cell type.
    pub blocks: Vec<HostBlock>,
}

/// One cell block of a [`HostMesh`]: equispaced Lagrange cells of one type and degree.
#[derive(Clone, Debug, PartialEq)]
pub struct HostBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// Cell → node lists in A.5 order, one row of `num_lagrange_nodes(cell, degree)` per cell.
    pub cells: Vec<u32>,
    /// Cell origin ids; `None` means the cell index in the block.
    pub cell_ids: Option<Vec<i64>>,
}

impl HostBlock {
    /// Nodes per cell.
    pub fn nodes_per_cell(&self) -> usize {
        gridcl_refcell::num_lagrange_nodes(self.cell, self.degree)
    }

    /// The number of cells.
    pub fn num_cells(&self) -> usize {
        self.cells.len() / self.nodes_per_cell()
    }
}

impl HostMesh {
    /// The number of nodes.
    pub fn num_nodes(&self) -> usize {
        self.nodes.len() / self.gdim
    }

    /// The number of cells over all blocks.
    pub fn num_cells(&self) -> usize {
        self.blocks.iter().map(HostBlock::num_cells).sum()
    }

    /// The origin id of node `i`.
    pub fn node_id(&self, i: usize) -> i64 {
        self.node_ids.as_ref().map_or(i as i64, |ids| ids[i])
    }

    /// The mesh on `client`, through `MeshBuilder`, with equispaced coordinate elements.
    ///
    /// # Errors
    ///
    /// `DimensionMismatch` if `GD::N` is not `gdim` (expected (TD::N, gdim)), `WrongCellDim` if a
    /// block's cell is not of dimension `TD::N`, and errors of `MeshBuilder::build`.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use gridcl::{D2, Mesh};
    /// use gridcl_dims::Cell2;
    ///
    /// let client = gridcl_testing::test_client();
    /// let h = gridcl_testing::square_box(Cell2::Triangle, 2, 2);
    /// let mesh: Mesh<f32, D2, D2> = h.build(&client).expect("mesh");
    /// ```
    pub fn build<F, TD, GD>(&self, client: &Client) -> Result<Mesh<F, TD, GD>>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let elements = self.elements::<TD, GD>()?;
        let mut builder =
            MeshBuilder::<F, TD, GD>::new(client).nodes(&self.nodes, self.node_ids.as_deref());
        for (element, block) in elements.into_iter().zip(&self.blocks) {
            builder = builder.cell_block(element, &block.cells, block.cell_ids.as_deref());
        }
        builder.build()
    }

    /// The coordinate element of every block, after the dimension checks of [`HostMesh::build`].
    fn elements<TD, GD>(&self) -> Result<Vec<CoordinateElement<TD>>>
    where
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        if GD::N != self.gdim {
            return Err(GridError::DimensionMismatch {
                expected: (TD::N, self.gdim),
                found: (TD::N, GD::N),
            });
        }
        self.blocks
            .iter()
            .map(|block| {
                let cell = TD::Cell::try_from(block.cell).map_err(|e| GridError::WrongCellDim {
                    found: cell_name(e.found),
                    expected_tdim: e.expected_tdim,
                })?;
                Ok(CoordinateElement {
                    cell,
                    degree: block.degree,
                    family: NodeFamily::Equispaced,
                })
            })
            .collect()
    }

    /// The mesh with interior nodes moved by a smooth field of size `amplitude` (relative to the
    /// unit box) that vanishes on the boundary of [0, 1]^gdim. Nodes of pyramid cells stay in
    /// place, so pyramid cells stay affine (docs/plans/M3.md, question 24).
    ///
    /// The field is d_g(x) = amplitude · c_g · Π_i sin(π x_i) · min(1, δ(x)), with
    /// c = (1, 0.7, 0.4) and δ(x) the distance from x to the nearest node of a pyramid cell
    /// (δ = ∞ without pyramids, so the cut-off is 1). The cut-off makes the field vanish at the
    /// pinned pyramid nodes with a gradient bounded independently of the cell size; without it,
    /// the hexahedra of the mixed box, whose faces are all pinned, fold at fine n and high p.
    /// Nodes on the boundary of the unit box are not moved at all. Amplitude 0.05 keeps
    /// det J > 0 on every box at p ≤ 4.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_dims::Cell2;
    ///
    /// let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1);
    /// let moved = h.perturbed(0.05);
    /// // Only the centre node (0.5, 0.5) is interior: it moves by (0.05, 0.035).
    /// let changed: Vec<usize> = (0..h.num_nodes())
    ///     .filter(|&i| h.nodes[2 * i..2 * i + 2] != moved.nodes[2 * i..2 * i + 2])
    ///     .collect();
    /// assert_eq!(changed.len(), 1);
    /// let i = changed[0];
    /// assert!((moved.nodes[2 * i] - 0.55).abs() < 1e-15);
    /// assert!((moved.nodes[2 * i + 1] - 0.535).abs() < 1e-15);
    /// ```
    pub fn perturbed(&self, amplitude: f64) -> HostMesh {
        const C: [f64; 3] = [1.0, 0.7, 0.4];
        const EPS: f64 = 1e-12;
        let gd = self.gdim;
        let mut fixed = vec![false; self.num_nodes()];
        for block in self.blocks.iter().filter(|b| b.cell == CellType::Pyramid) {
            for &n in &block.cells {
                fixed[n as usize] = true;
            }
        }
        let pinned: Vec<f64> = (0..self.num_nodes())
            .filter(|&i| fixed[i])
            .flat_map(|i| self.nodes[i * gd..(i + 1) * gd].iter().copied())
            .collect();
        let grid = PointGrid::new(gd, &pinned);
        let mut out = self.clone();
        for (x, fixed) in out.nodes.chunks_exact_mut(gd).zip(fixed) {
            let on_boundary = x.iter().any(|&xi| xi <= EPS || xi >= 1.0 - EPS);
            if fixed || on_boundary {
                continue;
            }
            let cutoff = grid.distance(x, 1.0);
            let s: f64 = x
                .iter()
                .map(|&xi| (core::f64::consts::PI * xi).sin())
                .product();
            for (xg, c) in x.iter_mut().zip(C) {
                *xg += amplitude * c * s * cutoff;
            }
        }
        out
    }

    /// The mesh with node and cell origin ids replaced by a random injective map onto
    /// [−2^40, 2^40], seeded by `seed`.
    ///
    /// Node ids are distinct from each other, and cell ids are distinct over all blocks.
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::interval_box(3, 1).with_shuffled_ids(7);
    /// let ids = h.node_ids.as_ref().unwrap();
    /// assert!(ids.iter().all(|id| id.abs() <= 1 << 40));
    /// let mut sorted = ids.clone();
    /// sorted.sort();
    /// sorted.dedup();
    /// assert_eq!(sorted.len(), 4);
    /// ```
    pub fn with_shuffled_ids(&self, seed: u64) -> HostMesh {
        const SPAN: u64 = (1 << 41) + 1;
        let mut rng = SplitMix64::new(seed);
        let mut fresh = |used: &mut HashSet<i64>| loop {
            let id = rng.below(SPAN) as i64 - (1 << 40);
            if used.insert(id) {
                return id;
            }
        };
        let mut out = self.clone();
        let mut used = HashSet::new();
        out.node_ids = Some((0..self.num_nodes()).map(|_| fresh(&mut used)).collect());
        let mut used = HashSet::new();
        for block in &mut out.blocks {
            block.cell_ids = Some((0..block.num_cells()).map(|_| fresh(&mut used)).collect());
        }
        out
    }

    /// The mesh with its nodes stored in a random order seeded by `seed`; every node keeps its
    /// origin id and coordinates.
    ///
    /// Node ids are always `Some` afterwards: index-based ids are materialised first.
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::interval_box(4, 2);
    /// let s = h.with_shuffled_nodes(3);
    /// // Every node of every cell is where it was, under the same id.
    /// for (a, b) in h.blocks[0].cells.iter().zip(&s.blocks[0].cells) {
    ///     let (a, b) = (*a as usize, *b as usize);
    ///     assert_eq!(h.nodes[a], s.nodes[b]);
    ///     assert_eq!(h.node_id(a), s.node_id(b));
    /// }
    /// ```
    pub fn with_shuffled_nodes(&self, seed: u64) -> HostMesh {
        let n = self.num_nodes();
        let gd = self.gdim;
        let mut rng = SplitMix64::new(seed);
        // perm[new] = old (Fisher–Yates).
        let mut perm: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = rng.below(i as u64 + 1) as usize;
            perm.swap(i, j);
        }
        let mut new_of_old = vec![0u32; n];
        for (new, &old) in perm.iter().enumerate() {
            new_of_old[old] = u32::try_from(new).expect("node index fits u32");
        }
        let mut out = self.clone();
        out.nodes = perm
            .iter()
            .flat_map(|&old| self.nodes[old * gd..(old + 1) * gd].iter().copied())
            .collect();
        out.node_ids = Some(perm.iter().map(|&old| self.node_id(old)).collect());
        for block in &mut out.blocks {
            for node in &mut block.cells {
                *node = new_of_old[*node as usize];
            }
        }
        out
    }

    /// The mesh with only the first `n` cells of every block (fewer if a block is smaller), and
    /// without the nodes no remaining cell uses.
    ///
    /// The remaining nodes keep their relative order, coordinates and origin ids (index-based
    /// ids are materialised first); blocks are kept even when they become empty.
    ///
    /// # Example
    ///
    /// ```
    /// let h = gridcl_testing::interval_box(4, 1).first_cells(2);
    /// assert_eq!((h.num_cells(), h.num_nodes()), (2, 3));
    /// assert_eq!(h.nodes, [0.0, 0.25, 0.5]);
    /// let empty = gridcl_testing::interval_box(4, 1).first_cells(0);
    /// assert_eq!((empty.num_cells(), empty.num_nodes(), empty.blocks.len()), (0, 0, 1));
    /// ```
    pub fn first_cells(&self, n: usize) -> HostMesh {
        let gd = self.gdim;
        let mut out = self.clone();
        let mut used = vec![false; self.num_nodes()];
        for block in &mut out.blocks {
            let keep = n.min(block.num_cells());
            block.cells.truncate(keep * block.nodes_per_cell());
            if let Some(ids) = &mut block.cell_ids {
                ids.truncate(keep);
            }
            for &node in &block.cells {
                used[node as usize] = true;
            }
        }
        let mut new_index = vec![u32::MAX; self.num_nodes()];
        let mut nodes = Vec::new();
        let mut ids = Vec::new();
        for old in (0..self.num_nodes()).filter(|&i| used[i]) {
            new_index[old] = u32::try_from(ids.len()).expect("node index fits u32");
            nodes.extend_from_slice(&self.nodes[old * gd..(old + 1) * gd]);
            ids.push(self.node_id(old));
        }
        out.nodes = nodes;
        out.node_ids = Some(ids);
        for block in &mut out.blocks {
            for node in &mut block.cells {
                *node = new_index[*node as usize];
            }
        }
        out
    }
}

/// The name of a cell type, for `GridError`.
fn cell_name(cell: CellType) -> &'static str {
    match cell {
        CellType::Point => "Point",
        CellType::Interval => "Interval",
        CellType::Triangle => "Triangle",
        CellType::Quadrilateral => "Quadrilateral",
        CellType::Tetrahedron => "Tetrahedron",
        CellType::Hexahedron => "Hexahedron",
        CellType::Prism => "Prism",
        CellType::Pyramid => "Pyramid",
    }
}
