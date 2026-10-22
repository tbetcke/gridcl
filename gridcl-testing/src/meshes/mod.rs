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

use cubecl::prelude::Client;
use gridcl::{Dim, EmbeddedIn, Mesh, Real, Result, TopoDim};
use gridcl_dims::CellType;

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
    pub fn build<F, TD, GD>(&self, client: &Client) -> Result<Mesh<F, TD, GD>>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let _ = client;
        todo!()
    }

    /// The mesh with interior nodes moved by a smooth field of size `amplitude` (relative to the
    /// unit box) that vanishes on the boundary of [0, 1]^gdim. Nodes of pyramid cells stay in
    /// place, so pyramid cells stay affine (docs/plans/M3.md, question 24).
    pub fn perturbed(&self, amplitude: f64) -> HostMesh {
        let _ = amplitude;
        todo!()
    }

    /// The mesh with node and cell origin ids replaced by a random injective map onto
    /// [−2^40, 2^40], seeded by `seed`.
    pub fn with_shuffled_ids(&self, seed: u64) -> HostMesh {
        let _ = seed;
        todo!()
    }

    /// The mesh with its nodes stored in a random order seeded by `seed`; every node keeps its
    /// origin id and coordinates.
    pub fn with_shuffled_nodes(&self, seed: u64) -> HostMesh {
        let _ = seed;
        todo!()
    }

    /// The mesh with only the first `n` cells of every block (fewer if a block is smaller), and
    /// without the nodes no remaining cell uses.
    pub fn first_cells(&self, n: usize) -> HostMesh {
        let _ = n;
        todo!()
    }
}
