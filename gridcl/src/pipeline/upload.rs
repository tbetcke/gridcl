//! Stage 9: the device-resident mesh of the stage-8 numbering (docs/design.md, "Parallel
//! creation pipeline"; ADR-0026, item 14).

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock};
use gridcl_parallel::{CommDyn, IndexMap};
use gridcl_topology::{CellBlock, Topology, encode_origin_ids};

use super::nodes::Nodes;
use super::number::{MapInput, Numbered};
use super::{BlockSpec, agree};
use crate::builder::cell_name;

/// The `IndexMap` of `m`. Collective.
fn index_map(comm: &dyn CommDyn, m: &MapInput) -> Result<IndexMap> {
    IndexMap::with_ghosts(comm, m.num_owned, m.ghosts.clone(), m.ghost_owners.clone())
}

/// A u32 of `n`.
fn index_u32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| GridError::TooLarge)
}

/// Stage 9: creates the index maps of `numbered` and uploads its cells (vertex rows, origin
/// ids, zero flags), dofmaps, vertex origin ids and node coordinates relative to the agreed
/// origin of `nodes`, in `F`. `Topology::new` computes the permutation info, and
/// `Topology::set_vertex_sharers` uploads the vertex shared flags. Collective.
///
/// # Errors
///
/// Agreed over the ranks: errors of `IndexMap::with_ghosts`; `WrongCellDim` for a block of
/// another dimension than `TD`; allocation and device errors.
pub(crate) fn upload<F, TD, GD>(
    client: &Client,
    comm: &dyn CommDyn,
    blocks: &[BlockSpec],
    nodes: &Nodes,
    numbered: Numbered,
) -> Result<(Topology<TD>, Geometry<F, TD, GD>)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    // The collective part first, so a local failure below cannot leave a rank waiting.
    let cell_maps = numbered
        .blocks
        .iter()
        .map(|b| index_map(comm, &b.map))
        .collect::<Result<Vec<_>>>()?;
    let vertex_map = index_map(comm, &numbered.vertex_map)?;
    let node_map = index_map(comm, &numbered.node_map)?;

    let local = (|| {
        let gd = GD::N;
        let mut origin = GD::Array::<f64>::default();
        origin.as_mut().copy_from_slice(&nodes.origin());
        let relative: Vec<F> = numbered
            .coords
            .chunks_exact(gd)
            .flat_map(|x| {
                x.iter()
                    .zip(origin.as_ref())
                    .map(|(&x, &o)| F::narrow(x - o))
            })
            .collect();

        let mut cell_blocks = Vec::with_capacity(blocks.len());
        let mut geometry_blocks = Vec::with_capacity(blocks.len());
        for ((spec, block), map) in blocks.iter().zip(&numbered.blocks).zip(cell_maps) {
            let cell = TD::Cell::try_from(spec.cell).map_err(|e| GridError::WrongCellDim {
                found: cell_name(e.found),
                expected_tdim: e.expected_tdim,
            })?;
            let n = block.cells.len();
            cell_blocks.push(CellBlock::new(
                cell,
                map,
                DeviceStrided {
                    data: DeviceBuffer::from_host(client, &block.vertices)?,
                    stride: index_u32(spec.num_vertices())?,
                },
                DeviceBuffer::from_host(client, &encode_origin_ids(&block.cells.ids))?,
                DeviceBuffer::zeros(client, n)?,
            ));
            geometry_blocks.push(GeometryBlock {
                element: CoordinateElement {
                    cell,
                    degree: spec.degree,
                    family: spec.family,
                },
                dofmap: DeviceStrided {
                    data: DeviceBuffer::from_host(client, &block.dofmap)?,
                    stride: index_u32(spec.num_nodes())?,
                },
            });
        }
        let mut topology = Topology::new(client, cell_blocks, vertex_map, &numbered.vertex_ids)?;
        topology.set_vertex_sharers(numbered.sharers);
        let geometry = Geometry::new(
            geometry_blocks,
            DeviceBuffer::from_host(client, &relative)?,
            node_map,
            origin,
        )?;
        Ok((topology, geometry))
    })();
    agree(comm, local)
}
