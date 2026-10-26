//! Benchmark hooks of the private mesh families (ADR-0015, item 4). Only with the Cargo feature
//! `bench`, which gridcl-bench turns on; not part of the library API and without any stability
//! promise.
//!
//! Each hook runs exactly one family's launch sequence on inputs already on the device, the
//! cell-fastest connectivity copy included, and returns the outputs in an opaque owner, so the
//! caller can sync, time and drop them.

use gridcl_device::{DeviceBuffer, GridError, Result};
use gridcl_dims::TopoDim;

use crate::Topology;
use crate::emission::emission_plans;
use crate::entity::{cell_counts, last_vertex};

/// The output buffers of one hook: held until dropped, so the timed launches stay alive until
/// the caller has synchronised.
pub struct BenchOutput {
    buffers: Vec<DeviceBuffer<u32>>,
}

impl BenchOutput {
    /// Words held, padding excluded.
    pub fn len(&self) -> usize {
        self.buffers.iter().map(DeviceBuffer::len).sum()
    }

    /// Whether no words are held.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Host copies of the held buffers, in order, padding excluded: for [`emit_keys`], the
    /// planar key words (`SortKeys::words`) and then the payloads of each emitted entity type;
    /// for [`permutation_info`], the orientation words. For the benchmarks' checks.
    ///
    /// # Errors
    ///
    /// Device read errors.
    pub fn to_host(&self) -> Result<Vec<Vec<u32>>> {
        self.buffers.iter().map(DeviceBuffer::to_host).collect()
    }
}

/// The key emission of `create_entities(dim)` ("Entity creation", step 1) for every entity type
/// of dimension `dim`.
///
/// # Errors
///
/// `EntitiesNotCreated(dim)` unless 1 ≤ dim < tdim; allocation and device errors.
pub fn emit_keys<TD: TopoDim>(topology: &Topology<TD>, dim: usize) -> Result<BenchOutput> {
    if dim == 0 || dim >= TD::N {
        return Err(GridError::EntitiesNotCreated(dim));
    }
    let cells = topology.cell_rows();
    let last_vertex = last_vertex(topology.vertex_origin_ids())?;
    let mut buffers = Vec::new();
    for plan in emission_plans(&cell_counts(&cells), dim)? {
        if plan.total == 0 {
            continue;
        }
        let (keys, payload) =
            crate::keys::emit_keys(topology.client(), &cells, dim, &plan, last_vertex)?;
        buffers.push(keys.words().share());
        buffers.push(payload);
    }
    Ok(BenchOutput { buffers })
}

/// The PermutationInfo launch of cell block `b`.
///
/// # Errors
///
/// `IndexOutOfRange` if there is no block `b`; allocation and device errors.
pub fn permutation_info<TD: TopoDim>(topology: &Topology<TD>, b: usize) -> Result<BenchOutput> {
    let blocks = topology.cell_blocks();
    let block = blocks.get(b).ok_or(GridError::IndexOutOfRange {
        index: b,
        len: blocks.len(),
    })?;
    let out = crate::permutation::permutation_info(
        block.cell.into(),
        &block.vertices,
        topology.vertex_origin_ids(),
        topology.ids_fit_u32(),
    )?;
    Ok(BenchOutput {
        buffers: out.into_iter().collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CellBlock, encode_origin_ids};
    use gridcl_device::DeviceStrided;
    use gridcl_dims::D3;
    use gridcl_parallel::{IndexMap, Serial};

    /// The topology of the P1 mixed box, with its node ids as vertex ids.
    fn mixed_topology(client: &cubecl::prelude::Client) -> Topology<D3> {
        let h = gridcl_testing::mixed_box(2, 1).with_shuffled_ids(7);
        let blocks = h
            .blocks
            .iter()
            .map(|b| {
                let nv = gridcl_refcell::vertices(b.cell).len();
                let n = b.cells.len() / nv;
                CellBlock::<D3>::new(
                    b.cell.try_into().unwrap(),
                    IndexMap::new(&Serial, n).unwrap(),
                    DeviceStrided {
                        data: DeviceBuffer::from_host(client, &b.cells).unwrap(),
                        stride: nv as u32,
                    },
                    DeviceBuffer::from_host(
                        client,
                        &encode_origin_ids(&(0..n as i64).collect::<Vec<_>>()),
                    )
                    .unwrap(),
                    DeviceBuffer::from_host(client, &vec![0u32; n]).unwrap(),
                )
            })
            .collect();
        let ids: Vec<i64> = (0..h.num_nodes()).map(|i| h.node_id(i)).collect();
        Topology::new(
            client,
            blocks,
            IndexMap::new(&Serial, ids.len()).unwrap(),
            &ids,
        )
        .unwrap()
    }

    /// The hooks reproduce what the library computes: PermutationInfo's words, and the keys and
    /// payloads of every emitted position of every entity type.
    #[test]
    fn hooks_match_the_library() {
        let client = gridcl_testing::test_client();
        let topology = mixed_topology(&client);
        for b in 0..topology.cell_blocks().len() {
            let out = permutation_info(&topology, b).unwrap();
            let want = topology.permutation_info(b).unwrap();
            assert_eq!(out.len(), want.len());
            assert_eq!(
                out.buffers[0].to_host().unwrap(),
                want.to_host().unwrap(),
                "block {b}"
            );
        }
        let counts = cell_counts(&topology.cell_rows());
        for dim in 1..3 {
            let out = emit_keys(&topology, dim).unwrap();
            let want: usize = emission_plans(&counts, dim)
                .unwrap()
                .iter()
                .map(|p| p.total * (gridcl_refcell::vertices(p.entity).len() + 1))
                .sum();
            assert!(want > 0);
            assert_eq!(out.len(), want, "dim {dim}");
        }
        for dim in [0, 3] {
            assert!(matches!(
                emit_keys(&topology, dim),
                Err(GridError::EntitiesNotCreated(d)) if d == dim
            ));
        }
        let len = topology.cell_blocks().len();
        assert!(matches!(
            permutation_info(&topology, len),
            Err(GridError::IndexOutOfRange { .. })
        ));
    }
}
