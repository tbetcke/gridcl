//! `IndexMap::with_ghosts` on `ThreadComm` with 1–5 ranks (docs/plans/M6.md, T5): random ghost
//! layouts against a host reference, and the agreed errors. `try_clone` of topologies and
//! geometries with ghosted maps (T6).

use gridcl::{
    Cell1, Cell2, CellBlock, CommDyn, CoordinateElement, D1, D2, DeviceBuffer, DeviceStrided,
    GeometryBlock, IndexMap, NodeFamily,
};
use gridcl_geometry::Geometry;
use gridcl_testing::{ThreadComm, comm_checks};
use gridcl_topology::{Topology, encode_origin_ids};
use proptest::prelude::*;

#[test]
fn with_ghosts_without_ghosts_equals_new_and_errors_are_agreed() {
    for n in 1..=5 {
        ThreadComm::run(n, |comm| comm_checks::index_map_basics(&comm));
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn random_ghost_layouts(n in 1..=5_usize, seed in any::<u64>()) {
        ThreadComm::run(n, |comm| comm_checks::index_map_layout(&comm, seed));
    }
}

/// Two owned indices per rank; on two or more ranks each rank also ghosts the first index of
/// the next rank.
fn ring_map(comm: &dyn CommDyn) -> IndexMap {
    let (r, n) = (comm.rank(), comm.size());
    let (ghosts, owners) = if n > 1 {
        let next = (r + 1) % n;
        (vec![2 * next as i64], vec![next as i32])
    } else {
        (vec![], vec![])
    };
    IndexMap::with_ghosts(comm, 2, ghosts, owners).unwrap()
}

fn assert_same_map(a: &IndexMap, b: &IndexMap) {
    assert_eq!(a.owned_range(), b.owned_range());
    assert_eq!(a.size_global(), b.size_global());
    assert_eq!(a.ghosts(), b.ghosts());
    assert_eq!(a.ghost_owners(), b.ghost_owners());
    assert_eq!(a.send_lists(), b.send_lists());
    assert_eq!(a.recv_lists(), b.recv_lists());
}

#[test]
fn try_clone_keeps_ghosted_maps() {
    for n in 1..=4 {
        ThreadComm::run(n, |comm| {
            let client = gridcl_testing::test_client();
            let vertices = ring_map(&comm);
            let nv = vertices.num_local();
            let (lo, _) = vertices.owned_range();
            let mut ids: Vec<i64> = (lo..lo + 2).collect();
            ids.extend_from_slice(vertices.ghosts());
            // One triangle over the local vertices; on one rank the last vertex repeats.
            let rows: Vec<u32> = (0..3).map(|i| i.min(nv as u32 - 1)).collect();
            let block = CellBlock::<D2>::new(
                Cell2::Triangle,
                IndexMap::with_ghosts(&comm, 1, vec![], vec![]).unwrap(),
                DeviceStrided {
                    data: DeviceBuffer::from_host(&client, &rows).unwrap(),
                    stride: 3,
                },
                DeviceBuffer::from_host(&client, &encode_origin_ids(&[comm.rank() as i64]))
                    .unwrap(),
                DeviceBuffer::from_host(&client, &[0u32]).unwrap(),
            );
            let topology = Topology::new(&client, vec![block], vertices.clone(), &ids).unwrap();
            let copy = topology.try_clone(&comm).unwrap();
            assert_same_map(copy.vertex_index_map(), topology.vertex_index_map());
            assert_same_map(
                &copy.cell_blocks()[0].index_map,
                &topology.cell_blocks()[0].index_map,
            );
            assert_eq!(copy.cell_blocks()[0].vertices.data.to_host().unwrap(), rows);

            let element = CoordinateElement::<D1> {
                cell: Cell1::Interval,
                degree: 1,
                family: NodeFamily::Equispaced,
            };
            let coords: Vec<f32> = (0..2 * nv).map(|i| i as f32).collect();
            let geometry = Geometry::<f32, D1, D2>::new(
                vec![GeometryBlock {
                    element,
                    dofmap: DeviceStrided {
                        data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(),
                        stride: 2,
                    },
                }],
                DeviceBuffer::from_host(&client, &coords).unwrap(),
                vertices,
                [0.0, 0.0],
            )
            .unwrap();
            let copy = geometry.try_clone(&comm).unwrap();
            assert_same_map(copy.node_index_map(), geometry.node_index_map());
            assert_eq!(copy.nodes().to_host().unwrap(), coords);
        });
    }
}
