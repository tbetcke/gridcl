//! The hidden submesh hooks of `Topology` (M9 T1, ADR-0039 items 16–17): `from_device_ids`,
//! `set_cell_flags`, `vertex_sharers` and the checks of `set_vertex_sharers`.

use gridcl::prelude::*;
use gridcl_device::DeviceStrided;
use gridcl_dims::{Cell2, Cell3, TopoDim};
use gridcl_parallel::{IndexMap, Serial};
use gridcl_testing::HostMesh;
use gridcl_topology::{CellBlock, Topology, VertexSharers};

/// `from_device_ids` on a deep copy of `h`'s cell blocks and encoded vertex ids gives the
/// permutation info and origin ids of `Topology::new`, on the general id path.
fn check_from_device_ids<TD: TopoDim + EmbeddedIn<TD>>(h: &HostMesh) {
    let client = gridcl_testing::test_client();
    let mesh = h.build::<f32, TD, TD>(&client).expect("build");
    let blocks: Vec<CellBlock<TD>> = mesh
        .cell_blocks()
        .iter()
        .map(|b| b.try_clone(&Serial).unwrap())
        .collect();
    let ids = mesh.vertex_origin_ids().deep_clone().unwrap();
    let map = IndexMap::new(&Serial, mesh.vertex_index_map().num_local()).unwrap();
    let t = Topology::from_device_ids(&client, blocks, map, ids).unwrap();
    assert!(!t.ids_fit_u32());
    assert_eq!(
        t.vertex_origin_ids().to_host().unwrap(),
        mesh.vertex_origin_ids().to_host().unwrap()
    );
    for b in 0..mesh.cell_blocks().len() {
        let bits = |p: Option<&DeviceBuffer<u32>>| p.map(|p| p.to_host().unwrap());
        assert_eq!(
            bits(t.permutation_info(b)),
            bits(mesh.permutation_info(b)),
            "block {b}"
        );
    }
}

#[test]
fn from_device_ids_equals_new() {
    check_from_device_ids::<D1>(&gridcl_testing::interval_box(5, 2));
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        let h = gridcl_testing::square_box(cell, 3, 1);
        check_from_device_ids::<D2>(&h);
        check_from_device_ids::<D2>(&h.with_shuffled_ids(7));
    }
    for cell in [
        Cell3::Tetrahedron,
        Cell3::Hexahedron,
        Cell3::Prism,
        Cell3::Pyramid,
    ] {
        let h = gridcl_testing::cube_box(cell, 2, 1);
        check_from_device_ids::<D3>(&h);
        check_from_device_ids::<D3>(&h.with_shuffled_ids(11));
    }
    check_from_device_ids::<D3>(&gridcl_testing::mixed_box(2, 1));
    check_from_device_ids::<D3>(&gridcl_testing::hex_prism_pyramid_box(2, 1));
}

/// The two triangles (0, 1, 2) and (3, 2, 1).
fn two_triangles() -> Topology<D2> {
    let client = gridcl_testing::test_client();
    let block = CellBlock::<D2>::new(
        Cell2::Triangle,
        IndexMap::new(&Serial, 2).unwrap(),
        DeviceStrided {
            data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(),
            stride: 3,
        },
        DeviceBuffer::from_host(&client, &gridcl_topology::encode_origin_ids(&[0, 1])).unwrap(),
        DeviceBuffer::zeros(&client, 2).unwrap(),
    );
    Topology::new(
        &client,
        vec![block],
        IndexMap::new(&Serial, 4).unwrap(),
        &[0, 1, 2, 3],
    )
    .unwrap()
}

#[test]
fn from_device_ids_checks_its_input() {
    let t = two_triangles();
    let client = t.client().clone();
    let blocks = |t: &Topology<D2>| vec![t.cell_blocks()[0].try_clone(&Serial).unwrap()];
    // One word per vertex instead of two.
    let ids = DeviceBuffer::from_host(&client, &[0u32; 4]).unwrap();
    let map = IndexMap::new(&Serial, 4).unwrap();
    assert!(matches!(
        Topology::from_device_ids(&client, blocks(&t), map, ids),
        Err(GridError::LengthMismatch {
            expected: 8,
            found: 4
        })
    ));
}

#[test]
fn set_cell_flags_checks_its_input() {
    let mut t = two_triangles();
    let client = t.client().clone();
    let flags = DeviceBuffer::from_host(&client, &[1u32, 0]).unwrap();
    t.set_cell_flags(0, flags).unwrap();
    assert_eq!(t.cell_blocks()[0].flags.to_host().unwrap(), [1, 0]);
    let three = DeviceBuffer::from_host(&client, &[1u32, 0, 1]).unwrap();
    assert!(matches!(
        t.set_cell_flags(0, three),
        Err(GridError::LengthMismatch {
            expected: 2,
            found: 3
        })
    ));
    let two = DeviceBuffer::from_host(&client, &[0u32, 0]).unwrap();
    assert!(matches!(
        t.set_cell_flags(1, two),
        Err(GridError::IndexOutOfRange { index: 1, len: 1 })
    ));
    // The failed calls left the flags alone.
    assert_eq!(t.cell_blocks()[0].flags.to_host().unwrap(), [1, 0]);
}

/// Rank 1's rows for the four vertices: 0 and 3 unshared, 1 with rank 4, 2 with 0 and 2.
fn sharers() -> VertexSharers {
    VertexSharers {
        offsets: vec![0, 1, 3, 6, 7],
        ranks: vec![1, 1, 4, 0, 1, 2, 1],
        owned_cell: vec![true, true, false, true, true, false, true],
    }
}

#[test]
fn vertex_sharers_round_trip() {
    let mut t = two_triangles();
    assert_eq!(t.vertex_sharers(), None);
    t.set_vertex_sharers(sharers()).unwrap();
    assert_eq!(t.vertex_sharers(), Some(&sharers()));
    let c = t.try_clone(&Serial).unwrap();
    assert_eq!(c.vertex_sharers(), Some(&sharers()));
    // Rank 0 alone in every row: no vertex is shared, but the rows are stored.
    let mut t = two_triangles();
    t.set_vertex_sharers(VertexSharers {
        offsets: vec![0, 1, 2, 3, 4],
        ranks: vec![0; 4],
        owned_cell: vec![true; 4],
    })
    .unwrap();
    assert_eq!(t.vertex_sharers().unwrap().ranks, [0; 4]);
}

#[test]
fn set_vertex_sharers_rejects_bad_offsets() {
    let reject = |s: VertexSharers| {
        let mut t = two_triangles();
        let e = t.set_vertex_sharers(s).unwrap_err();
        assert_eq!(t.vertex_sharers(), None, "nothing stored on {e}");
        e
    };
    let mut s = sharers();
    s.offsets.pop();
    assert!(matches!(
        reject(s),
        GridError::LengthMismatch {
            expected: 5,
            found: 4
        }
    ));
    let mut s = sharers();
    s.offsets[0] = 1;
    assert!(matches!(reject(s), GridError::LengthMismatch { .. }));
    let mut s = sharers();
    s.offsets[2] = 0;
    assert!(matches!(reject(s), GridError::LengthMismatch { .. }));
    let mut s = sharers();
    s.ranks.push(1);
    assert!(matches!(
        reject(s),
        GridError::LengthMismatch {
            expected: 7,
            found: 8
        }
    ));
    let mut s = sharers();
    s.owned_cell.pop();
    assert!(matches!(
        reject(s),
        GridError::LengthMismatch {
            expected: 7,
            found: 6
        }
    ));
}

#[test]
fn set_vertex_sharers_rejects_unsorted_rows() {
    for ranks in [
        vec![1, 4, 1, 0, 1, 2, 1],  // row 1 descending
        vec![1, 1, 1, 0, 1, 2, 1],  // row 1 repeats a rank
        vec![1, 1, 4, -1, 1, 2, 1], // row 2 holds a negative rank
    ] {
        let mut t = two_triangles();
        let s = VertexSharers { ranks, ..sharers() };
        let e = t.set_vertex_sharers(s).unwrap_err();
        assert!(
            matches!(e, GridError::IndexOutOfRange { len: 4, .. }),
            "{e}"
        );
        let GridError::IndexOutOfRange { index, .. } = e else {
            unreachable!()
        };
        assert!(index == 1 || index == 2, "{index}");
    }
}

#[test]
fn set_vertex_sharers_rejects_rows_without_the_own_rank() {
    let mut t = two_triangles();
    // Vertex 2's row lacks rank 1, which every other row holds.
    let s = VertexSharers {
        offsets: vec![0, 1, 3, 5, 6],
        ranks: vec![1, 1, 4, 0, 2, 1],
        owned_cell: vec![true; 6],
    };
    assert!(matches!(
        t.set_vertex_sharers(s),
        Err(GridError::IndexOutOfRange { index: 2, len: 4 })
    ));
    // An empty row cannot hold it either.
    let s = VertexSharers {
        offsets: vec![0, 1, 1, 2, 3],
        ranks: vec![1, 1, 1],
        owned_cell: vec![true; 3],
    };
    assert!(matches!(
        t.set_vertex_sharers(s),
        Err(GridError::IndexOutOfRange { index: 1, len: 4 })
    ));
}
