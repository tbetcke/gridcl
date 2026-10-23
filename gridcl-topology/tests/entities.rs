//! Entity creation (M3 T8): entity vertex lists in canonical orientation and the downward
//! connectivity, against a host reference built from sorted vertex lists and
//! `gridcl_refcell::canonical_order`, on both id paths; invariance under shuffled origin ids and
//! node order.

use std::collections::BTreeMap;

use gridcl::prelude::*;
use gridcl_dims::CellType;
use gridcl_refcell::{canonical_order, sub_entities, sub_entity_types};
use gridcl_testing::HostMesh;
use gridcl_topology::decode_origin_ids;
use proptest::prelude::*;

/// Entities of one dimension as canonical origin-id lists, per entity type.
type Lists = BTreeMap<CellType, Vec<Vec<i64>>>;

/// The canonical origin-id list of sub-entity `i` of dimension `dim` of host cell `c` of
/// block `b`.
fn host_entity(h: &HostMesh, b: usize, c: usize, dim: usize, i: usize) -> (CellType, Vec<i64>) {
    let block = &h.blocks[b];
    let k = block.nodes_per_cell();
    let list = sub_entities(block.cell, dim)[i];
    let ty = sub_entity_types(block.cell, dim)[i];
    let ids: Vec<i64> = list
        .iter()
        .map(|&v| h.node_id(block.cells[c * k + v] as usize))
        .collect();
    let slots = canonical_order(ty, &ids);
    (ty, slots.iter().map(|&s| ids[s]).collect())
}

/// Host reference: the distinct canonical lists of dimension `dim`, sorted, per type.
fn host_lists(h: &HostMesh, dim: usize) -> Lists {
    let mut out = Lists::new();
    for (b, block) in h.blocks.iter().enumerate() {
        for c in 0..block.num_cells() {
            for i in 0..sub_entities(block.cell, dim).len() {
                let (ty, list) = host_entity(h, b, c, dim, i);
                out.entry(ty).or_default().push(list);
            }
        }
    }
    for lists in out.values_mut() {
        lists.sort();
        lists.dedup();
    }
    out
}

/// Device entities of dimension `dim` as origin-id lists, per type in block order, and the
/// downward connectivity checked cell by cell against the host.
fn device_lists<TD: TopoDim + EmbeddedIn<TD>>(h: &HostMesh, dim: usize) -> Lists {
    let client = gridcl_testing::test_client();
    let mut mesh = h.build::<f32, TD, TD>(&client).expect("build");
    mesh.create_entities(dim).expect("create_entities");
    let vids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
    let blocks = mesh.entity_blocks(dim).expect("entity blocks");
    let mut flat: Vec<(CellType, Vec<i64>)> = Vec::new();
    let mut out = Lists::new();
    for block in blocks {
        assert_eq!(block.offset as usize, flat.len(), "flat offsets");
        let stride = block.vertices.stride as usize;
        let verts = block.vertices.data.to_host().unwrap();
        assert_eq!(verts.len(), block.index_map.num_local() * stride);
        for row in verts.chunks_exact(stride) {
            let list: Vec<i64> = row.iter().map(|&v| vids[v as usize]).collect();
            flat.push((block.cell, list.clone()));
            out.entry(block.cell).or_default().push(list);
        }
    }
    let key = ConnKey {
        from: TD::N as u8,
        to: dim as u8,
    };
    let Connectivity::Downward(down) = mesh.connectivity(key).expect("downward") else {
        panic!("(tdim, d) is downward");
    };
    for (b, strided) in down.iter().enumerate() {
        let cells = strided.data.to_host().unwrap();
        let k = strided.stride as usize;
        assert_eq!(cells.len(), h.blocks[b].num_cells() * k);
        for (c, row) in cells.chunks_exact(k).enumerate() {
            for (i, &e) in row.iter().enumerate() {
                assert_eq!(
                    flat[e as usize],
                    host_entity(h, b, c, dim, i),
                    "block {b}, cell {c}, sub-entity {i}"
                );
            }
        }
    }
    for lists in out.values_mut() {
        lists.sort();
    }
    out
}

fn check(h: &HostMesh, what: &str) {
    let tdim = gridcl_refcell::tdim(h.blocks[0].cell);
    for dim in 1..tdim {
        let got = match tdim {
            2 => device_lists::<D2>(h, dim),
            _ => device_lists::<D3>(h, dim),
        };
        // Each entity once (no duplicates), and exactly the host's entities.
        assert_eq!(got, host_lists(h, dim), "{what}, dimension {dim}");
    }
}

#[test]
fn entities_match_the_host_reference() {
    use gridcl_testing::*;
    let meshes = [
        ("mixed box", mixed_box(2, 1)),
        ("prism box", cube_box(Cell3::Prism, 2, 1)),
        ("pyramid box", cube_box(Cell3::Pyramid, 1, 2)),
        ("triangle box", square_box(Cell2::Triangle, 3, 1)),
        ("quadrilateral box", square_box(Cell2::Quadrilateral, 2, 3)),
    ];
    for (what, h) in meshes {
        // Ids in [0, 2^32): the fast path; shuffled ids in [-2^40, 2^40]: the (hi, lo) path.
        check(&h, what);
        check(&h.with_shuffled_ids(17), &format!("{what}, shuffled ids"));
        check(&h.first_cells(17), &format!("{what}, 17 cells"));
    }
}

#[test]
fn empty_blocks_have_no_entities() {
    let h = gridcl_testing::mixed_box(2, 1).first_cells(0);
    let client = gridcl_testing::test_client();
    let mut mesh = h.build::<f32, D3, D3>(&client).unwrap();
    mesh.create_entities(2).unwrap();
    assert!(
        mesh.entity_blocks(2)
            .unwrap()
            .iter()
            .all(|b| b.index_map.num_local() == 0)
    );
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 12, ..ProptestConfig::default() })]

    #[test]
    fn prop_entities_invariant_under_shuffles(ids in any::<u64>(), nodes in any::<u64>(), n in 1usize..3) {
        let h = gridcl_testing::mixed_box(n, 1).with_shuffled_ids(ids);
        let shuffled = h.with_shuffled_nodes(nodes);
        for dim in 1..3 {
            let want = device_lists::<D3>(&h, dim);
            prop_assert_eq!(&want, &host_lists(&h, dim));
            // Node order does not change the entities as origin ids.
            prop_assert_eq!(device_lists::<D3>(&shuffled, dim), want);
        }
    }
}
