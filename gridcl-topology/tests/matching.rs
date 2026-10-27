//! `Topology::match_entities` (M4 T9, ADR-0022 item 4): every entity of every box matches
//! itself, in any key order; absent keys, entity types the dimension lacks and empty key sets.

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, SortKeys};
use gridcl_dims::{Cell2, Cell3, CellType, D1, D2, D3, TopoDim};
use gridcl_parallel::{IndexMap, Serial};
use gridcl_testing::HostMesh;
use gridcl_topology::{CellBlock, Topology, encode_origin_ids};

/// The topology of the degree-1 host mesh `h` (nodes are vertices).
fn topology<TD: TopoDim>(client: &Client, h: &HostMesh) -> Topology<TD>
where
    TD::Cell: TryFrom<CellType>,
{
    let blocks = h
        .blocks
        .iter()
        .map(|b| {
            assert_eq!(b.degree, 1);
            let n = b.num_cells();
            let nv = gridcl_refcell::vertices(b.cell).len();
            let ids: Vec<i64> = (0..n as i64).collect();
            let Ok(cell) = TD::Cell::try_from(b.cell) else {
                panic!("{:?} is not a cell of this dimension", b.cell)
            };
            CellBlock::<TD>::new(
                cell,
                IndexMap::new(&Serial, n).unwrap(),
                DeviceStrided {
                    data: DeviceBuffer::from_host(client, &b.cells).unwrap(),
                    stride: nv as u32,
                },
                DeviceBuffer::from_host(client, &encode_origin_ids(&ids)).unwrap(),
                DeviceBuffer::zeros(client, n).unwrap(),
            )
        })
        .collect();
    let ids: Vec<i64> = (0..h.num_nodes()).map(|i| h.node_id(i)).collect();
    Topology::new(
        client,
        blocks,
        IndexMap::new(&Serial, h.num_nodes()).unwrap(),
        &ids,
    )
    .unwrap()
}

/// SplitMix64 for shuffles.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

/// Every entity of every dimension, given as its sorted vertex row in shuffled order, matches
/// its own flat index; duplicated keys match too.
fn check_all<TD: TopoDim>(client: &Client, h: &HostMesh, seed: u64)
where
    TD::Cell: TryFrom<CellType>,
{
    let mut t = topology::<TD>(client, h);
    let mut rng = Rng(seed);
    for dim in 0..TD::N {
        t.create_entities(&Serial, dim).unwrap();
        for block in t.entity_blocks(dim).unwrap() {
            let words = block.vertices.stride as usize;
            let rows = block.vertices.data.to_host().unwrap();
            let n = rows.len() / words;
            assert_eq!(n, block.index_map.num_local());
            let mut order: Vec<usize> = (0..n).chain(0..n.min(3)).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, rng.next() as usize % (i + 1));
            }
            let mut keys = Vec::with_capacity(order.len() * words);
            for &e in &order {
                let mut row = rows[e * words..(e + 1) * words].to_vec();
                row.sort_unstable();
                keys.extend(row);
            }
            let keys = SortKeys::from_host(client, &keys, words).unwrap();
            let found = t.match_entities(dim, &keys).unwrap().to_host().unwrap();
            let want: Vec<u32> = order.iter().map(|&e| block.offset + e as u32).collect();
            assert_eq!(found, want, "{:?} entities of dimension {dim}", block.cell);
        }
    }
}

#[test]
fn matching_every_entity_of_every_box() {
    let client = gridcl_testing::test_client();
    check_all::<D1>(&client, &gridcl_testing::interval_box(5, 1), 1);
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        check_all::<D2>(&client, &gridcl_testing::square_box(cell, 3, 1), 2);
    }
    for cell in [
        Cell3::Tetrahedron,
        Cell3::Hexahedron,
        Cell3::Prism,
        Cell3::Pyramid,
    ] {
        check_all::<D3>(&client, &gridcl_testing::cube_box(cell, 2, 1), 3);
    }
    check_all::<D3>(&client, &gridcl_testing::mixed_box(2, 1), 4);
    // Shuffled origin ids change the canonical vertex order, not the matching.
    check_all::<D3>(
        &client,
        &gridcl_testing::cube_box(Cell3::Hexahedron, 2, 1).with_shuffled_ids(7),
        5,
    );
}

/// Keys of no entity, out-of-range and unsorted vertex indices give `u32::MAX`; so does an
/// entity type the dimension lacks; an empty key set gives an empty result.
#[test]
fn matching_absent_keys() {
    let client = gridcl_testing::test_client();
    let h = gridcl_testing::square_box(Cell2::Triangle, 2, 1);
    let mut t = topology::<D2>(&client, &h);
    t.create_entities(&Serial, 0).unwrap();
    t.create_entities(&Serial, 1).unwrap();
    let nv = h.num_nodes() as u32;
    let edges = t.entity_blocks(1).unwrap()[0]
        .vertices
        .data
        .to_host()
        .unwrap();
    let mut e0 = [edges[0], edges[1]];
    e0.sort_unstable();
    let keys = [
        e0[0],
        e0[1], // an edge
        e0[1],
        e0[0], // the same, unsorted
        0,
        nv, // a vertex beyond the mesh
        0,
        8, // opposite corners of the square: no edge
        u32::MAX,
        u32::MAX,
    ];
    let found = t
        .match_entities(1, &SortKeys::from_host(&client, &keys, 2).unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_ne!(found[0], u32::MAX);
    assert_eq!(found[1..], [u32::MAX; 4]);
    // Dimension 0: points match vertices.
    let points = [3, 0, nv - 1, nv, u32::MAX];
    let found = t
        .match_entities(0, &SortKeys::from_host(&client, &points, 1).unwrap())
        .unwrap()
        .to_host()
        .unwrap();
    assert_eq!(found, [3, 0, nv - 1, u32::MAX, u32::MAX]);
    // Three or four words: a triangle mesh has no such edges.
    for words in [3, 4] {
        let keys: Vec<u32> = (0..words as u32).collect();
        let found = t
            .match_entities(1, &SortKeys::from_host(&client, &keys, words).unwrap())
            .unwrap()
            .to_host()
            .unwrap();
        assert_eq!(found, [u32::MAX]);
    }
    let none = t
        .match_entities(1, &SortKeys::from_host(&client, &[], 2).unwrap())
        .unwrap();
    assert_eq!(none.len(), 0);
}

#[test]
fn matching_requires_entities() {
    let client = gridcl_testing::test_client();
    let t = topology::<D2>(
        &client,
        &gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1),
    );
    let keys = SortKeys::from_host(&client, &[0, 1], 2).unwrap();
    assert!(matches!(
        t.match_entities(1, &keys),
        Err(GridError::EntitiesNotCreated(1))
    ));
    assert!(matches!(
        t.match_entities(2, &keys),
        Err(GridError::EntitiesNotCreated(2))
    ));
}

/// Many keys, most absent: the result is the same on every run (bitwise determinism).
#[test]
fn matching_large_key_sets_are_deterministic() {
    let client = gridcl_testing::test_client();
    let h = gridcl_testing::cube_box(Cell3::Tetrahedron, 4, 1);
    let mut t = topology::<D3>(&client, &h);
    t.create_entities(&Serial, 2).unwrap();
    let mut rng = Rng(9);
    let nv = h.num_nodes() as u64;
    let keys: Vec<u32> = (0..3 * 20_000)
        .map(|_| (rng.next() % (nv + 2)) as u32)
        .collect::<Vec<_>>()
        .as_chunks::<3>()
        .0
        .iter()
        .flat_map(|&k| {
            let mut k = k;
            k.sort_unstable();
            k
        })
        .collect();
    let keys = SortKeys::from_host(&client, &keys, 3).unwrap();
    let a = t.match_entities(2, &keys).unwrap().to_host().unwrap();
    let b = t.match_entities(2, &keys).unwrap().to_host().unwrap();
    assert_eq!(a, b);
    let faces = t.entity_blocks(2).unwrap()[0].index_map.num_local() as u32;
    assert!(a.iter().all(|&f| f == u32::MAX || f < faces));
}
