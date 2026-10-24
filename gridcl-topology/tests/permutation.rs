//! The PermutationInfo kernel (M3 T4) against the conformance fixtures and
//! `gridcl_refcell::orientation_bits`, on both id paths and at every vector size.

use cubecl::prelude::Client;
use gridcl_device::testing::{record_plans, with_vector_size};
use gridcl_device::{DeviceBuffer, DeviceStrided, KernelKind, LaunchPlan, VectorSizeOverride};
use gridcl_dims::{CellType, D2, D3, TopoDim};
use gridcl_parallel::{IndexMap, Serial};
use gridcl_refcell::orientation_bits;
use gridcl_topology::{CellBlock, Topology, encode_origin_ids};
use proptest::prelude::*;

/// The bits of one block of `cell` with vertex rows `rows` over vertices with origin ids `ids`.
fn bits<TD: TopoDim>(client: &Client, cell: CellType, rows: &[u32], ids: &[i64]) -> Vec<u32> {
    let nv = gridcl_refcell::vertices(cell).len();
    let n = rows.len() / nv;
    let block = CellBlock::<TD>::new(
        TD::Cell::try_from(cell).unwrap(),
        IndexMap::new(&Serial, n).unwrap(),
        DeviceStrided {
            data: DeviceBuffer::from_host(client, rows).unwrap(),
            stride: nv as u32,
        },
        DeviceBuffer::from_host(
            client,
            &encode_origin_ids(&(0..n as i64).collect::<Vec<_>>()),
        )
        .unwrap(),
        DeviceBuffer::from_host(client, &vec![0u32; n]).unwrap(),
    );
    let t = Topology::new(
        client,
        vec![block],
        IndexMap::new(&Serial, ids.len()).unwrap(),
        ids,
    )
    .unwrap();
    t.permutation_info(0).unwrap().to_host().unwrap()
}

fn bits_any(client: &Client, cell: CellType, rows: &[u32], ids: &[i64]) -> Vec<u32> {
    match gridcl_refcell::tdim(cell) {
        2 => bits::<D2>(client, cell, rows, ids),
        _ => bits::<D3>(client, cell, rows, ids),
    }
}

/// Host reference: the bits of every row.
fn reference(cell: CellType, rows: &[u32], ids: &[i64]) -> Vec<u32> {
    let nv = gridcl_refcell::vertices(cell).len();
    rows.chunks_exact(nv)
        .map(|r| {
            let local: Vec<i64> = r.iter().map(|&v| ids[v as usize]).collect();
            orientation_bits(cell, &local)
        })
        .collect()
}

const CELLS: [CellType; 6] = [
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

/// Advances `p` to the next permutation in lexicographic order; false after the last one.
fn next_permutation(p: &mut [u32]) -> bool {
    let Some(i) = (1..p.len()).rev().find(|&i| p[i - 1] < p[i]) else {
        return false;
    };
    let j = (i..p.len()).rev().find(|&j| p[j] > p[i - 1]).unwrap();
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

/// Every permutation of 0..nv in lexicographic order, flattened.
fn permutation_rows(nv: usize) -> Vec<u32> {
    let mut p: Vec<u32> = (0..nv as u32).collect();
    let mut rows = p.clone();
    while next_permutation(&mut p) {
        rows.extend_from_slice(&p);
    }
    rows
}

#[test]
fn fixture_permutations_on_both_id_paths() {
    let client = gridcl_testing::test_client();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gridcl-refcell/tests/fixtures/orientation.json"
    );
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    for cell in CELLS {
        let f = &fixture[format!("{cell:?}")];
        let want: Vec<u32> = f["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| b.as_u64().unwrap() as u32)
            .collect();
        let nv = gridcl_refcell::vertices(cell).len();
        let rows = permutation_rows(nv);
        // Fast path (ids < 2^32) and the (hi, lo) path, with negative ids and ids above 2^32.
        for shift in [0i64, 5, -(1 << 40), 1 << 33] {
            let ids: Vec<i64> = (0..nv as i64).map(|i| i + shift).collect();
            let got = bits_any(&client, cell, &rows, &ids);
            assert_eq!(got, want, "{cell:?}, ids shifted by {shift}");
        }
    }
}

#[test]
fn hi_words_decide_before_lo_words() {
    // Ids whose lo words order opposite to the ids: only the (hi, lo) comparison is right.
    let client = gridcl_testing::test_client();
    let ids = [(1i64 << 32) + 5, 7, -1, (2i64 << 32) + 1];
    for cell in [CellType::Tetrahedron, CellType::Quadrilateral] {
        let rows: Vec<u32> = permutation_rows(4);
        assert_eq!(
            bits_any(&client, cell, &rows, &ids),
            reference(cell, &rows, &ids),
            "{cell:?}"
        );
    }
}

/// The vector sizes of the invariance tests: 1 and the larger powers of two the device offers.
fn vector_sizes(client: &Client) -> Vec<usize> {
    let widest = client.io_optimized_vector_sizes(4).next().unwrap_or(1);
    let mut sizes = vec![1];
    sizes.extend([2, 4, 8, 16].into_iter().filter(|&n| n <= widest));
    sizes
}

#[test]
fn every_vector_size_gives_the_same_bits() {
    let client = gridcl_testing::test_client();
    let mut rng = 0x1234_5678_9abc_def0u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    for cell in CELLS {
        let nv = gridcl_refcell::vertices(cell).len();
        let num_vertices = 50;
        let ids: Vec<i64> = (0..num_vertices).map(|_| next() as i64 >> 20).collect();
        for cells in [0usize, 1, 15, 16, 17, 100] {
            let rows: Vec<u32> = (0..cells)
                .flat_map(|_| {
                    // nv distinct vertices.
                    let mut row: Vec<u32> = Vec::new();
                    while row.len() < nv {
                        let v = (next() % num_vertices as u64) as u32;
                        if !row.contains(&v) {
                            row.push(v);
                        }
                    }
                    row
                })
                .collect();
            let want = reference(cell, &rows, &ids);
            for n in vector_sizes(&client) {
                let (got, plans): (Vec<u32>, Vec<(KernelKind, LaunchPlan)>) = record_plans(|| {
                    with_vector_size(VectorSizeOverride::Fixed(n), || {
                        bits_any(&client, cell, &rows, &ids)
                    })
                });
                let plan = plans
                    .iter()
                    .find(|(k, _)| *k == KernelKind::PermutationInfo);
                if cells > 0 {
                    assert_eq!(plan.unwrap().1.vector_size, n, "{cell:?}, {cells} cells");
                } else {
                    assert!(plan.is_none(), "an empty block launched");
                }
                assert_eq!(got, want, "{cell:?}, {cells} cells, vector size {n}");
            }
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 256, ..ProptestConfig::default() })]

    #[test]
    fn prop_bits_match_orientation_bits(
        cell in prop::sample::select(CELLS.to_vec()),
        seed in any::<u64>(),
        cells in 0usize..40,
        fast in any::<bool>(),
    ) {
        let client = gridcl_testing::test_client();
        let nv = gridcl_refcell::vertices(cell).len();
        let mut rng = seed | 1;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let num_vertices = 12usize;
        // Distinct ids: in [0, 2^32) on the fast path, any i64 otherwise.
        let mut ids: Vec<i64> = Vec::new();
        while ids.len() < num_vertices {
            let id = if fast { (next() >> 32) as i64 } else { next() as i64 };
            if !ids.contains(&id) {
                ids.push(id);
            }
        }
        let rows: Vec<u32> = (0..cells)
            .flat_map(|_| {
                let mut row: Vec<u32> = Vec::new();
                while row.len() < nv {
                    let v = (next() % num_vertices as u64) as u32;
                    if !row.contains(&v) {
                        row.push(v);
                    }
                }
                row
            })
            .collect();
        prop_assert_eq!(bits_any(&client, cell, &rows, &ids), reference(cell, &rows, &ids));
    }
}
