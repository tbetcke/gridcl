//! M3 review (Major 3): `Topology::new` checks the lengths of every block's `origin_ids` and
//! `flags`, and a topology whose vertex rows hold indices past the local vertices reads no
//! memory out of bounds: every kernel clamps such an index to the last vertex, so the results
//! equal those of the clamped rows and are deterministic.

use cubecl::prelude::Client;
use gridcl_device::{ConnKey, DeviceBuffer, DeviceStrided, GridError};
use gridcl_dims::{Cell2, Cell3, D2, D3, TopoDim};
use gridcl_parallel::{IndexMap, Serial};
use gridcl_topology::{CellBlock, Connectivity, Topology, encode_origin_ids};

/// A block of `cell` with vertex rows `rows`, `origin_words` origin-id words and `flags` flag
/// words.
fn block<TD: TopoDim>(
    client: &Client,
    cell: TD::Cell,
    nv: usize,
    rows: &[u32],
    origin_words: usize,
    flags: usize,
) -> CellBlock<TD> {
    let n = rows.len() / nv;
    CellBlock::new(
        cell,
        IndexMap::new(&Serial, n).unwrap(),
        DeviceStrided {
            data: DeviceBuffer::from_host(client, rows).unwrap(),
            stride: nv as u32,
        },
        DeviceBuffer::from_host(client, &vec![0x8000_0000u32; origin_words]).unwrap(),
        DeviceBuffer::from_host(client, &vec![0u32; flags]).unwrap(),
    )
}

/// A topology of one block of `cell` over `num_vertices` vertices with origin ids
/// `100 − 3 · v`, so the id order differs from the index order.
fn topology<TD: TopoDim>(
    client: &Client,
    cell: TD::Cell,
    nv: usize,
    rows: &[u32],
    num_vertices: usize,
) -> Topology<TD> {
    let n = rows.len() / nv;
    let ids: Vec<i64> = (0..num_vertices as i64).map(|v| 100 - 3 * v).collect();
    let mut b = block::<TD>(client, cell, nv, rows, 0, n);
    b.origin_ids = DeviceBuffer::from_host(
        client,
        &encode_origin_ids(&(0..n as i64).collect::<Vec<_>>()),
    )
    .unwrap();
    Topology::new(
        client,
        vec![b],
        IndexMap::new(&Serial, num_vertices).unwrap(),
        &ids,
    )
    .unwrap()
}

#[test]
fn new_rejects_mismatched_flags_and_origin_ids() {
    let client = gridcl_testing::test_client();
    let rows = [0u32, 1, 2, 3, 2, 1];
    let new = |origin_words, flags| {
        Topology::<D2>::new(
            &client,
            vec![block::<D2>(
                &client,
                Cell2::Triangle,
                3,
                &rows,
                origin_words,
                flags,
            )],
            IndexMap::new(&Serial, 4).unwrap(),
            &[0, 1, 2, 3],
        )
    };
    assert!(new(4, 2).is_ok());
    for (origin_words, flags, expected, found) in [
        (3, 2, 4, 3),
        (2, 2, 4, 2),
        (6, 2, 4, 6),
        (4, 1, 2, 1),
        (4, 3, 2, 3),
        (4, 0, 2, 0),
    ] {
        match new(origin_words, flags) {
            Err(GridError::LengthMismatch {
                expected: e,
                found: f,
            }) => assert_eq!(
                (e, f),
                (expected, found),
                "{origin_words} origin words, {flags} flags"
            ),
            Err(e) => panic!("{origin_words} origin words, {flags} flags: {e}"),
            Ok(_) => panic!("{origin_words} origin words, {flags} flags: accepted"),
        }
    }
}

/// Everything the topology computes: permutation info, the entities of every dimension below
/// tdim, the downward and the upward connectivity.
fn snapshot<TD: TopoDim>(mut t: Topology<TD>) -> Vec<Vec<u32>> {
    let tdim = TD::N;
    let mut out = Vec::new();
    if let Some(bits) = t.permutation_info(0) {
        out.push(bits.to_host().unwrap());
    }
    for dim in 0..tdim {
        t.create_connectivity(&Serial, dim, tdim).unwrap();
        for b in t.entity_blocks(dim).unwrap() {
            out.push(b.vertices.data.to_host().unwrap());
        }
        let up = ConnKey {
            from: dim as u8,
            to: tdim as u8,
        };
        let Connectivity::Upward(csr) = t.connectivity(up).unwrap() else {
            panic!("upward");
        };
        out.push(csr.offsets.to_host().unwrap());
        out.push(csr.indices.to_host().unwrap());
        if dim > 0 {
            let down = ConnKey {
                from: tdim as u8,
                to: dim as u8,
            };
            let Connectivity::Downward(rows) = t.connectivity(down).unwrap() else {
                panic!("downward");
            };
            out.extend(rows.iter().map(|r| r.data.to_host().unwrap()));
        }
    }
    out
}

/// Rows of `n` cells of `nv` vertices out of `num_vertices`, with the last vertex of every
/// third cell replaced by an index past them (some u32::MAX), and the same rows clamped to
/// `num_vertices − 1`.
fn rows_out_of_range(n: usize, nv: usize, num_vertices: usize) -> (Vec<u32>, Vec<u32>) {
    let mut bad = Vec::new();
    for c in 0..n {
        for j in 0..nv {
            let v = ((c + 2 * j) % num_vertices) as u32;
            bad.push(match (c % 3, j + 1 == nv) {
                (0, true) => num_vertices as u32 + c as u32,
                (1, true) if c % 2 == 0 => u32::MAX,
                _ => v,
            });
        }
    }
    let last = num_vertices as u32 - 1;
    let clamped = bad.iter().map(|&v| v.min(last)).collect();
    (bad, clamped)
}

fn check_clamped<TD: TopoDim>(cell: TD::Cell, nv: usize, n: usize, num_vertices: usize) {
    let client = gridcl_testing::test_client();
    let (bad, clamped) = rows_out_of_range(n, nv, num_vertices);
    assert!(bad.iter().any(|&v| v as usize >= num_vertices));
    let got = snapshot(topology::<TD>(&client, cell, nv, &bad, num_vertices));
    let again = snapshot(topology::<TD>(&client, cell, nv, &bad, num_vertices));
    let want = snapshot(topology::<TD>(&client, cell, nv, &clamped, num_vertices));
    assert_eq!(got, again, "{cell:?}: deterministic");
    assert_eq!(got, want, "{cell:?}: as the clamped rows");
}

#[test]
fn out_of_range_vertices_read_the_last_vertex() {
    check_clamped::<D2>(Cell2::Quadrilateral, 4, 37, 11);
    check_clamped::<D3>(Cell3::Tetrahedron, 4, 23, 10);
}

#[test]
fn cells_without_vertices_read_no_vertex_memory() {
    // Every vertex index is out of range: the kernels read the zeroed padding of the empty
    // origin-id buffer, deterministically.
    let client = gridcl_testing::test_client();
    let rows = [5u32, 6, 7, 7, 6, 9];
    let run = || snapshot(topology::<D2>(&client, Cell2::Triangle, 3, &rows, 0));
    assert_eq!(run(), run());
}
