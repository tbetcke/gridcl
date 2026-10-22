//! M3 acceptance: permutation info equals the conformance fixtures for every cell.
//!
//! For each of the six cells with D_TOP ≥ 2, a mesh of V shared vertices with one cell per
//! permutation of them, in the fixture's lexicographic order. Origin ids 0..V take the u32 fast
//! path; ids shifted by −2^40 and by 2^33 keep their order and take the (hi, lo) path. Interval
//! blocks have no permutation info.

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::{HostBlock, HostMesh};

/// One cell per permutation of the reference vertices of `cell`, P1, node ids `0..V + shift`.
fn permutation_mesh(cell: CellType, shift: i64) -> HostMesh {
    let verts = gridcl_refcell::vertices(cell);
    let td = gridcl_refcell::tdim(cell);
    let nodes: Vec<f64> = verts.iter().flat_map(|v| v.iter().copied()).collect();
    let cells: Vec<u32> = permutations(verts.len())
        .into_iter()
        .flat_map(|p| p.into_iter().map(|i| i as u32))
        .collect();
    HostMesh {
        gdim: td,
        nodes,
        node_ids: Some((0..verts.len() as i64).map(|i| i + shift).collect()),
        blocks: vec![HostBlock {
            cell,
            degree: 1,
            cells,
            cell_ids: None,
        }],
    }
}

fn bits<F: Real, TD: TopoDim + EmbeddedIn<TD>>(h: &HostMesh, client: &Client) -> Vec<u32> {
    let mesh = build::<F, TD, TD>(h, client);
    mesh.permutation_info(0)
        .expect("permutation info")
        .to_host()
        .expect("read")
}

fn check<F: Real>(client: &Client) {
    for (cell, nv, want) in orientation_fixture() {
        assert_eq!(nv, gridcl_refcell::vertices(cell).len());
        for shift in [0, -(1i64 << 40), 1i64 << 33] {
            let h = permutation_mesh(cell, shift);
            let got = match gridcl_refcell::tdim(cell) {
                2 => bits::<F, D2>(&h, client),
                _ => bits::<F, D3>(&h, client),
            };
            assert_eq!(got.len(), want.len(), "{cell:?}, shift {shift}");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert_eq!(g, w, "{cell:?}, shift {shift}, permutation {k}");
            }
        }
    }
    let interval = build::<F, D1, D1>(&BoxKind::Interval.host(3, 1), client);
    assert!(
        interval.permutation_info(0).is_none(),
        "interval blocks have no permutation info"
    );
}

#[test]
fn permutation_info_matches_fixture() {
    for_each_precision!(client(), check);
}
