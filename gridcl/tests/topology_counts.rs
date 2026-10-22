//! M3 acceptance: closed-form entity counts and Euler characteristic 1 for boxes of every cell
//! type and the mixed box, at three refinements.

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;

const REFINEMENTS: [usize; 3] = [1, 2, 4];

/// The entity counts of `mesh`: per dimension (cells last) and per (dimension, type).
fn counts<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(mesh: &mut Mesh<F, TD, GD>) -> Counts {
    let tdim = TD::N;
    let mut per_dim = Vec::new();
    let mut per_type = Vec::new();
    for d in 0..tdim {
        mesh.create_entities(d).expect("create_entities");
        let blocks = mesh.entity_blocks(d).expect("entity_blocks");
        per_dim.push(blocks.iter().map(|b| b.index_map.num_local()).sum());
        if d > 0 {
            for b in blocks {
                per_type.push((d, b.cell, b.index_map.num_local()));
            }
        }
    }
    per_dim.push(
        mesh.cell_blocks()
            .iter()
            .map(|b| b.index_map.num_local())
            .sum(),
    );
    assert!(matches!(mesh.entity_blocks(tdim), Err(GridError::EntitiesNotCreated(d)) if d == tdim));
    Counts { per_dim, per_type }
}

fn check<F: Real>(client: &Client) {
    for kind in BoxKind::ALL {
        for n in REFINEMENTS {
            let h = kind.host(n, 1);
            let got = match kind.tdim() {
                1 => counts(&mut build::<F, D1, D1>(&h, client)),
                2 => counts(&mut build::<F, D2, D2>(&h, client)),
                _ => counts(&mut build::<F, D3, D3>(&h, client)),
            };
            let want = box_counts(kind, n);
            assert_eq!(
                got.per_dim, want.per_dim,
                "{kind:?} box, n = {n}: entities per dimension"
            );
            let mut got_types = got.per_type.clone();
            let mut want_types = want.per_type.clone();
            got_types.sort();
            want_types.sort();
            assert_eq!(
                got_types, want_types,
                "{kind:?} box, n = {n}: entities per type"
            );
            let euler: i64 = got
                .per_dim
                .iter()
                .enumerate()
                .map(|(d, &c)| if d % 2 == 0 { c as i64 } else { -(c as i64) })
                .sum();
            assert_eq!(euler, 1, "{kind:?} box, n = {n}: Euler characteristic");
        }
    }
}

#[test]
fn entity_counts_and_euler_characteristic() {
    for_each_precision!(client(), check);
}
