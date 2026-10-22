//! Determinism ("Testing and validation"): two runs on the same device produce bitwise
//! identical device arrays: entities, connectivity, permutation info, factors and measure.

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::HostMesh;

/// Every topology array of `h` after creating all entities and connectivities.
fn topology<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(h: &HostMesh, client: &Client) -> Vec<Vec<u32>> {
    let mut mesh = build::<f32, TD, GD>(h, client);
    let tdim = TD::N;
    let mut out = Vec::new();
    for b in 0..mesh.cell_blocks().len() {
        if let Some(p) = mesh.permutation_info(b) {
            out.push(p.to_host().expect("read"));
        }
    }
    for d in 0..tdim {
        mesh.create_connectivity(d, tdim).expect("upward");
        for block in mesh.entity_blocks(d).expect("entities") {
            out.push(block.vertices.data.to_host().expect("read"));
        }
        let keys = [
            ConnKey {
                from: tdim as u8,
                to: d as u8,
            },
            ConnKey {
                from: d as u8,
                to: tdim as u8,
            },
        ];
        for key in keys {
            match mesh.connectivity(key) {
                Ok(Connectivity::Downward(blocks)) => {
                    out.extend(blocks.iter().map(|s| s.data.to_host().expect("read")));
                }
                Ok(Connectivity::Upward(csr)) => {
                    out.push(csr.offsets.to_host().expect("read"));
                    out.push(csr.indices.to_host().expect("read"));
                }
                Err(GridError::UnsupportedConnectivity(_)) if d == 0 && key.to == 0 => {}
                Err(e) => panic!("{key}: {e}"),
            }
        }
    }
    out
}

/// Every factor of `h` and its measure, as bit patterns.
fn factors<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    h: &HostMesh,
    client: &Client,
) -> Vec<Vec<u64>> {
    let mesh = build::<F, TD, GD>(h, client);
    let rule = QuadRule::for_grid(&mesh, 4).expect("rule");
    let bits = |v: Vec<Vec<f64>>| {
        v.into_iter()
            .map(|b| b.into_iter().map(f64::to_bits).collect())
            .collect::<Vec<Vec<u64>>>()
    };
    let mut out = Vec::new();
    out.extend(bits(flat_f64(&mesh.physical_points(&rule).expect("X"))));
    out.extend(bits(flat_f64(&mesh.jacobians(&rule).expect("J"))));
    out.extend(bits(flat_f64(&mesh.detj(&rule).expect("det J"))));
    out.extend(bits(flat_f64(
        &mesh.pseudo_inverse_jacobians(&rule).expect("K"),
    )));
    out.push(vec![mesh.measure(4).expect("measure").to_bits()]);
    out
}

fn twice<T: PartialEq>(what: &str, run: impl Fn() -> T) {
    assert!(run() == run(), "{what}: two runs differ");
}

fn check<F: Real>(client: &Client) {
    let mixed = BoxKind::Mixed.host(2, 2).perturbed(0.05);
    let sphere = gridcl_testing::sphere(1, 2);
    twice("mixed box factors", || factors::<F, D3, D3>(&mixed, client));
    twice("sphere factors", || factors::<F, D2, D3>(&sphere, client));
    twice("mixed box facet normals", || {
        let mesh = build::<F, D3, D3>(&mixed, client);
        let rule = FacetQuadRule::for_grid(&mesh, 2).expect("rule");
        flat_f64(&mesh.facet_normals(&rule).expect("facet normals"))
            .into_iter()
            .map(|b| b.into_iter().map(f64::to_bits).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    });
    twice("sphere normals", || {
        let mesh = build::<F, D2, D3>(&sphere, client);
        let rule = QuadRule::for_grid(&mesh, 2).expect("rule");
        flat_f64(&mesh.normals(&rule).expect("normals"))
            .into_iter()
            .map(|b| b.into_iter().map(f64::to_bits).collect::<Vec<_>>())
            .collect::<Vec<_>>()
    });
}

#[test]
fn entities_connectivity_and_permutation_info() {
    let client = client();
    let mixed = BoxKind::Mixed.host(2, 1).with_shuffled_ids(7);
    twice("mixed box topology", || topology::<D3, D3>(&mixed, &client));
    let sphere = gridcl_testing::sphere(2, 1);
    twice("sphere topology", || topology::<D2, D3>(&sphere, &client));
}

#[test]
fn factors_and_measure() {
    for_each_precision!(client(), check);
}
