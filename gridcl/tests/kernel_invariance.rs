//! Vector-size invariance of the mesh kernels ("Testing and validation"; docs/plans/M3.md,
//! question 14): TopologyKeys, PermutationInfo and EVector give bitwise identical results at
//! vector size 1 and at every larger power of two up to 16 that the device offers, for 0, 1, 15,
//! 16 and 17 cells of every cell type and for 10^6 + 3 P1 triangles.

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_device::KernelKind;
use gridcl_testing::HostMesh;

/// The test meshes: every box at p = 1 with shuffled origin ids, cut to each invariance count,
/// and the large triangle strip.
fn meshes() -> Vec<(String, BoxKind, HostMesh)> {
    use BoxKind::*;
    let mut out = Vec::new();
    for (kind, n) in [
        (Interval, 17),
        (Triangle, 3),
        (Quadrilateral, 5),
        (Tetrahedron, 2),
        (Hexahedron, 3),
        (Prism, 3),
        (Pyramid, 2),
        (Mixed, 2),
    ] {
        let full = kind.host(n, 1).with_shuffled_ids(0x5eed ^ n as u64);
        for cells in INVARIANCE_CELLS {
            out.push((
                format!("{kind:?} box, {cells} cells"),
                kind,
                full.first_cells(cells),
            ));
        }
    }
    out.push((
        format!("triangle strip, {LARGE_CELLS} cells"),
        Triangle,
        gridcl_testing::triangle_strip(LARGE_CELLS),
    ));
    out
}

/// Runs `run` at every vector size and asserts the results equal the size-1 result bitwise.
fn sweep<T: PartialEq + core::fmt::Debug>(
    client: &Client,
    elem_bytes: usize,
    kind: KernelKind,
    launches: bool,
    what: &str,
    run: impl Fn() -> T,
) {
    let mut first: Option<T> = None;
    for n in vector_sizes(client, elem_bytes, 16) {
        let got = at_vector_size(n, kind, launches, &run);
        match &first {
            None => first = Some(got),
            Some(want) => assert!(
                got == *want,
                "{what}: {kind:?} at vector size {n} differs from size 1"
            ),
        }
    }
}

/// Entity vertex lists and downward connectivity of every dimension below tdim.
fn entities<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(h: &HostMesh, client: &Client) -> Vec<Vec<u32>> {
    let mut mesh = build::<f32, TD, GD>(h, client);
    let mut out = Vec::new();
    for d in 1..TD::N {
        mesh.create_entities(d).expect("create_entities");
        for block in mesh.entity_blocks(d).expect("entity blocks") {
            out.push(block.vertices.data.to_host().expect("read"));
        }
        match mesh
            .connectivity(ConnKey {
                from: TD::N as u8,
                to: d as u8,
            })
            .expect("downward")
        {
            Connectivity::Downward(blocks) => {
                out.extend(blocks.iter().map(|s| s.data.to_host().expect("read")));
            }
            Connectivity::Upward(_) => panic!("(tdim, d) is downward"),
        }
    }
    out
}

fn permutation_info<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    h: &HostMesh,
    client: &Client,
) -> Vec<Vec<u32>> {
    let mesh = build::<f32, TD, GD>(h, client);
    (0..mesh.cell_blocks().len())
        .map(|b| {
            mesh.permutation_info(b)
                .map(|p| p.to_host().expect("read"))
                .unwrap_or_default()
        })
        .collect()
}

fn evectors<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    mesh: &Mesh<F, TD, GD>,
) -> Vec<Vec<u64>> {
    let fresh = mesh.try_clone().expect("try_clone"); // an empty E-vector cache
    (0..fresh.cell_blocks().len())
        .flat_map(|b| {
            let e = fresh.evector(b).expect("evector");
            e.chunks()
                .iter()
                .map(|c| {
                    c.data
                        .to_host()
                        .expect("read")
                        .into_iter()
                        .map(|x| x.widen().to_bits())
                        .collect()
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn topology_keys_sizes() {
    let client = client();
    for (what, kind, h) in meshes() {
        let launches = h.num_cells() > 0 && kind.tdim() > 1;
        match kind.tdim() {
            1 => sweep(&client, 4, KernelKind::TopologyKeys, false, &what, || {
                entities::<D1, D1>(&h, &client)
            }),
            2 => sweep(
                &client,
                4,
                KernelKind::TopologyKeys,
                launches,
                &what,
                || entities::<D2, D2>(&h, &client),
            ),
            _ => sweep(
                &client,
                4,
                KernelKind::TopologyKeys,
                launches,
                &what,
                || entities::<D3, D3>(&h, &client),
            ),
        }
    }
}

#[test]
fn permutation_info_sizes() {
    let client = client();
    for (what, kind, h) in meshes() {
        let launches = h.num_cells() > 0 && kind.tdim() > 1;
        match kind.tdim() {
            1 => sweep(
                &client,
                4,
                KernelKind::PermutationInfo,
                false,
                &what,
                || permutation_info::<D1, D1>(&h, &client),
            ),
            2 => sweep(
                &client,
                4,
                KernelKind::PermutationInfo,
                launches,
                &what,
                || permutation_info::<D2, D2>(&h, &client),
            ),
            _ => sweep(
                &client,
                4,
                KernelKind::PermutationInfo,
                launches,
                &what,
                || permutation_info::<D3, D3>(&h, &client),
            ),
        }
    }
}

fn evector_cases<F: Real>(client: &Client) {
    let bytes = size_of::<F>();
    for (what, kind, h) in meshes() {
        let what = format!("{what}, {}", name::<F>());
        let launches = h.num_cells() > 0;
        match kind.tdim() {
            1 => {
                let mesh = build::<F, D1, D1>(&h, client);
                sweep(client, bytes, KernelKind::EVector, launches, &what, || {
                    evectors(&mesh)
                });
            }
            2 => {
                let mesh = build::<F, D2, D2>(&h, client);
                sweep(client, bytes, KernelKind::EVector, launches, &what, || {
                    evectors(&mesh)
                });
            }
            _ => {
                let mesh = build::<F, D3, D3>(&h, client);
                sweep(client, bytes, KernelKind::EVector, launches, &what, || {
                    evectors(&mesh)
                });
            }
        }
    }
}

#[test]
fn evector_sizes() {
    for_each_precision!(client(), evector_cases);
}
