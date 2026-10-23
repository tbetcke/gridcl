//! `MeshBuilder` and `Mesh` (M3 T6): input checks, numbering (pipeline stage 8), the origin and
//! `read_node` round trip, `try_clone`, and `boundary` until M9.

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_topology::decode_origin_ids;

const TRI: CoordinateElement<D2> = CoordinateElement {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
const QUAD: CoordinateElement<D2> = CoordinateElement {
    cell: Cell2::Quadrilateral,
    degree: 1,
    family: NodeFamily::Equispaced,
};
const TRI2: CoordinateElement<D2> = CoordinateElement {
    cell: Cell2::Triangle,
    degree: 2,
    family: NodeFamily::Equispaced,
};

/// Five nodes: the unit square's corners and an unused node.
const SQUARE: [f64; 10] = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 9.0, 9.0];

fn client() -> Client {
    gridcl_testing::test_client()
}

fn err<T>(r: Result<T>) -> GridError {
    match r {
        Ok(_) => panic!("expected an error"),
        Err(e) => e,
    }
}

#[test]
fn input_lengths_and_ranges() {
    let c = client();
    let b = || MeshBuilder::<f32, D2, D2>::new(&c);
    assert!(matches!(
        err(b().nodes(&SQUARE[..9], None).build()),
        GridError::LengthMismatch {
            expected: 10,
            found: 9
        }
    ));
    assert!(matches!(
        err(b().nodes(&SQUARE, Some(&[1, 2, 3])).build()),
        GridError::LengthMismatch {
            expected: 5,
            found: 3
        }
    ));
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 2, 3], None)
            .build()),
        GridError::LengthMismatch {
            expected: 6,
            found: 4
        }
    ));
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 5], None)
            .build()),
        GridError::IndexOutOfRange { index: 5, len: 5 }
    ));
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 2], Some(&[1, 2]))
            .build()),
        GridError::LengthMismatch {
            expected: 1,
            found: 2
        }
    ));
}

#[test]
fn duplicate_ids_roles_and_blocks() {
    let c = client();
    let b = || MeshBuilder::<f32, D2, D2>::new(&c);
    assert!(matches!(
        err(b().nodes(&SQUARE, Some(&[1, 2, 3, 2, 5])).build()),
        GridError::DuplicateOriginId(2)
    ));
    // Cell ids are unique over all blocks.
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 2], Some(&[7]))
            .cell_block(QUAD, &[0, 1, 2, 3], Some(&[7]))
            .build()),
        GridError::DuplicateOriginId(7)
    ));
    // Default cell ids count over all blocks, so two blocks do not collide.
    assert!(
        b().nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 2], None)
            .cell_block(QUAD, &[0, 1, 2, 3], None)
            .build()
            .is_ok()
    );
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(TRI, &[0, 1, 2], None)
            .cell_block(TRI, &[1, 3, 2], None)
            .build()),
        GridError::DuplicateCellBlock("Triangle")
    ));
    // Node 3 is a vertex of the first triangle and an edge node of the P2 triangle.
    let coords: Vec<f64> = (0..10).map(f64::from).collect();
    let e = err(MeshBuilder::<f32, D2, D2>::new(&c)
        .nodes(&coords, None)
        .cell_block(TRI, &[3, 1, 2], None)
        .cell_block(
            CoordinateElement {
                cell: Cell2::Quadrilateral,
                ..TRI2
            },
            &[0, 1, 2, 4, 3, 0, 1, 2, 4],
            None,
        )
        .build());
    assert!(matches!(e, GridError::NodeRoleMismatch(_)), "{e}");
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(
                CoordinateElement {
                    family: NodeFamily::Gll,
                    ..TRI
                },
                &[0, 1, 2],
                None
            )
            .build()),
        GridError::UnsupportedNodeFamily {
            cell: "Triangle",
            family: "Gll"
        }
    ));
    assert!(matches!(
        err(b()
            .nodes(&SQUARE, None)
            .cell_block(CoordinateElement { degree: 0, ..TRI }, &[0], None)
            .build()),
        GridError::UnsupportedDegree {
            cell: "Triangle",
            degree: 0
        }
    ));
}

#[test]
fn f64_needs_device_support() {
    let c = client();
    let r = MeshBuilder::<f64, D2, D2>::new(&c)
        .nodes(&SQUARE, None)
        .cell_block(TRI, &[0, 1, 2], None)
        .build();
    if gridcl_testing::supports::<f64>(&c) {
        assert!(r.is_ok());
    } else {
        assert!(matches!(
            err(r),
            GridError::UnsupportedPrecision { scalar: "f64", .. }
        ));
    }
}

/// A P2 triangle and a P1 quadrilateral in 3D, nodes given in scrambled order with ids.
fn mixed<F: Real>(c: &Client) -> (Mesh<F, D2, D3>, Vec<f64>, Vec<i64>) {
    // Nodes 0..=6 of a P2 triangle (vertices 4, 0, 6; edge nodes 1, 5, 2) and a quadrilateral
    // (vertices 0, 3, 6, 7); node 8 is unused.
    let coords: Vec<f64> = (0..9)
        .flat_map(|i| {
            let x = f64::from(i);
            [1e3 + 0.5 * x, -2.0 + x * x / 7.0, 0.25 * x - 1.0]
        })
        .collect();
    let ids: Vec<i64> = (0..9).map(|i| 100 - 3 * i).collect();
    let tri = [4u32, 0, 6, 1, 5, 2];
    let quad = [0u32, 3, 6, 7];
    let mesh = MeshBuilder::<F, D2, D3>::new(c)
        .nodes(&coords, Some(&ids))
        .cell_block(TRI2, &tri, Some(&[-5]))
        .cell_block(QUAD, &quad, Some(&[1 << 40]))
        .build()
        .unwrap();
    (mesh, coords, ids)
}

fn numbering_and_round_trip<F: Real>(c: &Client) {
    let (mesh, coords, ids) = mixed::<F>(c);
    // Nodes in order of first touch: 4 0 6 1 5 2, then 3 7; node 8 is dropped.
    let order = [4usize, 0, 6, 1, 5, 2, 3, 7];
    assert_eq!(mesh.node_index_map().num_local(), order.len());
    assert_eq!(
        mesh.geometry_block(0).dofmap.data.to_host().unwrap(),
        [0, 1, 2, 3, 4, 5]
    );
    assert_eq!(
        mesh.geometry_block(1).dofmap.data.to_host().unwrap(),
        [1, 6, 2, 7]
    );
    // Vertices in order of first touch: 4 0 6, then 3 7.
    let vorder = [4usize, 0, 6, 3, 7];
    assert_eq!(mesh.vertex_index_map().num_local(), vorder.len());
    assert_eq!(
        mesh.cell_blocks()[0].vertices.data.to_host().unwrap(),
        [0, 1, 2]
    );
    assert_eq!(
        mesh.cell_blocks()[1].vertices.data.to_host().unwrap(),
        [1, 3, 2, 4]
    );
    let vids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
    assert_eq!(vids, vorder.map(|n| ids[n]));
    assert_eq!(
        decode_origin_ids(&mesh.cell_blocks()[0].origin_ids.to_host().unwrap()),
        [-5]
    );
    assert_eq!(
        decode_origin_ids(&mesh.cell_blocks()[1].origin_ids.to_host().unwrap()),
        [1 << 40]
    );
    assert_eq!(mesh.cell_blocks()[1].flags.to_host().unwrap(), [0]);
    // The origin is the centre of the used nodes' bounding box.
    let used: Vec<&[f64]> = order.iter().map(|&n| &coords[3 * n..3 * n + 3]).collect();
    let origin = mesh.origin();
    for g in 0..3 {
        let lo = used.iter().map(|x| x[g]).fold(f64::INFINITY, f64::min);
        let hi = used.iter().map(|x| x[g]).fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(origin[g], 0.5 * (lo + hi));
    }
    // read_node adds the origin back.
    let tol = if size_of::<F>() == 4 { 1e-6 } else { 1e-15 };
    for (i, &n) in order.iter().enumerate() {
        let x = mesh.read_node(i).unwrap();
        for g in 0..3 {
            let want = coords[3 * n + g];
            let got = x[g].widen();
            assert!(
                (got - want).abs() <= tol * want.abs().max(1.0),
                "node {i}, {g}: {got} vs {want}"
            );
        }
    }
    assert!(matches!(
        mesh.read_node(order.len()),
        Err(GridError::IndexOutOfRange { index: 8, len: 8 })
    ));
}

#[test]
fn numbering_origin_and_read_node() {
    let c = client();
    numbering_and_round_trip::<f32>(&c);
    if gridcl_testing::supports::<f64>(&c) {
        numbering_and_round_trip::<f64>(&c);
    }
}

#[test]
fn try_clone_gives_an_equal_mesh() {
    let c = client();
    let (mesh, _, _) = mixed::<f32>(&c);
    let copy = mesh.try_clone().unwrap();
    assert_eq!(copy.origin(), mesh.origin());
    assert_eq!(
        copy.nodes().to_host().unwrap(),
        mesh.nodes().to_host().unwrap()
    );
    for b in 0..2 {
        let (a, z) = (&mesh.cell_blocks()[b], &copy.cell_blocks()[b]);
        assert_eq!(a.cell, z.cell);
        assert_eq!(
            a.vertices.data.to_host().unwrap(),
            z.vertices.data.to_host().unwrap()
        );
        assert_eq!(
            a.origin_ids.to_host().unwrap(),
            z.origin_ids.to_host().unwrap()
        );
        assert_eq!(
            mesh.geometry_block(b).dofmap.data.to_host().unwrap(),
            copy.geometry_block(b).dofmap.data.to_host().unwrap()
        );
        assert_eq!(
            mesh.permutation_info(b).unwrap().to_host().unwrap(),
            copy.permutation_info(b).unwrap().to_host().unwrap()
        );
    }
    assert_eq!(
        mesh.vertex_origin_ids().to_host().unwrap(),
        copy.vertex_origin_ids().to_host().unwrap()
    );
}

#[test]
fn empty_mesh_and_dimensions() {
    let c = client();
    let mesh = MeshBuilder::<f32, D2, D3>::new(&c)
        .cell_block(TRI, &[], None)
        .build()
        .unwrap();
    assert_eq!((mesh.tdim(), mesh.gdim()), (2, 3));
    assert_eq!((SurfaceMesh::<f32>::TDIM, SurfaceMesh::<f32>::GDIM), (2, 3));
    assert_eq!(mesh.origin(), [0.0; 3]);
    assert_eq!(mesh.cell_blocks()[0].index_map.num_local(), 0);
    assert_eq!(mesh.permutation_info(0).unwrap().len(), 0);
}

#[test]
fn boundary_is_not_yet_implemented() {
    let c = client();
    let (mesh, _, _) = mixed::<f32>(&c);
    assert!(matches!(
        mesh.boundary(),
        Err(GridError::NotYetImplemented("boundary: M9"))
    ));
    assert!(matches!(
        mesh.entity_blocks(2),
        Err(GridError::EntitiesNotCreated(2))
    ));
    assert!(matches!(
        mesh.connectivity(ConnKey { from: 2, to: 0 }),
        Err(GridError::UnsupportedConnectivity(_))
    ));
    assert!(matches!(
        mesh.connectivity(ConnKey { from: 2, to: 1 }),
        Err(GridError::ConnectivityNotCreated(_))
    ));
}
