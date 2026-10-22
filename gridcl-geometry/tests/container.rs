//! `Geometry` and `GeometryBlock` (M3 T5): uploads, origin, relative coordinates, checks and
//! deep copies.

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Real};
use gridcl_dims::{Cell2, D2, D3};
use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
use gridcl_parallel::{IndexMap, Serial};

const TRI: CoordinateElement<D2> = CoordinateElement {
    cell: Cell2::Triangle,
    degree: 1,
    family: NodeFamily::Equispaced,
};
const QUAD: CoordinateElement<D2> = CoordinateElement {
    cell: Cell2::Quadrilateral,
    degree: 2,
    family: NodeFamily::Gll,
};

/// A triangle and a P2 quadrilateral over 9 nodes in 3D, relative to `origin`.
fn geometry<F: Real>(client: &Client, origin: [f64; 3]) -> (Geometry<F, D2, D3>, Vec<F>) {
    let nodes: Vec<F> = (0..27)
        .map(|i| F::narrow(f64::from(i) * 0.25 - 3.0))
        .collect();
    let tri = vec![0u32, 1, 2];
    let quad: Vec<u32> = (0..9).rev().collect();
    let blocks = vec![
        GeometryBlock {
            element: TRI,
            dofmap: DeviceStrided {
                data: DeviceBuffer::from_host(client, &tri).unwrap(),
                stride: 3,
            },
        },
        GeometryBlock {
            element: QUAD,
            dofmap: DeviceStrided {
                data: DeviceBuffer::from_host(client, &quad).unwrap(),
                stride: 9,
            },
        },
    ];
    let g = Geometry::new(
        blocks,
        DeviceBuffer::from_host(client, &nodes).unwrap(),
        IndexMap::new(&Serial, 9).unwrap(),
        origin,
    )
    .unwrap();
    (g, nodes)
}

fn same_bits<F: Real>(a: &[F], b: &[F]) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| x.widen().to_bits() == y.widen().to_bits())
}

fn check<F: Real>(client: &Client) {
    let origin = [1.5, -2.0, 1e6];
    let (g, nodes) = geometry::<F>(client, origin);
    assert_eq!(g.origin(), origin);
    assert_eq!(g.node_index_map().num_local(), 9);
    assert_eq!(g.blocks().len(), 2);
    assert_eq!(g.blocks()[0].element, TRI);
    assert_eq!(g.blocks()[1].element, QUAD);
    assert_eq!(
        (g.blocks()[1].nodes_per_cell(), g.blocks()[1].num_cells()),
        (9, 1)
    );
    // The stored coordinates are the relative ones, unchanged.
    let stored = g.nodes().to_host().unwrap();
    assert!(same_bits(&stored, &nodes));
    assert_eq!(g.blocks()[0].dofmap.data.to_host().unwrap(), [0, 1, 2]);
    assert_eq!(
        g.blocks()[1].dofmap.data.to_host().unwrap(),
        (0..9).rev().collect::<Vec<u32>>()
    );
    // A deep copy holds the same data (DeviceBuffer::deep_clone makes new allocations).
    let copy = g.try_clone(&Serial).unwrap();
    assert_eq!(copy.origin(), origin);
    assert_eq!(copy.node_index_map().num_owned(), 9);
    assert!(same_bits(&copy.nodes().to_host().unwrap(), &stored));
    assert_eq!(
        copy.blocks()[1].dofmap.data.to_host().unwrap(),
        g.blocks()[1].dofmap.data.to_host().unwrap()
    );
}

#[test]
fn uploads_origin_and_relative_coordinates() {
    let client = gridcl_testing::test_client();
    check::<f32>(&client);
    if gridcl_testing::supports::<f64>(&client) {
        check::<f64>(&client);
    }
}

#[test]
fn node_count_must_match_the_index_map() {
    let client = gridcl_testing::test_client();
    let err = Geometry::<f32, D2, D3>::new(
        vec![],
        DeviceBuffer::from_host(&client, &[0.0f32; 7]).unwrap(),
        IndexMap::new(&Serial, 3).unwrap(),
        [0.0; 3],
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            err,
            GridError::LengthMismatch {
                expected: 9,
                found: 7
            }
        ),
        "{err}"
    );
}

#[test]
fn dofmap_stride_must_match_the_element() {
    let client = gridcl_testing::test_client();
    let block = GeometryBlock {
        element: TRI,
        dofmap: DeviceStrided {
            data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3]).unwrap(),
            stride: 4,
        },
    };
    let err = Geometry::<f32, D2, D3>::new(
        vec![block],
        DeviceBuffer::from_host(&client, &[0.0f32; 12]).unwrap(),
        IndexMap::new(&Serial, 4).unwrap(),
        [0.0; 3],
    )
    .err()
    .unwrap();
    assert!(
        matches!(
            err,
            GridError::LengthMismatch {
                expected: 3,
                found: 4
            }
        ),
        "{err}"
    );
}
