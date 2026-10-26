//! M3 review (Major 3): a geometry whose dofmap holds node indices past the local nodes reads
//! no memory out of bounds. The EVector kernel clamps such an index to the last node, so the
//! E-vector equals that of the clamped dofmap and is deterministic, in f32 and f64. A geometry
//! with cells but no nodes is rejected (M3 review, m11); the kernel's behaviour on that input is
//! a unit test of `evector.rs`.

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

/// Triangles with dofmap `dofmap` over `num_nodes` nodes in 3D.
fn geometry<F: Real>(client: &Client, dofmap: &[u32], num_nodes: usize) -> Geometry<F, D2, D3> {
    let nodes: Vec<F> = (0..3 * num_nodes)
        .map(|i| F::narrow(i as f64 * 0.5 - 1.0))
        .collect();
    Geometry::new(
        vec![GeometryBlock {
            element: TRI,
            dofmap: DeviceStrided {
                data: DeviceBuffer::from_host(client, dofmap).unwrap(),
                stride: 3,
            },
        }],
        DeviceBuffer::from_host(client, &nodes).unwrap(),
        IndexMap::new(&Serial, num_nodes).unwrap(),
        [0.0; 3],
    )
    .unwrap()
}

/// The E-vector of block 0, chunk by chunk.
fn evector<F: Real>(g: &Geometry<F, D2, D3>) -> Vec<Vec<F>> {
    g.evector(0)
        .unwrap()
        .chunks()
        .iter()
        .map(|c| c.data.to_host().unwrap())
        .collect()
}

fn check<F: Real>() {
    let client = gridcl_testing::test_client();
    let (cells, num_nodes) = (37usize, 7usize);
    let mut bad = Vec::new();
    for c in 0..cells {
        for k in 0..3 {
            bad.push(match (c % 4, k) {
                (0, 2) => (num_nodes + c) as u32,
                (1, 0) => u32::MAX,
                _ => ((c + k) % num_nodes) as u32,
            });
        }
    }
    let last = num_nodes as u32 - 1;
    let clamped: Vec<u32> = bad.iter().map(|&v| v.min(last)).collect();
    let got = evector(&geometry::<F>(&client, &bad, num_nodes));
    let again = evector(&geometry::<F>(&client, &bad, num_nodes));
    let want = evector(&geometry::<F>(&client, &clamped, num_nodes));
    // Compared bit for bit: the clamp only changes the read address.
    let bits = |x: &[Vec<F>]| -> Vec<Vec<u64>> {
        x.iter()
            .map(|c| c.iter().map(|v| v.widen().to_bits()).collect())
            .collect()
    };
    assert_eq!(bits(&got), bits(&again), "deterministic");
    assert_eq!(bits(&got), bits(&want), "as the clamped dofmap");
}

#[test]
fn out_of_range_nodes_read_the_last_node_f32() {
    check::<f32>();
}

#[test]
fn out_of_range_nodes_read_the_last_node_f64() {
    if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
        check::<f64>();
    } else {
        eprintln!("out_of_range_nodes_read_the_last_node_f64: f64 is not supported; skipped");
    }
}

fn rejects_cells_without_nodes<F: Real>(client: &Client) {
    let result = Geometry::<F, D2, D3>::new(
        vec![GeometryBlock {
            element: TRI,
            dofmap: DeviceStrided {
                data: DeviceBuffer::from_host(client, &[3u32, 4, 5, 9, 9, 1]).unwrap(),
                stride: 3,
            },
        }],
        DeviceBuffer::<F>::from_host(client, &[]).unwrap(),
        IndexMap::new(&Serial, 0).unwrap(),
        [0.0; 3],
    );
    assert!(matches!(
        result,
        Err(GridError::LengthMismatch {
            expected: 0,
            found: 6
        })
    ));
    // No nodes and no cells is a valid, empty geometry.
    let empty = Geometry::<F, D2, D3>::new(
        vec![GeometryBlock {
            element: TRI,
            dofmap: DeviceStrided {
                data: DeviceBuffer::from_host(client, &[]).unwrap(),
                stride: 3,
            },
        }],
        DeviceBuffer::<F>::from_host(client, &[]).unwrap(),
        IndexMap::new(&Serial, 0).unwrap(),
        [0.0; 3],
    );
    assert!(empty.is_ok());
}

#[test]
fn empty_nodes_with_cells_are_rejected() {
    let client = gridcl_testing::test_client();
    rejects_cells_without_nodes::<f32>(&client);
    if gridcl_testing::supports::<f64>(&client) {
        rejects_cells_without_nodes::<f64>(&client);
    } else {
        eprintln!("empty_nodes_with_cells_are_rejected: f64 is not supported; skipped");
    }
}
