//! `QuadField` and `EVector` (M3 T12): the tiled layout of Listing B4b read back cell-major
//! at every tile width and chunking, and the E-vector against a host transpose of the
//! L-vector with chunk boundaries on tile boundaries.

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl::{JacShape, QuadField};
use gridcl_device::copy;
use gridcl_device::testing::with_properties;

/// The value of component k at point q of cell c.
fn value(c: usize, q: usize, k: usize) -> f32 {
    (c * 100 + q * 10 + k) as f32
}

#[test]
fn to_host_cell_major_follows_the_tile_formula() {
    let client = gridcl_testing::test_client();
    let (num_cells, num_points, len) = (53, 3, 6); // JacShape<D2, D3>
    for tile in [1, 2, 4, 8, 16] {
        for limit in [None, Some((16 * num_points * len * 4) as u64)] {
            let mut props = client.properties().clone();
            if let Some(l) = limit {
                props.memory.max_page_size = l;
            }
            let field = with_properties(props, || {
                QuadField::<f32, JacShape<D2, D3>>::uninit(&client, num_cells, num_points, tile)
            })
            .unwrap();
            assert_eq!(
                (field.tile(), field.num_cells(), field.num_points()),
                (tile, num_cells, num_points)
            );
            if limit.is_some() {
                assert_eq!(
                    field.chunks().len(),
                    num_cells.div_ceil(16),
                    "tile {tile}: chunks"
                );
            }
            // Fill every chunk by the index formula of Listing B4b, counted from the chunk.
            for chunk in field.chunks() {
                assert_eq!(chunk.first_cell % tile, 0);
                let mut host = vec![-1.0f32; chunk.data.len()];
                for c in 0..chunk.num_cells {
                    for q in 0..num_points {
                        for k in 0..len {
                            let i = ((c / tile) * num_points * len + q * len + k) * tile + c % tile;
                            host[i] = value(chunk.first_cell + c, q, k);
                        }
                    }
                }
                let src = DeviceBuffer::from_host(&client, &host).unwrap();
                let mut alias = chunk.data.share();
                copy(&src, &mut alias).unwrap();
            }
            let back = field.to_host_cell_major().unwrap();
            assert_eq!(back.len(), num_cells * num_points);
            for c in 0..num_cells {
                for q in 0..num_points {
                    let j = back[c * num_points + q];
                    for (d, column) in j.iter().enumerate() {
                        for (g, &x) in column.iter().enumerate() {
                            assert_eq!(
                                x,
                                value(c, q, d * 3 + g),
                                "tile {tile}, cell {c}, point {q}"
                            );
                        }
                    }
                }
            }
        }
    }
}

/// The E-vector of block `b` against the host transpose of the L-vector.
fn check_evector<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
    mesh: &Mesh<f32, TD, GD>,
    b: usize,
    what: &str,
) {
    let gd = GD::N;
    let nodes = mesh.nodes().to_host().unwrap();
    let gb = mesh.geometry_block(b);
    let k = gb.dofmap.stride as usize;
    let dofmap = gb.dofmap.data.to_host().unwrap();
    let n = dofmap.len() / k;
    let xe = mesh.evector(b).unwrap();
    assert_eq!(xe.num_cells(), n);
    let mut covered = 0;
    for chunk in xe.chunks() {
        assert_eq!(chunk.first_cell, covered, "{what}: chunks in order");
        assert_eq!(
            chunk.first_cell % gridcl::VECTOR_PAD,
            0,
            "{what}: chunk on a tile boundary"
        );
        let stride = chunk.num_cells.next_multiple_of(gridcl::VECTOR_PAD);
        let data = chunk.data.to_host().unwrap();
        assert_eq!(data.len(), k * gd * stride);
        for node in 0..k {
            for g in 0..gd {
                for c in 0..stride {
                    // Padding lanes copy the block's last cell.
                    let cell = (chunk.first_cell + c).min(n - 1);
                    let want = nodes[dofmap[cell * k + node] as usize * gd + g];
                    assert_eq!(
                        data[(node * gd + g) * stride + c],
                        want,
                        "{what}: ({node}, {g}, {c})"
                    );
                }
            }
        }
        covered += chunk.num_cells;
    }
    assert_eq!(covered, n);
    // Built once: a second call returns the cached E-vector.
    assert!(core::ptr::eq(xe, mesh.evector(b).unwrap()));
}

#[test]
fn evector_is_the_transposed_lvector() {
    let client: Client = gridcl_testing::test_client();
    let h = gridcl_testing::mixed_box(2, 2).with_shuffled_nodes(4);
    let mesh = h.build::<f32, D3, D3>(&client).unwrap();
    for b in 0..mesh.cell_blocks().len() {
        check_evector(&mesh, b, "mixed box");
    }
    // An injected limit of 2 · VECTOR_PAD cells per chunk: hexahedra of degree 2 have 27 · 3
    // values per cell.
    let mut props = client.properties().clone();
    props.memory.max_page_size = (2 * 16 * 27 * 3 * 4) as u64;
    with_properties(props, || {
        let h = gridcl_testing::cube_box(Cell3::Hexahedron, 4, 2);
        let mesh = h.build::<f32, D3, D3>(&client).unwrap();
        assert_eq!(mesh.evector(0).unwrap().chunks().len(), 2);
        check_evector(&mesh, 0, "hexahedra in chunks of 32 cells");
    });
    // An empty block has no chunks.
    let mesh = h.first_cells(0).build::<f32, D3, D3>(&client).unwrap();
    assert!(mesh.evector(0).unwrap().chunks().is_empty());
}
