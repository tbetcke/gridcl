//! Connectivity (M3 T10): entity → cells rows in cell order, one or two cells per facet, and
//! every cell containing an entity reconstructs its canonical vertex list from the cell's
//! permutation info ("Topology invariants"); the connectivity errors.

use gridcl::prelude::*;
use gridcl_refcell::{sub_entities, sub_entity_types};
use gridcl_testing::HostMesh;
use gridcl_topology::decode_origin_ids;

/// The canonical vertex list (origin ids) of sub-entity `i` of dimension `dim` of a cell with
/// vertex origin ids `ids` and orientation word `bits` (A.6).
fn from_bits(cell: CellType, dim: usize, i: usize, ids: &[i64], bits: u32) -> Vec<i64> {
    let list = sub_entities(cell, dim)[i];
    if dim == 1 {
        let reversed = (bits >> i) & 1 == 1;
        let (a, b) = (ids[list[0]], ids[list[1]]);
        return if reversed { vec![b, a] } else { vec![a, b] };
    }
    let ne = sub_entities(cell, 1).len();
    let field = (bits >> (ne + 3 * i)) & 0b111;
    let (rotation, reflection) = ((field & 0b11) as usize, field >> 2);
    let quad = sub_entity_types(cell, 2)[i] == CellType::Quadrilateral;
    let cycle: &[usize] = if quad { &[0, 1, 3, 2] } else { &[0, 1, 2] };
    let m = cycle.len();
    let first = cycle[rotation];
    let succ = cycle[(rotation + 1) % m];
    let pred = cycle[(rotation + m - 1) % m];
    let (lower, other) = if reflection == 0 {
        (succ, pred)
    } else {
        (pred, succ)
    };
    let mut slots = vec![first, lower, other];
    if quad {
        slots.push(first ^ 3);
    }
    slots.iter().map(|&s| ids[list[s]]).collect()
}

fn check<TD: TopoDim + EmbeddedIn<TD>>(h: &HostMesh, what: &str) {
    let client = gridcl_testing::test_client();
    let mut mesh = h.build::<f32, TD, TD>(&client).unwrap();
    let tdim = TD::N;
    let vids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
    // Per flat cell: type, vertex origin ids and orientation word.
    let mut cells: Vec<(CellType, Vec<i64>, u32)> = Vec::new();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let cell: CellType = block.cell.into();
        let k = block.vertices.stride as usize;
        let rows = block.vertices.data.to_host().unwrap();
        let bits = mesh.permutation_info(b).unwrap().to_host().unwrap();
        for (row, &w) in rows.chunks_exact(k).zip(&bits) {
            cells.push((cell, row.iter().map(|&v| vids[v as usize]).collect(), w));
        }
    }
    for dim in 0..tdim {
        let up = ConnKey {
            from: dim as u8,
            to: tdim as u8,
        };
        assert!(matches!(
            mesh.connectivity(up),
            Err(GridError::ConnectivityNotCreated(_))
        ));
        mesh.create_connectivity(dim, tdim).unwrap();
        let Connectivity::Upward(csr) = mesh.connectivity(up).unwrap() else {
            panic!("(d, tdim) is upward");
        };
        let offsets = csr.offsets.to_host().unwrap();
        let indices = csr.indices.to_host().unwrap();
        let blocks = mesh.entity_blocks(dim).unwrap();
        let num: usize = blocks.iter().map(|b| b.index_map.num_local()).sum();
        assert_eq!(offsets.len(), num + 1, "{what}: offsets");
        assert_eq!(*offsets.last().unwrap() as usize, indices.len());
        // Entity vertex lists as origin ids, by flat index.
        let mut lists: Vec<Vec<i64>> = Vec::new();
        for block in blocks {
            let s = block.vertices.stride as usize;
            for row in block.vertices.data.to_host().unwrap().chunks_exact(s) {
                lists.push(row.iter().map(|&v| vids[v as usize]).collect());
            }
        }
        // cell → entity of dimension dim, by flat cell.
        let downward: Vec<Vec<u32>> = if dim == 0 {
            mesh.cell_blocks()
                .iter()
                .flat_map(|b| {
                    let k = b.vertices.stride as usize;
                    b.vertices
                        .data
                        .to_host()
                        .unwrap()
                        .chunks_exact(k)
                        .map(<[u32]>::to_vec)
                        .collect::<Vec<_>>()
                })
                .collect()
        } else {
            let Connectivity::Downward(d) = mesh
                .connectivity(ConnKey {
                    from: tdim as u8,
                    to: dim as u8,
                })
                .unwrap()
            else {
                panic!("downward");
            };
            d.iter()
                .flat_map(|s| {
                    let k = s.stride as usize;
                    s.data
                        .to_host()
                        .unwrap()
                        .chunks_exact(k)
                        .map(<[u32]>::to_vec)
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        let mut seen = 0;
        for e in 0..num {
            let row = &indices[offsets[e] as usize..offsets[e + 1] as usize];
            assert!(
                row.windows(2).all(|w| w[0] < w[1]),
                "{what}: row {e} not in cell order"
            );
            if dim == tdim - 1 {
                assert!(
                    (1..=2).contains(&row.len()),
                    "{what}: facet {e} has {} cells",
                    row.len()
                );
            } else {
                assert!(
                    !row.is_empty(),
                    "{what}: entity {e} of dimension {dim} has no cell"
                );
            }
            for &c in row {
                let i = downward[c as usize]
                    .iter()
                    .position(|&x| x as usize == e)
                    .unwrap_or_else(|| panic!("{what}: cell {c} does not contain entity {e}"));
                if dim >= 1 {
                    let (cell, ids, bits) = &cells[c as usize];
                    assert_eq!(
                        from_bits(*cell, dim, i, ids, *bits),
                        lists[e],
                        "{what}: dimension {dim}, entity {e}, cell {c}"
                    );
                }
                seen += 1;
            }
        }
        assert_eq!(
            seen,
            downward.iter().map(Vec::len).sum::<usize>(),
            "{what}: every pair once"
        );
    }
}

#[test]
fn rows_and_neighbouring_cells_agree() {
    use gridcl_testing::*;
    check::<D3>(&mixed_box(2, 1).with_shuffled_ids(11), "mixed box");
    check::<D3>(
        &cube_box(Cell3::Prism, 2, 1).with_shuffled_ids(12),
        "prism box",
    );
    check::<D3>(
        &cube_box(Cell3::Hexahedron, 2, 2).with_shuffled_nodes(2),
        "hexahedron box",
    );
    check::<D3>(&cube_box(Cell3::Tetrahedron, 2, 1), "tetrahedron box");
    check::<D2>(
        &square_box(Cell2::Triangle, 3, 1).with_shuffled_ids(13),
        "triangle box",
    );
    check::<D2>(&square_box(Cell2::Quadrilateral, 3, 1), "quadrilateral box");
}

#[test]
fn connectivity_errors() {
    let client = gridcl_testing::test_client();
    let mut mesh = gridcl_testing::mixed_box(1, 1)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    for (from, to) in [(1, 2), (2, 1), (0, 1), (3, 0), (3, 3), (2, 0), (4, 3)] {
        assert!(
            matches!(
                mesh.create_connectivity(from, to),
                Err(GridError::UnsupportedConnectivity(_))
            ),
            "{from} -> {to}"
        );
        let key = ConnKey {
            from: from as u8,
            to: to as u8,
        };
        assert!(matches!(
            mesh.connectivity(key),
            Err(GridError::UnsupportedConnectivity(_))
        ));
    }
    let down = ConnKey { from: 3, to: 2 };
    assert!(
        matches!(mesh.connectivity(down), Err(GridError::ConnectivityNotCreated(k)) if k == down)
    );
    // (tdim, d) comes from entity creation, and create_connectivity creates the entities.
    mesh.create_connectivity(3, 2).unwrap();
    assert!(mesh.entity_blocks(2).is_ok());
    assert!(matches!(
        mesh.connectivity(down),
        Ok(Connectivity::Downward(_))
    ));
}
