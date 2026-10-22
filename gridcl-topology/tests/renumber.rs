//! Entity renumbering (M3 T9): serial entities are all owned and ordered by the smallest flat
//! cell index that touches them; `EntitiesNotCreated` before creation and for the cell
//! dimension; the Point block of dimension 0.

use gridcl::prelude::*;
use gridcl_testing::HostMesh;

/// For every entity block of dimension `dim`: the smallest flat cell index touching each
/// entity, read from the downward connectivity.
fn min_cells<TD: TopoDim + EmbeddedIn<TD>>(
    mesh: &mut Mesh<f32, TD, TD>,
    dim: usize,
) -> Vec<Vec<u32>> {
    mesh.create_entities(dim).unwrap();
    let blocks = mesh.entity_blocks(dim).unwrap();
    let total: usize = blocks.iter().map(|b| b.index_map.num_local()).sum();
    let mut min = vec![u32::MAX; total];
    let key = ConnKey {
        from: TD::N as u8,
        to: dim as u8,
    };
    let Connectivity::Downward(down) = mesh.connectivity(key).unwrap() else {
        panic!("downward");
    };
    let mut first_cell = 0u32;
    for strided in down {
        let k = strided.stride as usize;
        let rows = strided.data.to_host().unwrap();
        for (c, row) in rows.chunks_exact(k).enumerate() {
            for &e in row {
                min[e as usize] = min[e as usize].min(first_cell + c as u32);
            }
        }
        first_cell += (rows.len() / k) as u32;
    }
    blocks
        .iter()
        .map(|b| {
            let o = b.offset as usize;
            min[o..o + b.index_map.num_local()].to_vec()
        })
        .collect()
}

fn check<TD: TopoDim + EmbeddedIn<TD>>(h: &HostMesh, what: &str) {
    let client = gridcl_testing::test_client();
    let mut mesh = h.build::<f32, TD, TD>(&client).unwrap();
    for dim in 1..TD::N {
        for (b, mins) in min_cells(&mut mesh, dim).iter().enumerate() {
            assert!(
                mins.iter().all(|&m| m != u32::MAX),
                "{what}: an entity touches no cell"
            );
            assert!(
                mins.windows(2).all(|w| w[0] <= w[1]),
                "{what}, dimension {dim}, block {b}: entities not in locality order"
            );
        }
        for block in mesh.entity_blocks(dim).unwrap() {
            // Serial: every entity is owned.
            assert_eq!(block.index_map.num_ghosts(), 0);
            assert_eq!(block.index_map.num_owned(), block.index_map.num_local());
            assert_eq!(
                block.index_map.owned_range(),
                (0, block.index_map.num_local() as i64)
            );
        }
    }
}

#[test]
fn entities_follow_their_first_cell() {
    use gridcl_testing::*;
    check::<D3>(&mixed_box(2, 1).with_shuffled_ids(3), "mixed box");
    check::<D3>(
        &cube_box(Cell3::Prism, 2, 1).with_shuffled_nodes(5),
        "prism box",
    );
    check::<D2>(
        &square_box(Cell2::Triangle, 4, 1).with_shuffled_ids(9),
        "triangle box",
    );
    check::<D2>(
        &square_box(Cell2::Quadrilateral, 3, 2).with_shuffled_ids(1),
        "quadrilateral box",
    );
}

#[test]
fn entities_exist_only_once_created() {
    let client = gridcl_testing::test_client();
    let mut mesh = gridcl_testing::mixed_box(1, 1)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    for dim in 0..4 {
        assert!(
            matches!(mesh.entity_blocks(dim), Err(GridError::EntitiesNotCreated(d)) if d == dim)
        );
    }
    assert!(matches!(
        mesh.create_entities(3),
        Err(GridError::EntitiesNotCreated(3))
    ));
    mesh.create_entities(2).unwrap();
    assert!(mesh.entity_blocks(2).is_ok());
    assert!(matches!(
        mesh.entity_blocks(1),
        Err(GridError::EntitiesNotCreated(1))
    ));
    assert!(matches!(
        mesh.entity_blocks(3),
        Err(GridError::EntitiesNotCreated(3))
    ));
    // A second call keeps the entities.
    let n = mesh.entity_blocks(2).unwrap()[0].index_map.num_local();
    mesh.create_entities(2).unwrap();
    assert_eq!(mesh.entity_blocks(2).unwrap()[0].index_map.num_local(), n);
}

#[test]
fn dimension_zero_is_one_point_block() {
    let client = gridcl_testing::test_client();
    let mut mesh = gridcl_testing::interval_box(5, 2)
        .build::<f32, D1, D1>(&client)
        .unwrap();
    mesh.create_entities(0).unwrap();
    let blocks = mesh.entity_blocks(0).unwrap();
    assert_eq!(blocks.len(), 1);
    assert_eq!(blocks[0].cell, CellType::Point);
    assert_eq!(blocks[0].vertices.stride, 1);
    assert_eq!(
        blocks[0].vertices.data.to_host().unwrap(),
        (0..6).collect::<Vec<u32>>()
    );
}
