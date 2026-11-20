//! Iterating over blocks, cells, entities and connectivity (the book, tutorial "Iterating over a mesh").
//!
//! The mesh is [`two_blocks`](crate::build_mesh::two_blocks): the rectangle [0, 2] × [0, 1] as
//! one quadrilateral and two triangles. Every read of a device array in this program
//! (`to_host`, `read_node`) is for inspection: it copies the array to the host and waits for
//! the device.

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds a mesh in `F` on `client` and walks its blocks, cells, entities and connectivity, reading them back to the host for inspection.
///
/// It checks the block structure, the cell origin ids and vertex coordinates, the number of
/// edges, their canonical orientation, the six boundary and two interior edges, and the
/// shape of the cell → edge connectivity.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client) -> tetraloma::Result<()> {
    let mut mesh: PlanarMesh<F> = crate::build_mesh::two_blocks(client)?;
    println!(
        "rectangle of {} cell blocks on {} in {}",
        mesh.cell_blocks().len(),
        client.name(),
        crate::scalar_name::<F>()
    );
    let tol = 10.0 * crate::tol::<F>();

    // ANCHOR: blocks
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let element = mesh.geometry_block(b).element;
        println!(
            "block {b}: {:?}, {} owned and {} ghost cells, degree {}, {} vertices per cell",
            block.cell,
            block.index_map.num_owned(),
            block.index_map.num_ghosts(),
            element.degree,
            block.vertices.stride
        );
    }
    // ANCHOR_END: blocks
    let cells: Vec<Cell2> = mesh.cell_blocks().iter().map(|b| b.cell).collect();
    assert_eq!(cells, [Cell2::Quadrilateral, Cell2::Triangle]);
    assert!(
        mesh.cell_blocks()
            .iter()
            .all(|b| b.index_map.num_ghosts() == 0)
    );

    // ANCHOR: cells
    // Copies for inspection: the nodes (relative to the origin) and, per block, the cells.
    let origin = mesh.origin();
    let nodes = mesh.nodes().to_host()?;
    let mut corners = Vec::new();
    for (b, block) in mesh.cell_blocks().iter().enumerate() {
        let ids = block.origin_ids.to_host()?;
        let dofmap = &mesh.geometry_block(b).dofmap;
        let rows = dofmap.data.to_host()?;
        let num_vertices = block.vertices.stride as usize;
        for (c, row) in rows.chunks_exact(dofmap.stride as usize).enumerate() {
            let id = origin_id(ids[2 * c], ids[2 * c + 1]);
            // The first nodes of a cell are its vertices, in reference-cell order.
            let points: Vec<[f64; 2]> = row[..num_vertices]
                .iter()
                .map(|&n| {
                    let n = n as usize;
                    [
                        nodes[2 * n].widen() + origin[0],
                        nodes[2 * n + 1].widen() + origin[1],
                    ]
                })
                .collect();
            println!("  cell {id}: vertices at {points:?}");
            corners.push((id, points));
        }
    }
    // ANCHOR_END: cells
    let ids: Vec<i64> = corners.iter().map(|(id, _)| *id).collect();
    assert_eq!(
        ids,
        [100, 200, 201],
        "the cells' origin ids, block by block"
    );
    // The quadrilateral's vertices in tensor order.
    let square = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    for (got, want) in corners[0].1.iter().zip(square) {
        assert!(
            (got[0] - want[0]).abs() <= tol && (got[1] - want[1]).abs() <= tol,
            "quadrilateral vertex {got:?}, expected {want:?}"
        );
    }

    // ANCHOR: entities
    mesh.create_entities(1)?; // the edges, and the connectivity cell → edge
    let vertex_ids = mesh.vertex_origin_ids().to_host()?;
    for block in mesh.entity_blocks(1)? {
        let vertices = block.vertices.data.to_host()?;
        println!(
            "{:?} block: {} edges from flat index {}",
            block.cell,
            block.index_map.num_local(),
            block.offset
        );
        for (e, edge) in vertices.as_chunks::<2>().0.iter().enumerate() {
            // Edges start at the vertex of the smaller origin id.
            let [a, b] = [edge[0], edge[1]].map(|v| {
                let v = v as usize;
                origin_id(vertex_ids[2 * v], vertex_ids[2 * v + 1])
            });
            println!("  edge {e}: from vertex {a} to vertex {b}");
            assert!(a < b, "edge {e} runs from {a} to {b}");
        }
    }
    // ANCHOR_END: entities
    let edges = mesh.entity_blocks(1)?;
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].cell, CellType::Interval);
    assert_eq!(edges[0].index_map.num_local(), 8);

    // ANCHOR: connectivity
    // edge → cell: variable-length rows (CSR), created on request.
    mesh.create_connectivity(1, 2)?;
    let Connectivity::Upward(edge_cells) = mesh.connectivity(ConnKey { from: 1, to: 2 })? else {
        unreachable!("edge → cell is upward")
    };
    let offsets = edge_cells.offsets.to_host()?;
    let cells_per_edge: Vec<u32> = offsets.windows(2).map(|w| w[1] - w[0]).collect();
    let boundary = cells_per_edge.iter().filter(|&&n| n == 1).count();
    println!("cells per edge {cells_per_edge:?}: {boundary} boundary edges");

    // cell → edge: one strided array per cell block, created with the edges.
    let Connectivity::Downward(cell_edges) = mesh.connectivity(ConnKey { from: 2, to: 1 })? else {
        unreachable!("cell → edge is downward")
    };
    for (b, strided) in cell_edges.iter().enumerate() {
        let data = strided.data.to_host()?;
        let rows: Vec<&[u32]> = data.chunks_exact(strided.stride as usize).collect();
        println!("block {b}: edges of each cell {rows:?}");
    }
    // ANCHOR_END: connectivity
    assert_eq!(cells_per_edge.len(), 8);
    assert_eq!(boundary, 6, "the rectangle's perimeter has six edges");
    assert_eq!(cells_per_edge.iter().filter(|&&n| n == 2).count(), 2);
    assert_eq!(edge_cells.indices.len(), 6 + 2 * 2);
    let shapes: Vec<(u32, usize)> = cell_edges
        .iter()
        .map(|rows| (rows.stride, rows.data.len()))
        .collect();
    assert_eq!(
        shapes,
        [(4, 4), (3, 6)],
        "four edges per quadrilateral, three per triangle"
    );

    // ANCHOR: host
    // One node, with the origin added back: one device read per call, for debugging.
    for i in 0..mesh.node_index_map().num_local() {
        let x = mesh.read_node(i)?;
        for (g, xg) in x.iter().enumerate() {
            let y = nodes[2 * i + g].widen() + origin[g];
            assert!(
                (xg.widen() - y).abs() <= tol,
                "node {i}: {xg:?}, expected {y}"
            );
        }
    }
    // Values at quadrature points come back in [cell][point] order.
    let rule = QuadRule::for_grid(&mesh, 1)?;
    let centres = mesh.physical_points(&rule)?;
    for (b, field) in centres.blocks.iter().enumerate() {
        let x = field.to_host_cell_major()?;
        println!(
            "block {b}: {} cells × {} points, first {:?}",
            field.num_cells(),
            field.num_points(),
            x[0]
        );
    }
    // ANCHOR_END: host
    let centre = centres.blocks[0].to_host_cell_major()?[0];
    assert!(
        centre.iter().all(|c| (c.widen() - 0.5).abs() <= tol),
        "centre of the quadrilateral {centre:?}"
    );
    Ok(())
}

// ANCHOR: origin_id
/// The origin id of an interleaved pair (hi ^ 0x8000_0000, lo), as cells, vertices and the
/// mesh store them on the device.
pub fn origin_id(hi: u32, lo: u32) -> i64 {
    ((u64::from(hi ^ 0x8000_0000) << 32) | u64::from(lo)) as i64
}
// ANCHOR_END: origin_id
