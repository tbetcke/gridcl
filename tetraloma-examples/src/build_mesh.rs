//! Building meshes from arrays (the book, tutorial "Building a mesh").
//!
//! [`run`] builds three meshes with [`MeshBuilder`]: the unit square from arrays
//! ([`unit_square`]), a rectangle of a quadrilateral and two triangles ([`two_blocks`]), and a
//! quarter annulus of curved quadrilaterals ([`quarter_annulus`]). The other book programs and
//! the book's snippets reuse these meshes.

use std::f64::consts::{FRAC_PI_2, PI};

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Builds meshes from host arrays in `F` on `client`: one cell type, several cell types, and higher-degree geometry, and checks their sizes and measures.
///
/// It checks the cell and vertex counts and the area of the unit square and of the rectangle
/// [0, 2] × [0, 1], and the area of the quarter annulus at degrees 1, 2 and 4: at degree 1 it
/// is the area of the straight-sided cells, and with the degree the error against the exact
/// 3π/4 falls.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client) -> tetraloma::Result<()> {
    println!(
        "building meshes on {} in {}",
        client.name(),
        crate::scalar_name::<F>()
    );
    let tol = 10.0 * crate::tol::<F>();

    // One cell type.
    let square: PlanarMesh<F> = unit_square(client)?;
    let block = &square.cell_blocks()[0];
    let area = square.measure(1)?;
    println!(
        "  unit square: {} {:?} cells, {} vertices, origin {:?}, area {area}",
        block.index_map.num_owned(),
        block.cell,
        square.vertex_index_map().num_local(),
        square.origin()
    );
    assert_eq!(square.cell_blocks().len(), 1);
    assert_eq!(block.index_map.num_owned(), 4);
    assert_eq!(square.vertex_index_map().num_local(), 9);
    assert_eq!(square.origin(), [0.5, 0.5]);
    assert!((area - 1.0).abs() <= tol, "area {area}, expected 1");

    // Several cell types.
    let rectangle: PlanarMesh<F> = two_blocks(client)?;
    let area = rectangle.measure(1)?;
    let blocks: Vec<(Cell2, usize)> = rectangle
        .cell_blocks()
        .iter()
        .map(|b| (b.cell, b.index_map.num_owned()))
        .collect();
    println!("  rectangle: blocks {blocks:?}, area {area}");
    assert_eq!(
        blocks,
        [(Cell2::Quadrilateral, 1), (Cell2::Triangle, 2)],
        "one block per cell type, in the order of the cell_block calls"
    );
    assert_eq!(rectangle.vertex_index_map().num_local(), 6);
    assert!((area - 2.0).abs() <= 2.0 * tol, "area {area}, expected 2");

    // Higher-degree geometry: the error of the area falls with the degree.
    let exact = 3.0 * PI / 4.0;
    let mut last = f64::INFINITY;
    for degree in [1, 2, 4] {
        let annulus: PlanarMesh<F> = quarter_annulus(client, degree, 1, 2)?;
        let area = annulus.measure(2 * degree)?;
        let error = (area - exact).abs() / exact;
        println!(
            "  quarter annulus, degree {degree}: {} nodes, area {area:.9}, relative error {error:.2e}",
            annulus.node_index_map().num_local()
        );
        if degree == 1 {
            // Two straight-sided cells of area (3/2) sin(π/4) each.
            let straight = 3.0 * (PI / 4.0).sin();
            assert!(
                (area - straight).abs() <= tol * straight,
                "area {area}, expected {straight}"
            );
        }
        assert!(
            error < last || error <= tol,
            "the area error grows to {error:.2e} at degree {degree}"
        );
        last = error;
    }
    // 1.2 · 10⁻⁶ at degree 4, as the interpolation error of the polar map predicts.
    assert!(last < 1e-5, "area error {last:.2e} at degree 4");
    Ok(())
}

// ANCHOR: unit_square
/// The unit square [0, 1]² as 2 × 2 quadrilaterals of degree 1.
///
/// # Errors
///
/// The errors of [`MeshBuilder::build`].
pub fn unit_square<F: Real>(client: &Client) -> tetraloma::Result<PlanarMesh<F>> {
    // Nine nodes, two coordinates each, row by row.
    #[rustfmt::skip]
    let coords: [f64; 18] = [
        0.0, 0.0,   0.5, 0.0,   1.0, 0.0,
        0.0, 0.5,   0.5, 0.5,   1.0, 0.5,
        0.0, 1.0,   0.5, 1.0,   1.0, 1.0,
    ];
    // Four cells, four node indices each, in tensor order:
    // (0, 0), (1, 0), (0, 1), (1, 1) of the reference square.
    #[rustfmt::skip]
    let cells: [u32; 16] = [
        0, 1, 3, 4,
        1, 2, 4, 5,
        3, 4, 6, 7,
        4, 5, 7, 8,
    ];
    let quadrilateral = CoordinateElement::<D2> {
        cell: Cell2::Quadrilateral,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(quadrilateral, &cells, None)
        .build()
}
// ANCHOR_END: unit_square

// ANCHOR: two_blocks
/// The rectangle [0, 2] × [0, 1] as one quadrilateral ([0, 1]², origin id 100) and two
/// triangles ([1, 2] × [0, 1], origin ids 200 and 201), with node origin ids 10 to 15.
///
/// # Errors
///
/// The errors of [`MeshBuilder::build`].
pub fn two_blocks<F: Real>(client: &Client) -> tetraloma::Result<PlanarMesh<F>> {
    #[rustfmt::skip]
    let coords: [f64; 12] = [
        0.0, 0.0,   1.0, 0.0,   2.0, 0.0,
        0.0, 1.0,   1.0, 1.0,   2.0, 1.0,
    ];
    let node_ids: [i64; 6] = [10, 11, 12, 13, 14, 15];
    let quadrilateral = CoordinateElement::<D2> {
        cell: Cell2::Quadrilateral,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    let triangle = CoordinateElement::<D2> {
        cell: Cell2::Triangle,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, Some(&node_ids))
        // One block per cell type; triangles counter-clockwise.
        .cell_block(quadrilateral, &[0, 1, 3, 4], Some(&[100]))
        .cell_block(triangle, &[1, 2, 5, 1, 5, 4], Some(&[200, 201]))
        .build()
}
// ANCHOR_END: two_blocks

// ANCHOR: quarter_annulus
/// The quarter annulus 1 ≤ r ≤ 2, 0 ≤ θ ≤ π/2 as `radial` × `angular` quadrilaterals of
/// degree `degree`, with equispaced nodes placed on the exact polar map: the reference
/// coordinate ξ₀ runs along r and ξ₁ along θ.
///
/// # Errors
///
/// `UnsupportedDegree` for degree 0, `TooLarge` if a node index does not fit u32, and the
/// errors of [`MeshBuilder::build`].
pub fn quarter_annulus<F: Real>(
    client: &Client,
    degree: u32,
    radial: usize,
    angular: usize,
) -> tetraloma::Result<PlanarMesh<F>> {
    let family = NodeFamily::Equispaced;
    // The reference coordinates of the element's nodes, in tetraloma's node order.
    let reference = tetraloma_refcell::lagrange_points(CellType::Quadrilateral, degree, family)
        .map_err(|_| GridError::UnsupportedDegree {
            cell: "Quadrilateral",
            degree,
        })?;
    // A grid of (radial · p + 1) × (angular · p + 1) nodes in (r, θ), mapped to (x, y).
    let p = degree as usize;
    let (nr, nt) = (radial * p + 1, angular * p + 1);
    let mut coords = Vec::with_capacity(2 * nr * nt);
    for j in 0..nt {
        for i in 0..nr {
            let r = 1.0 + i as f64 / (nr - 1) as f64;
            let theta = FRAC_PI_2 * j as f64 / (nt - 1) as f64;
            coords.extend([r * theta.cos(), r * theta.sin()]);
        }
    }
    // Each cell lists its nodes in the element's order; node (ξ₀, ξ₁) of cell (a, b) is grid
    // node (a p + p ξ₀, b p + p ξ₁).
    let mut cells = Vec::with_capacity(radial * angular * reference.len() / 2);
    for b in 0..angular {
        for a in 0..radial {
            for xi in reference.chunks_exact(2) {
                let i = a * p + (xi[0] * p as f64).round() as usize;
                let j = b * p + (xi[1] * p as f64).round() as usize;
                cells.push(u32::try_from(j * nr + i).map_err(|_| GridError::TooLarge)?);
            }
        }
    }
    let element = CoordinateElement::<D2> {
        cell: Cell2::Quadrilateral,
        degree,
        family,
    };
    MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(element, &cells, None)
        .build()
}
// ANCHOR_END: quarter_annulus
