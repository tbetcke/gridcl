//! Pulling physical points back to reference coordinates (the book, tutorial "Pulling back points").
//!
//! Two parts: the quadrature points of a curved mesh come back to the rule's reference points,
//! and the points of a moved copy of a mesh show what the pull-back returns for points outside
//! a cell. The reads of the results (`to_host_cell_major`) are for inspection.

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Pulls points of a curved mesh in `F` on `client` back to reference coordinates and checks that the forward map returns them.
///
/// The curved mesh is the quarter annulus of [`quarter_annulus`](crate::build_mesh::quarter_annulus)
/// at degree 3. Its physical points X(ξ_q) at a Gauss rule of degree 4 pull back to the rule's
/// points ξ_q, every lane converged. Then two unit squares side by side pull back the points of
/// copies moved by 0.25 and 0.5 along x: each point comes back as ξ_q + (shift, 0), inside or
/// outside the reference square, or, beyond the box the iteration is confined to, without
/// converging.
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
        "pull-back on {} in {}",
        client.name(),
        crate::scalar_name::<F>()
    );
    curved::<F>(client)?;
    outside::<F>(client)
}

/// The default tolerance of [`PullBackOptions`] (`tolerance: None`): 10⁻¹³ in f64 and 10⁻⁵ in
/// f32.
fn default_tolerance<F>() -> f64 {
    if size_of::<F>() == 8 { 1e-13 } else { 1e-5 }
}

/// The quadrature points of a curved mesh come back to the rule's points.
fn curved<F: Real>(client: &Client) -> tetraloma::Result<()> {
    // ANCHOR: curved
    let mesh: PlanarMesh<F> = crate::build_mesh::quarter_annulus(client, 3, 1, 2)?;
    let rule = QuadRule::for_grid(&mesh, 4)?;
    // Absolute physical points X(ξ_q), per cell and point.
    let x = mesh.physical_points(&rule)?;
    // ξ per cell and point, and |Δξ|∞ of the last iteration.
    let pb = mesh.pull_back(&x, &PullBackOptions::default())?;
    // ANCHOR_END: curved

    // ANCHOR: check
    // For inspection: the results on the host, in [cell][point] order.
    let points = rule.get(Cell2::Quadrilateral)?;
    let xi = pb.reference.blocks[0].to_host_cell_major()?;
    let steps = pb.last_step.blocks[0].to_host_cell_major()?;
    let mut worst = 0.0f64;
    for (i, (p, step)) in xi.iter().zip(&steps).enumerate() {
        let q = i % points.len();
        let want = &points.points()[2 * q..2 * q + 2];
        assert!(step.widen() <= default_tolerance::<F>(), "point {i}: not converged");
        for (got, want) in p.iter().zip(want) {
            worst = worst.max((got.widen() - want).abs());
        }
    }
    println!(
        "  curved: {} points, largest |ξ − ξ_q| = {worst:.2e}",
        xi.len()
    );
    // ANCHOR_END: check
    assert_eq!(xi.len(), 2 * points.len(), "two cells");
    assert!(worst <= 10.0 * crate::tol::<F>(), "|ξ − ξ_q| = {worst}");
    Ok(())
}

// ANCHOR: two_squares
/// Two unit squares side by side, [0, 1]² and [1, 2] × [0, 1], moved by `shift` along x.
fn two_squares<F: Real>(client: &Client, shift: f64) -> tetraloma::Result<PlanarMesh<F>> {
    let corners = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
    let coords: Vec<f64> = corners.iter().flat_map(|&[x, y]| [x + shift, y]).collect();
    let quadrilateral = CoordinateElement::<D2> {
        cell: Cell2::Quadrilateral,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(quadrilateral, &[0, 1, 3, 4, 1, 2, 4, 5], None)
        .build()
}
// ANCHOR_END: two_squares

/// Points outside a cell: inside, outside but converged, and not converged.
fn outside<F: Real>(client: &Client) -> tetraloma::Result<()> {
    let tol = 10.0 * crate::tol::<F>();
    // ANCHOR: outside
    let mesh = two_squares::<F>(client, 0.0)?;
    let rule = QuadRule::for_grid(&mesh, 3)?; // 2 × 2 Gauss points per cell
    let points = rule.get(Cell2::Quadrilateral)?;
    for (shift, expected) in [(0.25, [4, 4, 0]), (0.5, [4, 0, 4])] {
        // The points X(ξ_q) + (shift, 0), in the layout pull_back takes.
        let moved = two_squares::<F>(client, shift)?;
        let x = moved.physical_points(&rule)?;
        let pb = mesh.pull_back(&x, &PullBackOptions::default())?;

        // For inspection: classify every point on the host.
        let xi = pb.reference.blocks[0].to_host_cell_major()?;
        let steps = pb.last_step.blocks[0].to_host_cell_major()?;
        let mut counts = [0; 3]; // inside, outside, not converged
        for (i, (p, step)) in xi.iter().zip(&steps).enumerate() {
            // The exact preimage: ξ_q moved by the shift.
            let q = i % points.len();
            let exact = [points.points()[2 * q] + shift, points.points()[2 * q + 1]];
            if step.widen() > default_tolerance::<F>() {
                // The preimage lies outside the box [−0.1, 1.1]² of the iteration.
                assert!(exact[0] > 1.1, "point {i}: ξ = {exact:?} did not converge");
                counts[2] += 1;
            } else {
                for (got, exact) in p.iter().zip(exact) {
                    assert!((got.widen() - exact).abs() <= tol, "point {i}: ξ = {p:?}");
                }
                if p.iter().all(|c| (-tol..=1.0 + tol).contains(&c.widen())) {
                    counts[0] += 1;
                } else {
                    counts[1] += 1; // converged, outside the reference square
                }
            }
        }
        let [inside, outside, failed] = counts;
        println!("  shift {shift}: {inside} inside, {outside} outside, {failed} not converged");
        assert_eq!(counts, expected, "shift {shift}");
    }
    // ANCHOR_END: outside
    Ok(())
}
