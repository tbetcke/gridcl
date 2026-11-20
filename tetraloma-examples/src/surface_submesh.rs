//! A boundary surface mesh with its normals, Jacobians and area (Listing B9).

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Runs the example on `client`, in f64 where the device supports it and f32 otherwise.
///
/// The volume mesh is the unit ball as 64 curved hexahedra of degree 2, whose boundary facets
/// carry the generated tags `boundary-100` and `boundary-101` (every second facet). Each tag
/// becomes a surface mesh, as `scalp` in Listing B9. `run` checks that the normals are unit
/// vectors pointing out of the ball, that K J = I at every point, and that the two areas add up
/// to the sphere's 4π within the error of the degree-2 geometry.
///
/// # Errors
///
/// The errors of mesh creation, `facet_submesh` and the geometric factors; allocation and
/// device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run(client: &Client) -> tetraloma::Result<()> {
    if crate::has_f64(client) {
        surfaces::<f64>(client)
    } else {
        surfaces::<f32>(client)
    }
}

fn surfaces<F: Real>(client: &Client) -> tetraloma::Result<()> {
    let mut head: VolumeMesh<F> = tetraloma_testing::ball(1, 2)
        .chunk(0, 1)
        .builder::<F, D3, D3>(client)?
        .build()?;
    head.create_entities(2)?;
    println!(
        "ball: {} hexahedra of degree 2 on {} in {}",
        head.cell_blocks()[0].index_map.num_owned(),
        client.name(),
        crate::scalar_name::<F>()
    );

    let mut total = 0.0;
    for name in ["boundary-100", "boundary-101"] {
        let scalp_tag = head.tags().named(Codim::Bnd, name)?; // boundary facets only
        let (scalp, emap): (SurfaceMesh<F>, EntityMap) = head.facet_submesh(&scalp_tag)?;
        // Each facet has one parent cell; normals inherit outward orientation from it.
        assert_eq!(emap.child_to_parent().len(), scalp_tag.indices.len());

        let rule = QuadRule::for_grid(&scalp, 6)?; // one point set per cell type present
        let x = scalp.physical_points(&rule)?;
        let n = scalp.normals(&rule)?; // compiles: D2 is Codim1 in D3
        let j = scalp.jacobians(&rule)?; // per point a [[F; 3]; 2]: two columns of three
        let k = scalp.pseudo_inverse_jacobians(&rule)?; // per point a [[F; 2]; 3]
        let area = scalp.measure(6)?; // pseudo-determinant path, chosen at comptime
        total += area;

        let tol = 100.0 * crate::tol::<F>();
        let mut points = 0;
        for b in 0..scalp.cell_blocks().len() {
            let x = x.blocks[b].to_host_cell_major()?;
            let n = n.blocks[b].to_host_cell_major()?;
            let j = j.blocks[b].to_host_cell_major()?;
            let k = k.blocks[b].to_host_cell_major()?;
            for (((x, n), j), k) in x.iter().zip(&n).zip(&j).zip(&k) {
                let length: f64 = n.iter().map(|c| c.widen() * c.widen()).sum();
                assert!(
                    (length.sqrt() - 1.0).abs() <= tol,
                    "|n| = {}",
                    length.sqrt()
                );
                let outward: f64 = x.iter().zip(n).map(|(a, b)| a.widen() * b.widen()).sum();
                assert!(outward > 0.0, "n · x = {outward}: the normal points inward");
                // K J = I: K is 2 × 3 and J is 3 × 2, both column-major.
                for (a, row) in [[1.0, 0.0], [0.0, 1.0]].iter().enumerate() {
                    for (c, want) in row.iter().enumerate() {
                        let kj: f64 = k
                            .iter()
                            .zip(&j[c])
                            .map(|(k, j)| k[a].widen() * j.widen())
                            .sum();
                        assert!((kj - want).abs() <= tol, "(K J)[{a}][{c}] = {kj}");
                    }
                }
                points += 1;
            }
        }
        println!(
            "  {name}: {} quadrilaterals, {points} points, area {area:.6}",
            scalp.cell_blocks()[0].index_map.num_owned()
        );
    }

    let sphere = 4.0 * std::f64::consts::PI;
    let err = (total - sphere).abs() / sphere;
    println!("  total area {total:.6}, 4π = {sphere:.6}, relative error {err:.2e}");
    // The degree-2 surface of 96 quadrilaterals is within 10^−4 of the sphere (5.5 · 10^−5).
    assert!(err < 1e-4, "total area {total}, expected about 4π");
    Ok(())
}
