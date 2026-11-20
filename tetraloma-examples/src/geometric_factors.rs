//! Geometric factors at p = 1, 4 and 8 on a curved sphere.

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// The degrees of the example.
const DEGREES: [u32; 3] = [1, 4, 8];

/// Runs the example on `client`, in f64 where the device supports it and f32 otherwise.
///
/// The mesh is the unit sphere as 20 curved triangles of degree p (the icosahedron with every
/// node on the sphere), for p = 1, 4 and 8. At the points of a Gauss rule of degree 2p it
/// computes the physical points X, the Jacobians J, det J, the pseudo-inverses K and the
/// normals n, and it prints, per degree, the largest distance of X from the sphere, the largest
/// angle between n and X and the error of the area. `run` checks that n is a unit vector
/// pointing out of the sphere, that det J > 0, that both errors fall with p and that the area
/// at p = 8 is within 10^−4 of 4π.
///
/// # Errors
///
/// The errors of mesh creation and of the geometric factors; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run(client: &Client) -> tetraloma::Result<()> {
    if crate::has_f64(client) {
        factors::<f64>(client)
    } else {
        factors::<f32>(client)
    }
}

/// The errors of one degree.
struct Errors {
    /// Largest | |X| − 1 |.
    radius: f64,
    /// Relative error of the area against 4π.
    area: f64,
}

fn factors<F: Real>(client: &Client) -> tetraloma::Result<()> {
    println!(
        "curved sphere, 20 triangles, on {} in {}",
        client.name(),
        crate::scalar_name::<F>()
    );
    println!("   p  points  max ||X|-1|  max angle(n, X)  area error");
    let errors: Vec<Errors> = DEGREES
        .iter()
        .map(|&p| degree::<F>(client, p))
        .collect::<tetraloma::Result<_>>()?;
    // The geometry error falls with p, until rounding in F takes over at p = 8.
    let floor = 100.0 * crate::tol::<F>();
    for (p, pair) in DEGREES.windows(2).zip(errors.windows(2)) {
        let (lower, higher) = (&pair[0], &pair[1]);
        assert!(
            higher.radius < lower.radius || higher.radius <= floor,
            "| |X| − 1 | grows from p = {} to p = {}",
            p[0],
            p[1]
        );
        assert!(
            higher.area < lower.area || higher.area <= floor,
            "the area error grows from p = {} to p = {}",
            p[0],
            p[1]
        );
    }
    // At p = 8 the area is within 10^−4 of 4π (9.2 · 10^−6 in f64).
    let last = &errors[errors.len() - 1];
    assert!(last.area < 1e-4, "area error {} at p = 8", last.area);
    Ok(())
}

/// The geometric factors of the sphere at degree `p`.
fn degree<F: Real>(client: &Client, p: u32) -> tetraloma::Result<Errors> {
    let mesh: SurfaceMesh<F> = tetraloma_testing::sphere(0, p).build(client)?;
    // ANCHOR: factors
    let rule = QuadRule::for_grid(&mesh, 2 * p)?;
    let x = mesh.physical_points(&rule)?;
    let j = mesh.jacobians(&rule)?;
    let detj = mesh.detj(&rule)?;
    let k = mesh.pseudo_inverse_jacobians(&rule)?;
    let n = mesh.normals(&rule)?;
    // ANCHOR_END: factors

    let tol = 100.0 * crate::tol::<F>();
    let (mut radius, mut angle, mut points) = (0.0f64, 0.0f64, 0);
    for b in 0..mesh.cell_blocks().len() {
        let x = x.blocks[b].to_host_cell_major()?;
        let n = n.blocks[b].to_host_cell_major()?;
        let j = j.blocks[b].to_host_cell_major()?;
        let detj = detj.blocks[b].to_host_cell_major()?;
        let k = k.blocks[b].to_host_cell_major()?;
        assert_eq!((j.len(), detj.len(), k.len()), (x.len(), x.len(), x.len()));
        for (x, n) in x.iter().zip(&n) {
            let r = norm(x);
            let length = norm(n);
            assert!((length - 1.0).abs() <= tol, "p = {p}: |n| = {length}");
            let cos: f64 = x.iter().zip(n).map(|(a, b)| a.widen() * b.widen()).sum();
            assert!(
                cos > 0.0,
                "p = {p}: n · X = {cos}: the normal points inward"
            );
            radius = radius.max((r - 1.0).abs());
            angle = angle.max((cos / r).clamp(-1.0, 1.0).acos());
            points += 1;
        }
        assert!(detj.iter().all(|d| d.widen() > 0.0), "p = {p}: det J ≤ 0");
    }
    // ANCHOR: area
    let sphere = 4.0 * std::f64::consts::PI;
    let area = (mesh.measure(2 * p + 2)? - sphere).abs() / sphere;
    // ANCHOR_END: area
    println!("  {p:>2}  {points:>6}  {radius:>11.3e}  {angle:>15.3e}  {area:>10.3e}");
    Ok(Errors { radius, area })
}

/// The Euclidean length of `v`, in f64.
fn norm<F: Real>(v: &[F; 3]) -> f64 {
    v.iter().map(|c| c.widen() * c.widen()).sum::<f64>().sqrt()
}
