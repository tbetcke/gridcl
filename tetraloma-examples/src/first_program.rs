//! A first program: a mesh, its geometry and a VTU file on one device (the book, "A first
//! program").
//!
//! The mesh is the rectangle [0, 2] × [0, 1], cut into 4 × 2 squares and each square into two
//! triangles with straight sides. `run` builds it from host arrays, computes the physical
//! points, Jacobians and Jacobian determinants at the points of a Gauss rule, integrates the
//! constant 1 over the mesh in two ways, and writes the mesh as a VTU file.

use std::path::Path;

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// The rectangle [0, `LX`] × [0, `LY`].
const LX: f64 = 2.0;
/// The height of the rectangle.
const LY: f64 = 1.0;
/// Squares along x.
const NX: usize = 4;
/// Squares along y.
const NY: usize = 2;

/// Builds the rectangle mesh in `F` on `client`, computes its geometric data and writes the
/// VTU file `rectangle.vtu` to `out_dir` (which must exist).
///
/// It prints the mesh's size, the data of the first quadrature point and the area, and checks
/// that det J equals the area of a square at every point (each triangle is half a square, and
/// the reference triangle has area 1/2), that det J is the determinant of J, that every
/// physical point lies in the rectangle, that the quadrature sum of det J and `measure` both
/// give the area 2, and that the written file is a VTK XML file.
///
/// # Errors
///
/// The errors of mesh creation, of the geometric data and of [`VtkWriter::write`]; `Io` if the
/// written file cannot be read back; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client, out_dir: &Path) -> tetraloma::Result<()> {
    // ANCHOR: mesh
    // Host arrays: node coordinates [node][2] in f64, and three node indices per triangle.
    let (coords, triangles) = rectangle();
    // Triangles with straight sides: Lagrange geometry of degree 1.
    let p1 = CoordinateElement::<D2> {
        cell: Cell2::Triangle,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    // A mesh of topological dimension 2 in the plane, with scalar F.
    let mesh: PlanarMesh<F> = MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(p1, &triangles, None)
        .build()?;

    let num_cells = mesh.cell_blocks()[0].index_map.num_owned();
    let num_nodes = mesh.node_index_map().num_owned();
    println!(
        "{num_cells} triangles, {num_nodes} nodes, on {} in {}",
        client.name(),
        std::any::type_name::<F>()
    );
    // ANCHOR_END: mesh
    assert_eq!(num_cells, 2 * NX * NY);
    assert_eq!(num_nodes, (NX + 1) * (NY + 1));
    // The origin is the centre of the bounding box; the coordinates are stored relative to it.
    assert_eq!(mesh.origin(), [LX / 2.0, LY / 2.0]);

    // ANCHOR: geometry
    // A Gauss rule on the triangle, exact for polynomials of degree 2.
    let rule = QuadRule::for_grid(&mesh, 2)?;
    // One field per cell block, computed and kept on the device.
    let x = mesh.physical_points(&rule)?;
    let j = mesh.jacobians(&rule)?;
    let detj = mesh.detj(&rule)?;

    // Copies to the host, for inspection: [cell][point] order.
    let x = x.blocks[0].to_host_cell_major()?;
    let j = j.blocks[0].to_host_cell_major()?;
    let detj = detj.blocks[0].to_host_cell_major()?;
    let weights = rule.get(Cell2::Triangle)?.weights();
    println!(
        "{} points per triangle; at the first: x = {:?}, J = {:?} (by columns), det J = {}",
        weights.len(),
        x[0].map(Real::widen),
        j[0].map(|column| column.map(Real::widen)),
        detj[0].widen()
    );

    // The integral of 1 over the mesh: the sum over cells and points of w_q · det J.
    let area: f64 = detj
        .chunks_exact(weights.len())
        .flat_map(|cell| cell.iter().zip(weights).map(|(d, w)| w * d.widen()))
        .sum();
    // The same integral on the device, finished in f64 on the host.
    let measure = mesh.measure(1)?;
    println!("area: {area} from det J and the weights, {measure} from measure");
    // ANCHOR_END: geometry

    let tol = 10.0 * crate::tol::<F>();
    let square = (LX / NX as f64) * (LY / NY as f64);
    assert_eq!(x.len(), num_cells * weights.len());
    assert_eq!((j.len(), detj.len()), (x.len(), x.len()));
    for ((x, j), d) in x.iter().zip(&j).zip(&detj) {
        let d = d.widen();
        assert!(
            (d - square).abs() <= tol * square,
            "det J = {d}, not {square}"
        );
        // J is stored by columns: j[0] = ∂x/∂ξ₁ and j[1] = ∂x/∂ξ₂.
        let det = j[0][0].widen() * j[1][1].widen() - j[1][0].widen() * j[0][1].widen();
        assert!(
            (det - d).abs() <= tol * square,
            "det of J = {det}, det J = {d}"
        );
        let [x0, x1] = x.map(Real::widen);
        assert!(
            (-tol..=LX + tol).contains(&x0) && (-tol..=LY + tol).contains(&x1),
            "the point ({x0}, {x1}) is outside the rectangle"
        );
    }
    for (name, value) in [("the quadrature sum", area), ("measure", measure)] {
        assert!(
            (value - LX * LY).abs() <= tol * LX * LY,
            "{name} gives the area {value}, not {}",
            LX * LY
        );
    }

    // ANCHOR: vtu
    let path = out_dir.join("rectangle.vtu");
    VtkWriter::vtu(&path).write(&mesh)?;
    // ANCHOR_END: vtu
    let bytes =
        std::fs::read(&path).map_err(|e| GridError::Io(format!("{}: {e}", path.display())))?;
    assert!(
        bytes.starts_with(b"<?xml") || bytes.starts_with(b"<VTKFile"),
        "{} is not a VTK XML file",
        path.display()
    );
    println!("wrote {} ({} bytes)", path.display(), bytes.len());
    Ok(())
}

// ANCHOR: rectangle
/// The nodes and triangles of the rectangle [0, `LX`] × [0, `LY`] in `NX` × `NY` squares: the
/// coordinates `[node][2]`, and three node indices per triangle, counter-clockwise. Node (i, j)
/// of the grid has index j (`NX` + 1) + i, and the square with lower-left corner (i, j) is cut
/// along its diagonal from (i, j) to (i + 1, j + 1).
fn rectangle() -> (Vec<f64>, Vec<u32>) {
    let node = |i: usize, j: usize| u32::try_from(j * (NX + 1) + i).expect("a small grid");
    let mut coords = Vec::with_capacity(2 * (NX + 1) * (NY + 1));
    for j in 0..=NY {
        for i in 0..=NX {
            coords.push(LX * i as f64 / NX as f64);
            coords.push(LY * j as f64 / NY as f64);
        }
    }
    let mut triangles = Vec::with_capacity(6 * NX * NY);
    for j in 0..NY {
        for i in 0..NX {
            let (a, b) = (node(i, j), node(i + 1, j));
            let (c, d) = (node(i, j + 1), node(i + 1, j + 1));
            triangles.extend([a, b, d, a, d, c]);
        }
    }
    (coords, triangles)
}
// ANCHOR_END: rectangle
