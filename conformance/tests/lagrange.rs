//! M1b acceptance: point sets equal ndelement's node by node for intervals, triangles,
//! quadrilaterals, tetrahedra and hexahedra, p ≤ 4, equispaced and (tensor cells) GLL (A.5).
//!
//! "Node by node": 1e-14 absolute per coordinate (docs/plans/M1b.md, question 10).

use gridcl_conformance::reference_cell_type;
use gridcl_dims::CellType;
use gridcl_refcell::{NodeFamily, lagrange_points, tdim};
use ndelement::ciarlet::{LagrangeVariant, lagrange};
use ndelement::traits::MappedFiniteElement;
use ndelement::types::Continuity;

const TOL: f64 = 1e-14;

/// ndelement's interpolation points concatenated by dimension, entity and point (its DOF
/// order), flat and node-major.
fn ndelement_points(cell: CellType, degree: u32, variant: LagrangeVariant) -> Vec<f64> {
    let element = lagrange::create::<f64, f64>(
        reference_cell_type(cell),
        degree as usize,
        Continuity::Standard,
        variant,
    );
    let td = tdim(cell);
    let mut out = Vec::new();
    for entities in element.interpolation_points() {
        for points in entities {
            for point in points.col_iter() {
                out.extend((0..td).map(|j| point[[j]]));
            }
        }
    }
    out
}

/// Stack for the test body. With OpenBLAS (the hosted Linux job), ndelement's element
/// construction overflowed the test harness's 2 MiB thread stack in debug builds; with
/// Accelerate 256 KiB suffice.
const STACK_BYTES: usize = 64 << 20;

#[test]
fn points_match_ndelement() {
    std::thread::Builder::new()
        .stack_size(STACK_BYTES)
        .spawn(check_points)
        .unwrap()
        .join()
        .unwrap();
}

fn check_points() {
    use CellType::*;
    let cases = [
        (Interval, true),
        (Triangle, false),
        (Quadrilateral, true),
        (Tetrahedron, false),
        (Hexahedron, true),
    ];
    for (cell, tensor) in cases {
        for degree in 1..=4 {
            let mut families = vec![(NodeFamily::Equispaced, LagrangeVariant::Equispaced)];
            if tensor {
                families.push((NodeFamily::Gll, LagrangeVariant::Gll));
            }
            for (family, variant) in families {
                let ours = lagrange_points(cell, degree, family).unwrap();
                let theirs = ndelement_points(cell, degree, variant);
                assert_eq!(ours.len(), theirs.len(), "{cell:?} {family:?} p = {degree}");
                let td = tdim(cell);
                for (k, (a, b)) in ours.iter().zip(&theirs).enumerate() {
                    assert!(
                        (a - b).abs() <= TOL,
                        "{cell:?} {family:?} p = {degree}, node {}, coordinate {}: {a} vs {b}",
                        k / td,
                        k % td
                    );
                }
            }
        }
    }
}
