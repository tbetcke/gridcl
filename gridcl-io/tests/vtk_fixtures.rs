//! The VTK-written reference cells (`tests/fixtures/vtk/`, written by
//! `scripts/fixtures/vtk/generate.py` with VTK 9.7.1) against gridcl's VTK node maps
//! (docs/design.md, "High-order node orderings / Checks"; ADR-0008).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use gridcl_dims::CellType;
use gridcl_io::vtk::{DataValues, parse_vtu};
use gridcl_refcell::{NodeFamily, lagrange_points, tdim, vtk_to_internal};
use serde_json::Value;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/vtk")
}

/// The cell of a VTK Lagrange type (A.8).
fn lagrange_cell(vtk_type: u64) -> CellType {
    match vtk_type {
        68 => CellType::Interval,
        69 => CellType::Triangle,
        70 => CellType::Quadrilateral,
        71 => CellType::Tetrahedron,
        72 => CellType::Hexahedron,
        73 => CellType::Prism,
        _ => panic!("VTK type {vtk_type} is not a Lagrange type"),
    }
}

/// For every VTK-written reference cell (every Lagrange type at p = 1–6, among them the
/// prisms at p = 3, 5 and 6 of ADR-0008), `lagrange_points(cell, p)[map[k]]` equals the
/// file's k-th point to 10^−14, with `map = vtk_to_internal(cell, p)`.
#[test]
fn vtk_written_cells_match_vtk_to_internal() {
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(fixtures_dir().join("manifest.json")).expect("manifest.json"),
    )
    .expect("manifest.json is JSON");
    assert_eq!(manifest["vtk_version"], "9.7.1");
    let cells = manifest["cells"].as_array().expect("cells");
    let mut seen = BTreeSet::new();
    for entry in cells {
        let name = entry["file"].as_str().expect("file");
        let vtk_type = entry["vtk_type"].as_u64().expect("vtk_type");
        let degree = u32::try_from(entry["degree"].as_u64().expect("degree")).unwrap();
        let n = usize::try_from(entry["num_points"].as_u64().expect("num_points")).unwrap();
        assert!(seen.insert((vtk_type, degree)), "{name}: listed twice");

        let bytes = std::fs::read(fixtures_dir().join(name)).expect(name);
        let file = parse_vtu(&bytes).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(file.types, [vtk_type as u8], "{name}");
        assert_eq!(file.offsets, [n as i64], "{name}");
        assert_eq!(
            file.connectivity,
            (0..n as i64).collect::<Vec<_>>(),
            "{name}"
        );
        let DataValues::F64(points) = &file.points else {
            panic!("{name}: points are not Float64");
        };
        assert_eq!(points.len(), 3 * n, "{name}");

        let cell = lagrange_cell(vtk_type);
        let td = tdim(cell);
        let map = vtk_to_internal(cell, degree).unwrap();
        let reference = lagrange_points(cell, degree, NodeFamily::Equispaced).unwrap();
        assert_eq!(map.len(), n, "{name}: node count");
        for (k, &i) in map.iter().enumerate() {
            let expected = &reference[i * td..(i + 1) * td];
            let found = &points[3 * k..3 * k + 3];
            for j in 0..3 {
                let e = expected.get(j).copied().unwrap_or(0.0);
                assert!(
                    (found[j] - e).abs() <= 1e-14,
                    "{name}: VTK node {k} is {found:?}, internal node {i} is {expected:?}"
                );
            }
        }
    }
    let expected: BTreeSet<(u64, u32)> = (68..=73)
        .flat_map(|t| (1..=6).map(move |p| (t, p)))
        .collect();
    assert_eq!(seen, expected, "the manifest lists every type at p = 1–6");
}
