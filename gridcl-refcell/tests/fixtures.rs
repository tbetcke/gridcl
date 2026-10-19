//! gridcl-refcell against the fixtures exported from ndelement by gridcl-conformance
//! (docs/design.md, A.10): the workspace mirror of the conformance tests, needing neither
//! ndelement nor BLAS. Regenerate with
//! `cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures`.

use gridcl_dims::CellType;
use gridcl_refcell::{
    NodeFamily, lagrange_points, orientation_bits, sub_entities, sub_entity_types, tdim, vertices,
};
use serde_json::Value;

fn fixture(file: &str) -> Value {
    let path = format!("{}/tests/fixtures/{file}", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {path}: {e}"))
}

/// The cell type named as `format!("{cell:?}")` in the fixtures.
fn cell(name: &str) -> CellType {
    use CellType::*;
    [
        Point,
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ]
    .into_iter()
    .find(|c| format!("{c:?}") == name)
    .unwrap_or_else(|| panic!("unknown cell type {name}"))
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect()
}

fn usizes(v: &Value) -> Vec<usize> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect()
}

#[test]
fn tables_match_fixture() {
    let tables = fixture("tables.json");
    let tables = tables.as_object().unwrap();
    assert_eq!(tables.len(), 8);
    for (name, t) in tables {
        let c = cell(name);
        assert_eq!(tdim(c), t["tdim"].as_u64().unwrap() as usize, "{name}");
        let want: Vec<f64> = t["vertices"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(floats)
            .collect();
        let ours: Vec<f64> = vertices(c).iter().flat_map(|v| v.iter().copied()).collect();
        assert_eq!(ours, want, "vertices of {name}");
        for dim in 0..=3 {
            let want: Vec<Vec<usize>> = t["sub_entities"][dim]
                .as_array()
                .unwrap()
                .iter()
                .map(usizes)
                .collect();
            let ours: Vec<Vec<usize>> = sub_entities(c, dim).iter().map(|e| e.to_vec()).collect();
            assert_eq!(ours, want, "sub-entities of dimension {dim} of {name}");
            let want: Vec<CellType> = t["sub_entity_types"][dim]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| cell(n.as_str().unwrap()))
                .collect();
            assert_eq!(
                sub_entity_types(c, dim),
                want,
                "sub-entity types of dimension {dim} of {name}"
            );
        }
    }
}

/// Advances `p` to the next permutation in lexicographic order; false after the last one.
fn next_permutation(p: &mut [i64]) -> bool {
    let Some(i) = (1..p.len()).rev().find(|&i| p[i - 1] < p[i]) else {
        return false;
    };
    let j = (i..p.len()).rev().find(|&j| p[j] > p[i - 1]).unwrap();
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

#[test]
fn orientation_matches_fixture() {
    let fixture = fixture("orientation.json");
    let cells = fixture.as_object().unwrap();
    assert_eq!(cells.len(), 6, "the six cells with D_TOP >= 2");
    for (name, f) in cells {
        let c = cell(name);
        let nv = f["num_vertices"].as_u64().unwrap() as usize;
        assert_eq!(nv, vertices(c).len(), "{name}");
        let want = f["bits"].as_array().unwrap();
        // Every permutation of 0..V in lexicographic order.
        let mut ids: Vec<i64> = (0..nv as i64).collect();
        let mut k = 0;
        loop {
            assert_eq!(
                u64::from(orientation_bits(c, &ids)),
                want[k].as_u64().unwrap(),
                "{name} with origin ids {ids:?}"
            );
            k += 1;
            if !next_permutation(&mut ids) {
                break;
            }
        }
        assert_eq!(k, want.len(), "{name}: number of permutations");
    }
}

/// "Node by node": 1e-14 absolute per coordinate (docs/plans/M1b.md, question 10).
#[test]
fn lagrange_points_match_fixture() {
    let fixture = fixture("lagrange_points.json");
    let cells = fixture.as_object().unwrap();
    assert_eq!(cells.len(), 5);
    for (name, families) in cells {
        let c = cell(name);
        for (family_name, degrees) in families.as_object().unwrap() {
            let family = match family_name.as_str() {
                "Equispaced" => NodeFamily::Equispaced,
                "Gll" => NodeFamily::Gll,
                other => panic!("unknown node family {other}"),
            };
            for (degree, want) in degrees.as_object().unwrap() {
                let p: u32 = degree.parse().unwrap();
                let want = floats(want);
                let ours = lagrange_points(c, p, family).unwrap();
                assert_eq!(ours.len(), want.len(), "{name} {family_name} p = {p}");
                for (k, (a, b)) in ours.iter().zip(&want).enumerate() {
                    assert!(
                        (a - b).abs() <= 1e-14,
                        "{name} {family_name} p = {p}, entry {k}: {a} vs {b}"
                    );
                }
            }
        }
    }
}
