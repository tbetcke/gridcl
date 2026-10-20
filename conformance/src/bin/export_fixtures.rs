//! Exports ndelement's reference data to `gridcl-refcell/tests/fixtures/` (docs/design.md, A.10),
//! so that `cargo test --workspace` checks gridcl-refcell against it without ndelement or BLAS.
//!
//! Run after every ndelement pin bump, then commit the fixtures with the new pin:
//!
//! ```text
//! cargo run --manifest-path conformance/Cargo.toml --bin export-fixtures
//! ```
//!
//! - `tables.json`: vertices, sub-entities and sub-entity types of the eight cells (A.1–A.3);
//! - `orientation.json`: `compute_orientation` of every permutation of 0..V, in lexicographic
//!   order, for each cell with D_TOP ≥ 2 (A.6; docs/plans/M1b.md, question 5);
//! - `lagrange_points.json`: ndelement's interpolation points in DOF order for intervals,
//!   triangles, quadrilaterals, tetrahedra and hexahedra, p ≤ 4, equispaced and (tensor cells)
//!   GLL (A.5);
//! - `sources.json`: the pinned ndelement, Basix, DOLFINx and Symfem commits.
//!
//! The output is deterministic: a re-export at the same pin changes nothing.

use std::fs;
use std::path::{Path, PathBuf};

use gridcl_conformance::{CELLS, reference_cell_type};
use gridcl_dims::CellType;
use ndelement::ciarlet::{LagrangeVariant, lagrange};
use ndelement::orientation::compute_orientation;
use ndelement::reference_cell;
use ndelement::traits::MappedFiniteElement;
use ndelement::types::Continuity;
use serde_json::{Map, Value, json};

/// The pinned commits of the other sources (docs/design.md, A.10; ADR-0007). ndelement's is
/// read from Cargo.lock, so it follows the pin.
const BASIX: (&str, &str) = ("v0.11.0", "8ba83f732cd9d83cec643fa9059d49afb4482485");
const DOLFINX: (&str, &str) = ("main", "be8352e677303b686d6862b70923bab086b279ff");
const SYMFEM: (&str, &str) = ("v2025.12.0", "c37ffd6951a7fddd99353335fd5a511efdcf18fc");

fn name(cell: CellType) -> String {
    format!("{cell:?}")
}

fn tables() -> Value {
    let mut out = Map::new();
    for cell in CELLS {
        let nd = reference_cell_type(cell);
        let types = reference_cell::entity_types(nd);
        let counts = reference_cell::entity_counts(nd);
        let mut entities = vec![];
        let mut entity_types = vec![];
        for (dim, types) in types.iter().enumerate() {
            let list: Vec<Vec<usize>> = match dim {
                0 => (0..counts[0]).map(|v| vec![v]).collect(),
                1 => reference_cell::edges(nd),
                2 => reference_cell::faces(nd),
                _ => reference_cell::volumes(nd),
            };
            entities.push(json!(list));
            let names: Vec<String> = types.iter().map(|t| format!("{t:?}")).collect();
            entity_types.push(json!(names));
        }
        out.insert(
            name(cell),
            json!({
                "tdim": reference_cell::dim(nd),
                "vertices": reference_cell::vertices::<f64>(nd),
                "sub_entities": entities,
                "sub_entity_types": entity_types,
            }),
        );
    }
    Value::Object(out)
}

/// Advances `p` to the next permutation in lexicographic order; false after the last one.
fn next_permutation(p: &mut [usize]) -> bool {
    let Some(i) = (1..p.len()).rev().find(|&i| p[i - 1] < p[i]) else {
        return false;
    };
    let j = (i..p.len())
        .rev()
        .find(|&j| p[j] > p[i - 1])
        .expect("p[i] > p[i - 1]");
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

fn orientation() -> Value {
    let mut out = Map::new();
    for cell in CELLS.into_iter().filter(|&c| gridcl_refcell::tdim(c) >= 2) {
        let nd = reference_cell_type(cell);
        let nv = reference_cell::entity_counts(nd)[0];
        let mut ids: Vec<usize> = (0..nv).collect();
        let mut bits = vec![];
        loop {
            bits.push(compute_orientation(nd, &ids) as u32);
            if !next_permutation(&mut ids) {
                break;
            }
        }
        out.insert(name(cell), json!({ "num_vertices": nv, "bits": bits }));
    }
    Value::Object(out)
}

fn lagrange_points() -> Value {
    let mut out = Map::new();
    let cases = [
        (CellType::Interval, true),
        (CellType::Triangle, false),
        (CellType::Quadrilateral, true),
        (CellType::Tetrahedron, false),
        (CellType::Hexahedron, true),
    ];
    for (cell, tensor) in cases {
        let mut variants = vec![("Equispaced", LagrangeVariant::Equispaced)];
        if tensor {
            variants.push(("Gll", LagrangeVariant::Gll));
        }
        let mut per_family = Map::new();
        for (family, variant) in variants {
            let mut per_degree = Map::new();
            for degree in 1..=4usize {
                let element = lagrange::create::<f64, f64>(
                    reference_cell_type(cell),
                    degree,
                    Continuity::Standard,
                    variant,
                );
                let td = gridcl_refcell::tdim(cell);
                let mut points = vec![];
                for entities in element.interpolation_points() {
                    for block in entities {
                        for point in block.col_iter() {
                            points.extend((0..td).map(|j| point[[j]]));
                        }
                    }
                }
                per_degree.insert(degree.to_string(), json!(points));
            }
            per_family.insert(family.to_string(), Value::Object(per_degree));
        }
        out.insert(name(cell), Value::Object(per_family));
    }
    Value::Object(out)
}

/// ndelement's git revision, from the `source` line of its Cargo.lock entry.
fn ndelement_rev(manifest_dir: &Path) -> String {
    let lock = fs::read_to_string(manifest_dir.join("Cargo.lock")).expect("read Cargo.lock");
    let mut lines = lock.lines().skip_while(|l| *l != "name = \"ndelement\"");
    let source = lines
        .find(|l| l.starts_with("source = "))
        .expect("ndelement has a source line in Cargo.lock");
    source
        .rsplit('#')
        .next()
        .expect("a git source ends in #<rev>")
        .trim_end_matches('"')
        .to_string()
}

fn sources(manifest_dir: &Path) -> Value {
    let pin = |(tag, commit): (&str, &str), repo: &str| json!({ "repository": repo, "ref": tag, "commit": commit });
    json!({
        "ndelement": {
            "repository": "https://codeberg.org/nd-project/nd",
            "ref": "main",
            "commit": ndelement_rev(manifest_dir),
        },
        "basix": pin(BASIX, "https://github.com/FEniCS/basix"),
        "dolfinx": pin(DOLFINX, "https://github.com/FEniCS/dolfinx"),
        "symfem": pin(SYMFEM, "https://github.com/mscroggs/symfem"),
    })
}

fn write(dir: &Path, file: &str, value: &Value, pretty: bool) {
    let mut text = if pretty {
        serde_json::to_string_pretty(value)
    } else {
        serde_json::to_string(value)
    }
    .expect("serialise");
    text.push('\n');
    let path = dir.join(file);
    fs::write(&path, text).unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    println!("wrote {}", path.display());
}

fn main() {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let dir = manifest_dir.join("../gridcl-refcell/tests/fixtures");
    fs::create_dir_all(&dir).expect("create the fixture directory");
    write(&dir, "tables.json", &tables(), true);
    // Compact: the hexahedron alone has 8! = 40 320 words.
    write(&dir, "orientation.json", &orientation(), false);
    write(&dir, "lagrange_points.json", &lagrange_points(), false);
    write(&dir, "sources.json", &sources(&manifest_dir), true);
}
