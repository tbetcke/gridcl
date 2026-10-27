//! M5 acceptance tests (docs/design.md, "M5 VTK output", as amended by ADR-0024): VTK's
//! reader reads every fixture, VTK integrates affine P2 and P3 meshes to `measure` and
//! evaluates curved cells like `physical_points`, meshio reads identical raw arrays, and P2
//! pyramids are written linear with one warning.
//!
//! Every test checks the written files in Rust through `parse_vtu` first; the Python oracles
//! (`scripts/oracles/vtk_oracle.py`, pinned VTK and meshio) follow unless
//! `GRIDCL_SKIP_PYTHON=1` (ADR-0024, item 11).

mod common;

use std::collections::BTreeMap;
use std::path::PathBuf;

use common::python::Oracle;
use common::vtk::{
    Case, Expected, TempDir, build_any, capture_warnings, check_file, corpus, data_cases,
    decode_array, expected_geometry, expected_type, floats, ints, my_warnings, parse_file,
    reference_cells, same_bits, variant, write_case,
};
use common::{build, client, name};
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_testing::HostMesh;
use serde_json::Value;

/// The corpus of `F`, written and checked in Rust, plus a PVTU with data: (expected contents
/// of every VTU, the PVTU path and the expected contents of its piece).
fn write_corpus<F: Real>(client: &Client, dir: &TempDir) -> (Vec<Expected>, PathBuf, Expected) {
    let mut out = Vec::new();
    for case in corpus() {
        let e = write_case::<F>(client, &case, dir);
        check_file(&e, &parse_file(&e.path));
        out.push(e);
    }

    // The PVTU skeleton: one piece on one rank, with data.
    let h = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 2);
    let any = build_any::<F>(client, &h, 2);
    let (points, cells) = with_mesh!(any.as_ref(), m => expected_geometry(m));
    let data = common::vtk::DataSet::new(client, points.len() / 3, cells.len());
    let stem = dir.join(&format!("{}-pvtu", name::<F>()));
    data.attach(VtkWriter::pvtu(&stem))
        .write_any(any.as_ref())
        .unwrap_or_else(|e| panic!("pvtu ({}): {e}", name::<F>()));
    let pvtu = stem.with_extension("pvtu");
    let piece = Expected {
        what: format!("pvtu piece ({})", name::<F>()),
        path: dir.join(&format!("{}-pvtu_0.vtu", name::<F>())),
        tdim: 2,
        gdim: 2,
        lagrange: true,
        f32_points: size_of::<F>() == 4,
        points,
        cells,
        point_data: data.point,
        cell_data: data.cell,
    };
    assert!(pvtu.exists(), "{}: not written", pvtu.display());
    check_file(&piece, &parse_file(&piece.path));
    (out, pvtu, piece)
}

/// `r` (an oracle `read` result) has no errors or warnings, and the counts, types, bounds and
/// arrays of `e`.
fn check_read(e: &Expected, r: &Value) {
    let what = &e.what;
    assert_eq!(r["errors"], Value::Array(vec![]), "{what}: VTK errors");
    assert_eq!(r["warnings"], Value::Array(vec![]), "{what}: VTK warnings");
    assert_eq!(
        r["num_points"].as_u64(),
        Some(e.num_points() as u64),
        "{what}: points"
    );
    assert_eq!(
        r["num_cells"].as_u64(),
        Some(e.cells.len() as u64),
        "{what}: cells"
    );
    let types: Vec<u8> = ints(&r["types"]).iter().map(|&t| t as u8).collect();
    assert_eq!(types, e.types(), "{what}: cell types as VTK reads them");
    let bounds = floats(&r["bounds"]);
    assert_eq!(bounds, e.bounds().to_vec(), "{what}: bounds");
    for (section, want) in [("point_data", &e.point_data), ("cell_data", &e.cell_data)] {
        let got: BTreeMap<String, (u64, String)> = r[section]
            .as_array()
            .expect("arrays")
            .iter()
            .map(|a| {
                (
                    a["name"].as_str().expect("name").to_owned(),
                    (
                        a["components"].as_u64().expect("components"),
                        a["dtype"].as_str().expect("dtype").to_owned(),
                    ),
                )
            })
            .collect();
        let want: BTreeMap<String, (u64, String)> = want
            .iter()
            .map(|(n, a)| {
                (
                    n.clone(),
                    (a.components as u64, variant(&a.values).to_owned()),
                )
            })
            .collect();
        assert_eq!(got, want, "{what}: {section} as VTK reads it");
    }
}

/// VTK's Python reader reads every fixture: the corpus written by this test (every cell type
/// at p = 1–4, the pyramid fallback, all six (D_TOP, D_GEO), the mixed box, every encoding,
/// data of every scalar type, the PVTU) and the committed VTK-written reference cells.
#[test]
fn vtk_reads_every_fixture() {
    let client = client();
    let dir = TempDir::new("reads");
    let (mut files, mut pvtus) = (Vec::new(), Vec::new());
    let (f, p, piece) = write_corpus::<f32>(&client, &dir);
    files.extend(f);
    pvtus.push((p, piece));
    if gridcl_testing::supports::<f64>(&client) {
        let (f, p, piece) = write_corpus::<f64>(&client, &dir);
        files.extend(f);
        pvtus.push((p, piece));
    }
    let references = reference_cells();
    let mut want_refs: Vec<(u8, u32)> = (68..=73)
        .flat_map(|t| (1..=6).map(move |p| (t, p)))
        .collect();
    let mut have_refs: Vec<(u8, u32)> = references.iter().map(|c| (c.vtk_type, c.degree)).collect();
    want_refs.sort_unstable();
    have_refs.sort_unstable();
    assert_eq!(
        have_refs, want_refs,
        "VTK-written reference cells: types 68-73 at p = 1-6"
    );

    let Some(oracle) = Oracle::find("vtk_reads_every_fixture") else {
        return;
    };
    let mut paths: Vec<PathBuf> = files.iter().map(|e| e.path.clone()).collect();
    paths.extend(pvtus.iter().map(|(p, _)| p.clone()));
    paths.extend(references.iter().map(|c| c.path()));
    let results = oracle.run_files("read", &paths);
    let (corpus_results, rest) = results.split_at(files.len());
    let (pvtu_results, ref_results) = rest.split_at(pvtus.len());
    for (e, r) in files.iter().zip(corpus_results) {
        check_read(e, r);
    }
    for ((path, piece), r) in pvtus.iter().zip(pvtu_results) {
        let mut e = piece.clone();
        e.what = format!("{}", path.display());
        check_read(&e, r);
    }
    for (c, r) in references.iter().zip(ref_results) {
        assert_eq!(r["errors"], Value::Array(vec![]), "{}: VTK errors", c.file);
        assert_eq!(r["num_cells"].as_u64(), Some(1), "{}", c.file);
        assert_eq!(
            r["num_points"].as_u64(),
            Some(c.num_points as u64),
            "{}",
            c.file
        );
        assert_eq!(ints(&r["types"]), [i64::from(c.vtk_type)], "{}", c.file);
    }
}

/// A shear-and-scale map with det > 0, and a shift away from the origin.
fn affine3(x: &[f64]) -> Vec<f64> {
    vec![
        1.5 * x[0] + 0.3 * x[1] - 0.2 * x[2] + 2.0,
        -0.4 * x[0] + 0.9 * x[1] + 0.1 * x[2] - 1.0,
        0.25 * x[0] + 0.2 * x[1] + 1.1 * x[2] + 0.5,
    ]
}

/// A planar shear-and-scale map with det > 0.
fn affine2(x: &[f64]) -> Vec<f64> {
    vec![
        1.2 * x[0] + 0.5 * x[1] - 3.0,
        -0.3 * x[0] + 0.8 * x[1] + 1.0,
    ]
}

/// The unit square onto a tilted plane in 3D.
fn tilted(x: &[f64]) -> Vec<f64> {
    vec![
        0.8 * x[0] - 0.2 * x[1] + 1.0,
        0.6 * x[0] + 0.5 * x[1],
        0.3 * x[0] + 0.7 * x[1] - 2.0,
    ]
}

/// The affine meshes of the integration bullet at degree `p`: affine images of the boxes of
/// every cell type of D_TOP 2 and 3 and of the mixed box, and planar triangle and quadrilateral
/// meshes in 2D and 3D.
fn affine_meshes(p: u32) -> Vec<(String, HostMesh, usize)> {
    use gridcl_testing::*;
    let mut out = Vec::new();
    for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
        out.push((
            format!("{cell:?}-2d-p{p}"),
            square_box(cell, 2, p).mapped(2, affine2),
            2,
        ));
        out.push((
            format!("{cell:?}-3d-p{p}"),
            square_box(cell, 2, p).mapped(3, tilted),
            2,
        ));
    }
    for cell in [
        Cell3::Tetrahedron,
        Cell3::Hexahedron,
        Cell3::Prism,
        Cell3::Pyramid,
    ] {
        out.push((
            format!("{cell:?}-p{p}"),
            cube_box(cell, 2, p).mapped(3, affine3),
            3,
        ));
    }
    out.push((format!("mixed-p{p}"), mixed_box(2, p).mapped(3, affine3), 3));
    out
}

/// The integrated area or volume of affine P2 and P3 meshes (`vtkIntegrateAttributes`) equals
/// `measure` to 10^−10 (f64; ADR-0024, item 6).
#[test]
fn vtk_integrated_measure_equals_measure() {
    let client = client();
    if !gridcl_testing::supports::<f64>(&client) {
        eprintln!("vtk_integrated_measure_equals_measure: the device has no f64; skipped");
        return;
    }
    let dir = TempDir::new("integrate");
    let mut written = Vec::new();
    for p in [2, 3] {
        for (what, h, tdim) in affine_meshes(p) {
            let any = build_any::<f64>(&client, &h, tdim);
            let measure = with_mesh!(any.as_ref(), m => m.measure(2 * p))
                .unwrap_or_else(|e| panic!("{what}: measure: {e}"));
            let path = dir.join(&format!("{what}.vtu"));
            VtkWriter::vtu(&path)
                .write_any(any.as_ref())
                .unwrap_or_else(|e| panic!("{what}: write: {e}"));
            parse_file(&path);
            written.push((what, path, tdim, measure));
        }
    }
    let Some(oracle) = Oracle::find("vtk_integrated_measure_equals_measure") else {
        return;
    };
    let paths: Vec<PathBuf> = written.iter().map(|w| w.1.clone()).collect();
    let results = oracle.run_files("integrate", &paths);
    for ((what, _, tdim, measure), r) in written.iter().zip(&results) {
        assert_eq!(r["errors"], Value::Array(vec![]), "{what}: VTK errors");
        let key = if *tdim == 2 { "area" } else { "volume" };
        let vtk = r[key]
            .as_f64()
            .unwrap_or_else(|| panic!("{what}: no {key}"));
        let err = (vtk - measure).abs() / measure.abs();
        assert!(
            err <= 1e-10,
            "{what}: VTK {key} {vtk} vs measure {measure} (relative error {err:e})"
        );
    }
}

/// The curved meshes of the evaluation bullet at degree `p`.
fn curved_meshes(p: u32) -> Vec<(String, HostMesh, usize)> {
    use gridcl_testing::*;
    vec![
        (format!("sphere-p{p}"), sphere(1, p), 2),
        (format!("mixed-surface-p{p}"), mixed_surface(1, p), 2),
        (format!("ball-p{p}"), ball(0, p), 3),
        (
            format!("tetrahedra-p{p}"),
            cube_box(Cell3::Tetrahedron, 2, p).perturbed(0.05),
            3,
        ),
        (
            format!("prisms-p{p}"),
            cube_box(Cell3::Prism, 2, p).perturbed(0.05),
            3,
        ),
    ]
}

/// gridcl's physical points of every cell at the Gauss points of degree 4, `[cell][q][3]`
/// (zero-padded) in flat cell order, and the oracle request for the same parametric points.
fn gauss_points<TD, GD>(m: &Mesh<f64, TD, GD>, p: u32) -> (Vec<Vec<f64>>, Value)
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let td = TD::N;
    let gd = GD::N;
    let rule = QuadRule::for_grid(m, 4).expect("rule");
    let fields = m.physical_points(&rule).expect("physical_points");
    let flat = common::flat_f64(&fields);
    let mut cells = Vec::new();
    let mut request = serde_json::Map::new();
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let ps = rule.get(block.cell).expect("point set");
        let nq = ps.len();
        let mut pcoords = Vec::with_capacity(nq * 3);
        for q in ps.points().chunks_exact(td) {
            pcoords.extend((0..3).map(|d| if d < td { q[d] } else { 0.0 }));
        }
        let cell: CellType = block.cell.into();
        request.insert(
            expected_type(cell, p, true).to_string(),
            serde_json::json!(pcoords),
        );
        for x in flat[b].chunks_exact(nq * gd) {
            cells.push(
                x.chunks_exact(gd)
                    .flat_map(|y| (0..3).map(move |g| if g < gd { y[g] } else { 0.0 }))
                    .collect(),
            );
        }
    }
    (cells, Value::Object(request))
}

/// On curved P2 and P3 meshes, VTK's `EvaluateLocation` at Gauss points equals
/// `physical_points` to 10^−10 (f64; ADR-0024, item 6).
#[test]
fn vtk_evaluates_curved_cells_like_gridcl() {
    let client = client();
    if !gridcl_testing::supports::<f64>(&client) {
        eprintln!("vtk_evaluates_curved_cells_like_gridcl: the device has no f64; skipped");
        return;
    }
    let dir = TempDir::new("evaluate");
    let mut written = Vec::new();
    let mut request = Vec::new();
    for p in [2, 3] {
        for (what, h, tdim) in curved_meshes(p) {
            let path = dir.join(&format!("{what}.vtu"));
            let (cells, points) = match tdim {
                2 => {
                    let m = build::<f64, D2, D3>(&h, &client);
                    VtkWriter::vtu(&path)
                        .write(&m)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    gauss_points(&m, p)
                }
                _ => {
                    let m = build::<f64, D3, D3>(&h, &client);
                    VtkWriter::vtu(&path)
                        .write(&m)
                        .unwrap_or_else(|e| panic!("{what}: {e}"));
                    gauss_points(&m, p)
                }
            };
            parse_file(&path);
            request.push(serde_json::json!({ "file": path, "points": points }));
            written.push((what, cells));
        }
    }
    let Some(oracle) = Oracle::find("vtk_evaluates_curved_cells_like_gridcl") else {
        return;
    };
    let request_path = dir.join("evaluate.json");
    std::fs::write(&request_path, serde_json::to_vec(&request).expect("JSON")).expect("write");
    let out = oracle.run("evaluate", &[request_path.into_os_string()]);
    let results = out.as_array().expect("a list");
    assert_eq!(results.len(), written.len(), "one result per file");
    for ((what, cells), r) in written.iter().zip(results) {
        assert_eq!(r["errors"], Value::Array(vec![]), "{what}: VTK errors");
        let vtk = r["cells"].as_array().expect("cells");
        assert_eq!(vtk.len(), cells.len(), "{what}: cells");
        let scale = cells.iter().flatten().fold(1.0f64, |a, x| a.max(x.abs()));
        for (c, (v, want)) in vtk.iter().zip(cells).enumerate() {
            let got = floats(v);
            assert_eq!(got.len(), want.len(), "{what}: cell {c}: points");
            for (i, (a, b)) in got.iter().zip(want).enumerate() {
                assert!(
                    (a - b).abs() <= 1e-10 * scale,
                    "{what}: cell {c}, coordinate {i}: VTK {a} vs gridcl {b}"
                );
            }
        }
    }
}

/// The files of the meshio test for `F`: the data meshes in every encoding, and corpus meshes
/// without data.
fn meshio_files<F: Real>(client: &Client, dir: &TempDir) -> Vec<Expected> {
    let mut cases: Vec<Case> = data_cases();
    let plain = common::vtk::corpus_meshes();
    for (i, name) in [
        "Hexahedron-p3",
        "Prism-p2",
        "Pyramid-p2",
        "curve3-p3",
        "mixed-p2",
    ]
    .iter()
    .enumerate()
    {
        let mut c = plain
            .iter()
            .find(|c| c.name == *name)
            .unwrap_or_else(|| panic!("no corpus mesh {name}"))
            .clone();
        c.encoding = common::vtk::ENCODINGS[i % 4];
        cases.push(c);
    }
    cases
        .iter()
        .map(|c| {
            let e = write_case::<F>(client, c, dir);
            check_file(&e, &parse_file(&e.path));
            e
        })
        .collect()
}

/// meshio reads identical raw arrays: points, connectivity, offsets, types and every point and
/// cell array are bitwise equal to `parse_vtu` of the same file, and the data arrays equal the
/// uploaded buffers.
#[test]
fn meshio_reads_identical_raw_arrays() {
    let client = client();
    let dir = TempDir::new("meshio");
    let mut files = meshio_files::<f32>(&client, &dir);
    if gridcl_testing::supports::<f64>(&client) {
        files.extend(meshio_files::<f64>(&client, &dir));
    }
    let Some(oracle) = Oracle::find("meshio_reads_identical_raw_arrays") else {
        return;
    };
    let paths: Vec<PathBuf> = files.iter().map(|e| e.path.clone()).collect();
    let results = oracle.run_files("meshio", &paths);
    for (e, r) in files.iter().zip(&results) {
        let what = &e.what;
        let f = parse_file(&e.path);
        let (points, shape) = decode_array(&r["points"]);
        assert!(same_bits(&points, &f.points), "{what}: points");
        assert_eq!(shape, [e.num_points(), 3], "{what}: points shape");
        assert_eq!(
            ints(&r["connectivity"]),
            f.connectivity,
            "{what}: connectivity"
        );
        assert_eq!(ints(&r["offsets"]), f.offsets, "{what}: offsets");
        let types: Vec<u8> = ints(&r["types"]).iter().map(|&t| t as u8).collect();
        assert_eq!(types, f.types, "{what}: types");
        for (section, parsed) in [("point_data", &f.point_data), ("cell_data", &f.cell_data)] {
            let got = r[section].as_object().expect("arrays");
            assert_eq!(got.len(), parsed.len(), "{what}: {section} arrays");
            for a in parsed {
                let (values, shape) = decode_array(&got[&a.name]);
                assert!(
                    same_bits(&values, &a.values),
                    "{what}: {section} {:?}",
                    a.name
                );
                let components = if shape.len() == 1 { 1 } else { shape[1] };
                assert_eq!(components, a.components, "{what}: {section} {:?}", a.name);
            }
        }
    }
}

/// A P2 pyramid mesh is written as linear cells with one warning: `cube_box(Pyramid, n, 2)` and
/// the P2 mixed box give type 14 for every pyramid and exactly one `log::warn!` per file; linear
/// output by request warns not.
#[test]
fn p2_pyramids_written_linear_with_one_warning() {
    capture_warnings();
    let client = client();
    let dir = TempDir::new("pyramids");
    let mut written = Vec::new();
    let mut run = |f64s: bool| {
        let meshes = [
            (
                "pyramids-p2",
                gridcl_testing::cube_box(Cell3::Pyramid, 2, 2),
            ),
            ("mixed-p2", gridcl_testing::mixed_box(2, 2)),
            (
                "pyramids-p3",
                gridcl_testing::cube_box(Cell3::Pyramid, 1, 3),
            ),
        ];
        for (n, h) in meshes {
            for lagrange in [true, false] {
                let case = Case {
                    name: format!("{n}-{lagrange}"),
                    host: h.clone(),
                    tdim: 3,
                    lagrange,
                    encoding: (false, false),
                    data: false,
                };
                let before = my_warnings().len();
                let e = if f64s {
                    write_case::<f64>(&client, &case, &dir)
                } else {
                    write_case::<f32>(&client, &case, &dir)
                };
                let warnings = my_warnings();
                let new = &warnings[before..];
                let pyramids = h
                    .blocks
                    .iter()
                    .filter(|b| b.cell == CellType::Pyramid)
                    .map(|b| b.num_cells())
                    .sum::<usize>();
                if lagrange {
                    assert_eq!(new.len(), 1, "{}: one warning per file: {new:?}", e.what);
                    assert!(
                        new[0].contains(&pyramids.to_string()),
                        "{}: the warning names the number of pyramids: {}",
                        e.what,
                        new[0]
                    );
                } else {
                    assert!(
                        new.is_empty(),
                        "{}: no warning for linear output: {new:?}",
                        e.what
                    );
                }
                let f = parse_file(&e.path);
                check_file(&e, &f);
                assert_eq!(
                    f.types.iter().filter(|&&t| t == 14).count(),
                    pyramids,
                    "{}: every pyramid is type 14",
                    e.what
                );
                written.push(e);
            }
        }
    };
    run(false);
    if gridcl_testing::supports::<f64>(&client) {
        run(true);
    }
    let Some(oracle) = Oracle::find("p2_pyramids_written_linear_with_one_warning") else {
        return;
    };
    let paths: Vec<PathBuf> = written.iter().map(|e| e.path.clone()).collect();
    let results = oracle.run_files("read", &paths);
    for (e, r) in written.iter().zip(&results) {
        check_read(e, r);
        assert!(
            !ints(&r["types"]).contains(&74),
            "{}: no VTK_LAGRANGE_PYRAMID",
            e.what
        );
    }
}
