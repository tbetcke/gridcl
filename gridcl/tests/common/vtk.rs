//! Helpers of the M5 acceptance tests (ADR-0024, item 12): the output corpus (meshes,
//! encodings, data arrays), writing it, the Rust-side checks of every written file through
//! `parse_vtu`, the VTK-written reference cells, oracle array decoding and a capturing logger.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::thread::ThreadId;

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_io::vtk::{DataArray, DataValues, VtuFile, parse_vtu};
use gridcl_testing::HostMesh;
use serde_json::Value;

use super::{build, lift_curve, name};

// ---------------------------------------------------------------------------------------------
// Expected cell types (A.8), written out independently of `vtk_cell_type`.

/// The VTK type gridcl writes for `cell` of degree `p` (A.8; pyramids p ≥ 2 → 14).
pub fn expected_type(cell: CellType, p: u32, lagrange: bool) -> u8 {
    let linear = match cell {
        CellType::Point => 1,
        CellType::Interval => 3,
        CellType::Triangle => 5,
        CellType::Quadrilateral => 9,
        CellType::Tetrahedron => 10,
        CellType::Hexahedron => 12,
        CellType::Prism => 13,
        CellType::Pyramid => 14,
    };
    if p == 1 || !lagrange || cell == CellType::Pyramid || cell == CellType::Point {
        return linear;
    }
    match cell {
        CellType::Interval => 68,
        CellType::Triangle => 69,
        CellType::Quadrilateral => 70,
        CellType::Tetrahedron => 71,
        CellType::Hexahedron => 72,
        CellType::Prism => 73,
        _ => unreachable!(),
    }
}

/// Whether `cell` of degree `p` is written from its vertices only.
pub fn written_linear(cell: CellType, p: u32, lagrange: bool) -> bool {
    p == 1 || !lagrange || cell == CellType::Pyramid
}

// ---------------------------------------------------------------------------------------------
// Encodings and temporary directories.

/// The four encodings: (base64, zlib).
pub const ENCODINGS: [(bool, bool); 4] =
    [(false, false), (false, true), (true, false), (true, true)];

/// A short name of an encoding.
pub fn encoding_name((base64, zlib): (bool, bool)) -> &'static str {
    match (base64, zlib) {
        (false, false) => "raw",
        (false, true) => "raw-zlib",
        (true, false) => "b64",
        (true, true) => "b64-zlib",
    }
}

/// A directory removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    /// A fresh directory for `test`, unique per process.
    pub fn new(test: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("gridcl-m5-{}-{test}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display()));
        TempDir(dir)
    }

    /// `name` inside the directory.
    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

// ---------------------------------------------------------------------------------------------
// Meshes.

/// The erased mesh of `h` with D_TOP `tdim`.
pub fn build_any<F: Real>(client: &Client, h: &HostMesh, tdim: usize) -> AnyMesh<F> {
    match (tdim, h.gdim) {
        (1, 1) => AnyMesh::Curve1(build::<F, D1, D1>(h, client)),
        (1, 2) => AnyMesh::Curve2(build::<F, D1, D2>(h, client)),
        (1, 3) => AnyMesh::Curve3(build::<F, D1, D3>(h, client)),
        (2, 2) => AnyMesh::Planar(build::<F, D2, D2>(h, client)),
        (2, 3) => AnyMesh::Surface(build::<F, D2, D3>(h, client)),
        (3, 3) => AnyMesh::Volume(build::<F, D3, D3>(h, client)),
        other => panic!("no mesh of (D_TOP, D_GEO) = {other:?}"),
    }
}

/// One mesh of the corpus.
#[derive(Clone, Debug)]
pub struct Case {
    pub name: String,
    pub host: HostMesh,
    pub tdim: usize,
    pub lagrange: bool,
    /// (base64, zlib).
    pub encoding: (bool, bool),
    /// Whether point and cell data of every scalar type is attached.
    pub data: bool,
}

impl Case {
    fn new(name: String, host: HostMesh, tdim: usize) -> Case {
        Case {
            name,
            host,
            tdim,
            lagrange: true,
            encoding: (false, false),
            data: false,
        }
    }
}

/// The meshes of the corpus (ADR-0024, item 12), without encodings: every cell type at
/// p = 1–4 (pyramids at p = 1–3, the fallback at 2 and 3), all six (D_TOP, D_GEO), the mixed
/// box, and linear output of high-order cells.
pub fn corpus_meshes() -> Vec<Case> {
    use gridcl_testing::*;
    let mut out = Vec::new();
    for p in 1..=4 {
        out.push(Case::new(format!("interval-p{p}"), interval_box(2, p), 1));
        for cell in [Cell2::Triangle, Cell2::Quadrilateral] {
            out.push(Case::new(
                format!("{cell:?}-p{p}"),
                square_box(cell, 2, p),
                2,
            ));
        }
        for cell in [
            Cell3::Tetrahedron,
            Cell3::Hexahedron,
            Cell3::Prism,
            Cell3::Pyramid,
        ] {
            if cell == Cell3::Pyramid && p == 4 {
                continue;
            }
            out.push(Case::new(format!("{cell:?}-p{p}"), cube_box(cell, 1, p), 3));
        }
    }
    for p in [2, 3] {
        out.push(Case::new(
            format!("curve2-p{p}"),
            lift_curve(&interval_box(3, p), 2),
            1,
        ));
        out.push(Case::new(
            format!("curve3-p{p}"),
            lift_curve(&interval_box(3, p), 3),
            1,
        ));
    }
    out.push(Case::new("sphere-p2".into(), sphere(0, 2), 2));
    out.push(Case::new("mixed-surface-p3".into(), mixed_surface(0, 3), 2));
    out.push(Case::new(
        "tilted-quadrilateral-p2".into(),
        square_box(Cell2::Quadrilateral, 2, 2)
            .mapped(3, |x| vec![x[0], x[1], 0.5 * x[0] - 0.25 * x[1]]),
        2,
    ));
    for p in [1, 2] {
        out.push(Case::new(format!("mixed-p{p}"), mixed_box(2, p), 3));
    }
    for (name, h, tdim) in [
        (
            "linear-Tetrahedron-p2",
            cube_box(Cell3::Tetrahedron, 1, 2),
            3,
        ),
        ("linear-Hexahedron-p3", cube_box(Cell3::Hexahedron, 1, 3), 3),
        ("linear-Triangle-p3", square_box(Cell2::Triangle, 2, 3), 2),
    ] {
        let mut c = Case::new(name.into(), h, tdim);
        c.lagrange = false;
        out.push(c);
    }
    out
}

/// The corpus: [`corpus_meshes`] with the encodings in turn, plus the data meshes (a P2 mixed
/// box and a P3 triangle surface with data of every scalar type) in every encoding.
pub fn corpus() -> Vec<Case> {
    let mut out: Vec<Case> = corpus_meshes()
        .into_iter()
        .enumerate()
        .map(|(i, mut c)| {
            c.encoding = ENCODINGS[i % ENCODINGS.len()];
            c
        })
        .collect();
    out.extend(data_cases());
    out
}

/// The meshes with data, in every encoding.
pub fn data_cases() -> Vec<Case> {
    use gridcl_testing::*;
    let meshes = [
        ("data-mixed-p2", mixed_box(2, 2), 3),
        ("data-sphere-p3", sphere(0, 3), 2),
    ];
    let mut out = Vec::new();
    for (name, h, tdim) in meshes {
        for e in ENCODINGS {
            let mut c = Case::new(name.into(), h.clone(), tdim);
            c.encoding = e;
            c.data = true;
            out.push(c);
        }
    }
    out
}

// ---------------------------------------------------------------------------------------------
// Data arrays.

/// Point and cell data of every scalar type with 1, 3 and 9 components, on the device and as
/// the expected host arrays.
pub struct DataSet {
    f32s: Vec<(bool, String, DeviceBuffer<f32>, usize)>,
    f64s: Vec<(bool, String, DeviceBuffer<f64>, usize)>,
    i32s: Vec<(bool, String, DeviceBuffer<i32>, usize)>,
    u32s: Vec<(bool, String, DeviceBuffer<u32>, usize)>,
    /// The expected point data, by name.
    pub point: BTreeMap<String, DataArray>,
    /// The expected cell data, by name.
    pub cell: BTreeMap<String, DataArray>,
}

impl DataSet {
    /// Data for `num_points` points and `num_cells` cells; f64 arrays only if the device has f64.
    pub fn new(client: &Client, num_points: usize, num_cells: usize) -> DataSet {
        let with_f64 = gridcl_testing::supports::<f64>(client);
        let mut rng = super::SplitMix::new(0x5eed_0005);
        let mut set = DataSet {
            f32s: Vec::new(),
            f64s: Vec::new(),
            i32s: Vec::new(),
            u32s: Vec::new(),
            point: BTreeMap::new(),
            cell: BTreeMap::new(),
        };
        for point in [true, false] {
            let n = if point { num_points } else { num_cells };
            let section = if point { "point" } else { "cell" };
            for c in [1, 3, 9] {
                let len = n * c;
                // Names with XML special characters and spaces.
                let names = [
                    format!("{section} f32 <{c}>"),
                    format!("{section}_f64_{c}"),
                    format!("{section}&i32\"{c}\""),
                    format!("{section}_u32_{c}"),
                ];
                let f32s: Vec<f32> = (0..len)
                    .map(|_| (rng.next_u64() as i64 as f64 / 2f64.powi(60)) as f32 / 3.0)
                    .collect();
                let f64s: Vec<f64> = (0..len)
                    .map(|_| rng.next_u64() as i64 as f64 / 2f64.powi(50) / 7.0)
                    .collect();
                let i32s: Vec<i32> = (0..len).map(|_| rng.next_u64() as i32).collect();
                let u32s: Vec<u32> = (0..len).map(|_| rng.next_u64() as u32).collect();
                set.f32s.push((
                    point,
                    names[0].clone(),
                    DeviceBuffer::from_host(client, &f32s).expect("upload"),
                    c,
                ));
                set.expect(point, &names[0], c, DataValues::F32(f32s));
                if with_f64 {
                    set.f64s.push((
                        point,
                        names[1].clone(),
                        DeviceBuffer::from_host(client, &f64s).expect("upload"),
                        c,
                    ));
                    set.expect(point, &names[1], c, DataValues::F64(f64s));
                }
                set.i32s.push((
                    point,
                    names[2].clone(),
                    DeviceBuffer::from_host(client, &i32s).expect("upload"),
                    c,
                ));
                set.expect(point, &names[2], c, DataValues::I32(i32s));
                set.u32s.push((
                    point,
                    names[3].clone(),
                    DeviceBuffer::from_host(client, &u32s).expect("upload"),
                    c,
                ));
                set.expect(point, &names[3], c, DataValues::U32(u32s));
            }
        }
        set
    }

    fn expect(&mut self, point: bool, name: &str, components: usize, values: DataValues) {
        let section = if point {
            &mut self.point
        } else {
            &mut self.cell
        };
        section.insert(
            name.to_owned(),
            DataArray {
                name: name.to_owned(),
                components,
                values,
            },
        );
    }

    /// `w` with every array attached.
    pub fn attach<'a>(&'a self, mut w: VtkWriter<'a>) -> VtkWriter<'a> {
        fn add<'a, T: vtk::VtkScalar>(
            w: VtkWriter<'a>,
            point: bool,
            name: &str,
            b: &'a DeviceBuffer<T>,
            c: usize,
        ) -> VtkWriter<'a> {
            if point {
                w.point_data(name, b, c)
            } else {
                w.cell_data(name, b, c)
            }
        }
        for (p, n, b, c) in &self.f32s {
            w = add(w, *p, n, b, *c);
        }
        for (p, n, b, c) in &self.f64s {
            w = add(w, *p, n, b, *c);
        }
        for (p, n, b, c) in &self.i32s {
            w = add(w, *p, n, b, *c);
        }
        for (p, n, b, c) in &self.u32s {
            w = add(w, *p, n, b, *c);
        }
        w
    }
}

// ---------------------------------------------------------------------------------------------
// Writing and the Rust-side checks.

/// One cell of a mesh, in flat order: type, degree and node coordinates `[k][3]` (zero-padded)
/// in internal (A.5) order, as the writer must write them: in f64 for f64 meshes, narrowed to
/// f32 for f32 meshes, with the origin added in f64.
#[derive(Clone, Debug)]
pub struct CellCoords {
    pub cell: CellType,
    pub degree: u32,
    pub coords: Vec<f64>,
}

/// What a written file must contain.
#[derive(Clone, Debug)]
pub struct Expected {
    /// A name for messages.
    pub what: String,
    pub path: PathBuf,
    pub tdim: usize,
    pub gdim: usize,
    pub lagrange: bool,
    /// Whether points are Float32.
    pub f32_points: bool,
    /// The points `[n][3]`, as written (see [`CellCoords`]).
    pub points: Vec<f64>,
    /// The cells in flat order.
    pub cells: Vec<CellCoords>,
    /// Expected data arrays by name; `None` without data.
    pub point_data: BTreeMap<String, DataArray>,
    pub cell_data: BTreeMap<String, DataArray>,
}

impl Expected {
    /// The number of points.
    pub fn num_points(&self) -> usize {
        self.points.len() / 3
    }

    /// The VTK type of every cell.
    pub fn types(&self) -> Vec<u8> {
        self.cells
            .iter()
            .map(|c| expected_type(c.cell, c.degree, self.lagrange))
            .collect()
    }

    /// (min, max) of every coordinate over all points, as VTK's bounds.
    pub fn bounds(&self) -> [f64; 6] {
        let mut b = [
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::INFINITY,
            f64::NEG_INFINITY,
        ];
        for x in self.points.as_chunks::<3>().0 {
            for g in 0..3 {
                b[2 * g] = b[2 * g].min(x[g]);
                b[2 * g + 1] = b[2 * g + 1].max(x[g]);
            }
        }
        b
    }
}

/// The points and cells of `m` as the writer must write them, for a mesh of equispaced
/// geometry.
pub fn expected_geometry<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> (Vec<f64>, Vec<CellCoords>)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let origin = m.origin();
    let origin = origin.as_ref();
    let nodes = m.nodes().to_host().expect("nodes");
    let mut points = Vec::with_capacity(nodes.len() / gd * 3);
    for x in nodes.chunks_exact(gd) {
        for g in 0..3 {
            points.push(if g < gd {
                F::narrow(x[g].widen() + origin[g]).widen()
            } else {
                0.0
            });
        }
    }
    let mut cells = Vec::new();
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let geom = m.geometry_block(b);
        assert_eq!(
            geom.element.family,
            NodeFamily::Equispaced,
            "the corpus is equispaced"
        );
        let k = geom.dofmap.stride as usize;
        let dofmap = geom.dofmap.data.to_host().expect("dofmap");
        for row in dofmap.chunks_exact(k) {
            cells.push(CellCoords {
                cell: block.cell.into(),
                degree: geom.element.degree,
                coords: row
                    .iter()
                    .flat_map(|&n| points[n as usize * 3..n as usize * 3 + 3].to_vec())
                    .collect(),
            });
        }
    }
    (points, cells)
}

/// Writes `case` for scalar `F` into `dir` and returns what the file must contain.
pub fn write_case<F: Real>(client: &Client, case: &Case, dir: &TempDir) -> Expected {
    let any = build_any::<F>(client, &case.host, case.tdim);
    let what = format!(
        "{} ({}, {})",
        case.name,
        name::<F>(),
        encoding_name(case.encoding)
    );
    let path = dir.join(&format!(
        "{}-{}-{}.vtu",
        name::<F>(),
        case.name,
        encoding_name(case.encoding)
    ));
    let (points, cells, num_cells) = with_mesh!(any.as_ref(), m => {
        let (p, c) = expected_geometry(m);
        let n = c.len();
        (p, c, n)
    });
    let data = case
        .data
        .then(|| DataSet::new(client, points.len() / 3, num_cells));
    let (base64, zlib) = case.encoding;
    let mut w = VtkWriter::vtu(&path)
        .lagrange(case.lagrange)
        .base64(base64)
        .zlib(zlib);
    if let Some(d) = &data {
        w = d.attach(w);
    }
    w.write_any(any.as_ref())
        .unwrap_or_else(|e| panic!("{what}: write: {e}"));
    let (point_data, cell_data) = data.map_or_else(Default::default, |d| (d.point, d.cell));
    Expected {
        what,
        path,
        tdim: case.tdim,
        gdim: case.host.gdim,
        lagrange: case.lagrange,
        f32_points: size_of::<F>() == 4,
        points,
        cells,
        point_data,
        cell_data,
    }
}

/// The values of `v` widened to f64 (integers exactly).
pub fn as_f64(v: &DataValues) -> Vec<f64> {
    match v {
        DataValues::F32(x) => x.iter().map(|&a| f64::from(a)).collect(),
        DataValues::F64(x) => x.clone(),
        DataValues::I32(x) => x.iter().map(|&a| f64::from(a)).collect(),
        DataValues::U32(x) => x.iter().map(|&a| f64::from(a)).collect(),
        DataValues::I64(x) => x.iter().map(|&a| a as f64).collect(),
        DataValues::U8(x) => x.iter().map(|&a| f64::from(a)).collect(),
    }
}

/// Checks the parsed file `f` against `e` exactly: points (type and bits), connectivity
/// through `vtk_to_internal` (corners in VTK linear order for linear cells), offsets, types
/// and data arrays.
pub fn check_file(e: &Expected, f: &VtuFile) {
    let what = &e.what;
    match (&f.points, e.f32_points) {
        (DataValues::F32(_), true) | (DataValues::F64(_), false) => {}
        (p, _) => panic!("{what}: points are {}", variant(p)),
    }
    let points = as_f64(&f.points);
    assert_eq!(points.len(), e.points.len(), "{what}: number of points");
    for (i, (a, b)) in points.iter().zip(&e.points).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{what}: point coordinate {i}: {a} vs {b}"
        );
    }
    assert_eq!(f.types, e.types(), "{what}: cell types");
    assert_eq!(f.offsets.len(), e.cells.len(), "{what}: offsets");
    let mut start = 0usize;
    for (c, (cell, &end)) in e.cells.iter().zip(&f.offsets).enumerate() {
        let end = usize::try_from(end).expect("offset");
        let conn = &f.connectivity[start..end];
        let linear = written_linear(cell.cell, cell.degree, e.lagrange);
        let p = if linear { 1 } else { cell.degree };
        let map = gridcl_refcell::vtk_to_internal(cell.cell, p).expect("VTK map");
        assert_eq!(conn.len(), map.len(), "{what}: cell {c}: number of nodes");
        // Corners are the first internal nodes at every degree (A.5).
        for (k, (&pt, &internal)) in conn.iter().zip(&map).enumerate() {
            let pt = usize::try_from(pt).expect("point index");
            assert_eq!(
                &points[pt * 3..pt * 3 + 3],
                &cell.coords[internal * 3..internal * 3 + 3],
                "{what}: cell {c}, VTK node {k} (internal {internal})"
            );
        }
        start = end;
    }
    assert_eq!(start, f.connectivity.len(), "{what}: connectivity length");
    check_data(what, "point", &e.point_data, &f.point_data);
    check_data(what, "cell", &e.cell_data, &f.cell_data);
}

/// The expected arrays equal the written ones, by name and bitwise.
pub fn check_data(
    what: &str,
    section: &str,
    want: &BTreeMap<String, DataArray>,
    got: &[DataArray],
) {
    let got: BTreeMap<&str, &DataArray> = got.iter().map(|a| (a.name.as_str(), a)).collect();
    assert_eq!(
        got.keys().copied().collect::<Vec<_>>(),
        want.keys().map(String::as_str).collect::<Vec<_>>(),
        "{what}: {section} data names"
    );
    for (name, w) in want {
        let g = got[name.as_str()];
        assert_eq!(
            g.components, w.components,
            "{what}: {section} data {name:?}: components"
        );
        assert!(
            same_bits(&g.values, &w.values),
            "{what}: {section} data {name:?}: values"
        );
    }
}

/// Whether `a` and `b` are the same variant with bitwise equal values.
pub fn same_bits(a: &DataValues, b: &DataValues) -> bool {
    match (a, b) {
        (DataValues::F32(x), DataValues::F32(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (DataValues::F64(x), DataValues::F64(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (DataValues::I32(x), DataValues::I32(y)) => x == y,
        (DataValues::U32(x), DataValues::U32(y)) => x == y,
        (DataValues::I64(x), DataValues::I64(y)) => x == y,
        (DataValues::U8(x), DataValues::U8(y)) => x == y,
        _ => false,
    }
}

/// The name of a variant, and the numpy dtype of the oracle's arrays.
pub fn variant(v: &DataValues) -> &'static str {
    match v {
        DataValues::F32(_) => "float32",
        DataValues::F64(_) => "float64",
        DataValues::I32(_) => "int32",
        DataValues::U32(_) => "uint32",
        DataValues::I64(_) => "int64",
        DataValues::U8(_) => "uint8",
    }
}

/// Reads and parses `path`.
pub fn parse_file(path: &Path) -> VtuFile {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    parse_vtu(&bytes).unwrap_or_else(|e| panic!("{}: parse_vtu: {e}", path.display()))
}

// ---------------------------------------------------------------------------------------------
// The oracle's arrays.

/// An oracle array `{"dtype", "shape", "hex"}` (little-endian bytes) as values and shape.
pub fn decode_array(v: &Value) -> (DataValues, Vec<usize>) {
    let dtype = v["dtype"].as_str().expect("dtype");
    let shape: Vec<usize> = v["shape"]
        .as_array()
        .expect("shape")
        .iter()
        .map(|s| s.as_u64().expect("shape entry") as usize)
        .collect();
    let hex = v["hex"].as_str().expect("hex").as_bytes();
    assert!(hex.len().is_multiple_of(2), "odd hex length");
    let digit = |c: u8| (c as char).to_digit(16).expect("hex digit") as u8;
    let bytes: Vec<u8> = hex
        .as_chunks::<2>()
        .0
        .iter()
        .map(|p| digit(p[0]) << 4 | digit(p[1]))
        .collect();
    fn le<const N: usize, T>(bytes: &[u8], f: impl Fn([u8; N]) -> T) -> Vec<T> {
        assert!(bytes.len().is_multiple_of(N), "byte count");
        bytes.as_chunks::<N>().0.iter().map(|c| f(*c)).collect()
    }
    let values = match dtype {
        "float32" => DataValues::F32(le(&bytes, f32::from_le_bytes)),
        "float64" => DataValues::F64(le(&bytes, f64::from_le_bytes)),
        "int32" => DataValues::I32(le(&bytes, i32::from_le_bytes)),
        "uint32" => DataValues::U32(le(&bytes, u32::from_le_bytes)),
        "int64" => DataValues::I64(le(&bytes, i64::from_le_bytes)),
        "uint8" => DataValues::U8(bytes),
        other => panic!("unexpected dtype {other}"),
    };
    (values, shape)
}

/// A JSON list of integers.
pub fn ints(v: &Value) -> Vec<i64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected a list, got {v}"))
        .iter()
        .map(|x| x.as_i64().expect("integer"))
        .collect()
}

/// A JSON list of numbers.
pub fn floats(v: &Value) -> Vec<f64> {
    v.as_array()
        .unwrap_or_else(|| panic!("expected a list, got {v}"))
        .iter()
        .map(|x| x.as_f64().expect("number"))
        .collect()
}

// ---------------------------------------------------------------------------------------------
// The VTK-written reference cells (ADR-0008; T1).

/// One committed VTK-written reference cell.
#[derive(Clone, Debug)]
pub struct ReferenceCell {
    pub file: String,
    pub vtk_type: u8,
    pub degree: u32,
    pub num_points: usize,
}

impl ReferenceCell {
    pub fn path(&self) -> PathBuf {
        reference_dir().join(&self.file)
    }
}

/// The directory of the VTK-written reference cells.
pub fn reference_dir() -> PathBuf {
    super::python::repo().join("gridcl-io/tests/fixtures/vtk")
}

/// The reference cells of `manifest.json`.
pub fn reference_cells() -> Vec<ReferenceCell> {
    let path = reference_dir().join("manifest.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    let v: Value = serde_json::from_str(&text).expect("manifest JSON");
    v["cells"]
        .as_array()
        .expect("cells")
        .iter()
        .map(|c| ReferenceCell {
            file: c["file"].as_str().expect("file").to_owned(),
            vtk_type: c["vtk_type"].as_u64().expect("vtk_type") as u8,
            degree: c["degree"].as_u64().expect("degree") as u32,
            num_points: c["num_points"].as_u64().expect("num_points") as usize,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Warnings.

/// Warnings by thread, so that tests running in parallel see only their own.
static RECORDS: Mutex<Vec<(ThreadId, String)>> = Mutex::new(Vec::new());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
    }
    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let mut records = RECORDS.lock().unwrap();
            records.push((std::thread::current().id(), record.args().to_string()));
        }
    }
    fn flush(&self) {}
}

static LOGGER: Capture = Capture;

/// Installs the capturing logger (once per test binary).
pub fn capture_warnings() {
    let _ = log::set_logger(&LOGGER);
    log::set_max_level(log::LevelFilter::Warn);
}

/// The warnings logged on this thread so far.
pub fn my_warnings() -> Vec<String> {
    let me = std::thread::current().id();
    let records = RECORDS.lock().unwrap();
    records
        .iter()
        .filter(|r| r.0 == me)
        .map(|r| r.1.clone())
        .collect()
}
