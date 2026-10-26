//! The manifest of the Gmsh fixture corpus (`tests/fixtures/gmsh/manifest.json`), written by
//! `scripts/fixtures/gmsh/generate.py` from Gmsh's own counts. gridcl's tests include this file
//! too (`gridcl/tests/common/gmsh.rs`).
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use serde_json::Value;

/// The fixture directory.
pub fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../gridcl-io/tests/fixtures/gmsh")
        .canonicalize()
        .unwrap_or_else(|e| panic!("gridcl-io/tests/fixtures/gmsh: {e} (made in M4 T1)"))
}

/// One physical group, as Gmsh reports it.
#[derive(Clone, Debug)]
pub struct PhysicalGroup {
    pub dim: usize,
    pub tag: i32,
    pub name: Option<String>,
    /// The number of elements in the group.
    pub size: usize,
    /// The sorted corner node tags of every element of the group, sorted.
    pub elements: Vec<Vec<i64>>,
}

/// What reading a fixture must give.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Expect {
    /// A mesh.
    Mesh,
    /// `DuplicateOriginId(id)` for a repeated node tag (`nodes`) or element tag.
    DuplicateOriginId { id: i64, nodes: bool },
}

/// One fixture file.
#[derive(Clone, Debug)]
pub struct Fixture {
    /// The file name in [`fixtures_dir`].
    pub file: String,
    /// Whether the file is binary.
    pub binary: bool,
    /// Gmsh's node count of the file.
    pub num_nodes: usize,
    /// The distinct nodes of the D_TOP elements; for a serendipity file, those of its
    /// complete-P2 twin.
    pub num_mesh_nodes: usize,
    /// Element counts per Gmsh type id.
    pub element_counts: BTreeMap<i32, usize>,
    /// The expected (D_TOP, D_GEO).
    pub tdim: usize,
    pub gdim: usize,
    pub physical_groups: Vec<PhysicalGroup>,
    /// For a serendipity file, the file of the same mesh written as complete P2.
    pub twin: Option<String>,
    pub partitioned: bool,
    pub periodic: bool,
    pub expect: Expect,
}

impl Fixture {
    /// The path of the file.
    pub fn path(&self) -> PathBuf {
        fixtures_dir().join(&self.file)
    }

    /// The path of the complete-P2 twin.
    pub fn twin_path(&self) -> Option<PathBuf> {
        self.twin.as_ref().map(|t| fixtures_dir().join(t))
    }
}

/// The manifest.
#[derive(Clone, Debug)]
pub struct Manifest {
    /// The Gmsh version that wrote the corpus.
    pub gmsh_version: String,
    pub files: Vec<Fixture>,
}

impl Manifest {
    /// The fixtures that read into a mesh.
    pub fn meshes(&self) -> impl Iterator<Item = &Fixture> {
        self.files.iter().filter(|f| f.expect == Expect::Mesh)
    }
}

fn usize_of(v: &Value, what: &str) -> usize {
    v.as_u64()
        .unwrap_or_else(|| panic!("manifest: {what} is not an unsigned integer: {v}")) as usize
}

fn i64_of(v: &Value, what: &str) -> i64 {
    v.as_i64()
        .unwrap_or_else(|| panic!("manifest: {what} is not an integer: {v}"))
}

fn group(v: &Value) -> PhysicalGroup {
    let elements: Vec<Vec<i64>> = v["elements"]
        .as_array()
        .expect("manifest: physical group elements")
        .iter()
        .map(|e| {
            e.as_array()
                .expect("manifest: element corners")
                .iter()
                .map(|t| i64_of(t, "corner tag"))
                .collect()
        })
        .collect();
    PhysicalGroup {
        dim: usize_of(&v["dim"], "group dim"),
        tag: i32::try_from(i64_of(&v["tag"], "group tag")).expect("i32 physical tag"),
        name: v["name"].as_str().map(str::to_owned),
        size: usize_of(&v["size"], "group size"),
        elements,
    }
}

fn fixture(v: &Value) -> Fixture {
    let file = v["file"].as_str().expect("manifest: file").to_owned();
    let expect = match v["expect_error"].as_str() {
        None => Expect::Mesh,
        Some("DuplicateOriginId") => Expect::DuplicateOriginId {
            id: i64_of(&v["duplicate_id"], "duplicate_id"),
            nodes: match v["duplicate_of"].as_str() {
                Some("nodes") => true,
                Some("elements") => false,
                other => panic!("manifest: {file}: duplicate_of {other:?}"),
            },
        },
        Some(other) => panic!("manifest: {file}: unknown expect_error {other}"),
    };
    let element_counts = v["element_counts"]
        .as_object()
        .expect("manifest: element_counts")
        .iter()
        .map(|(k, n)| (k.parse().expect("type id"), usize_of(n, "element count")))
        .collect();
    Fixture {
        binary: v["binary"].as_bool().expect("manifest: binary"),
        num_nodes: usize_of(&v["num_nodes"], "num_nodes"),
        num_mesh_nodes: usize_of(&v["num_mesh_nodes"], "num_mesh_nodes"),
        element_counts,
        tdim: usize_of(&v["tdim"], "tdim"),
        gdim: usize_of(&v["gdim"], "gdim"),
        physical_groups: v["physical_groups"]
            .as_array()
            .expect("manifest: physical_groups")
            .iter()
            .map(group)
            .collect(),
        twin: v["twin"].as_str().map(str::to_owned),
        partitioned: v["partitioned"].as_bool().unwrap_or(false),
        periodic: v["periodic"].as_bool().unwrap_or(false),
        expect,
        file,
    }
}

/// Loads the manifest; panics if it is missing or malformed.
pub fn manifest() -> Manifest {
    let path = fixtures_dir().join("manifest.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e} (made in M4 T1)", path.display()));
    let v: Value = serde_json::from_str(&text).expect("manifest.json is JSON");
    Manifest {
        gmsh_version: v["gmsh_version"]
            .as_str()
            .expect("manifest: gmsh_version")
            .to_owned(),
        files: v["files"]
            .as_array()
            .expect("manifest: files")
            .iter()
            .map(fixture)
            .collect(),
    }
}

/// The Gmsh type ids the corpus must cover: every A.7 type up to order 4, pyramids up to
/// order 3, and the serendipity types 16–19.
pub const REQUIRED_TYPES: [i32; 32] = [
    15, // point
    1, 8, 26, 27, // interval
    2, 9, 21, 23, // triangle
    3, 10, 36, 37, 16, // quadrilateral
    4, 11, 29, 30, // tetrahedron
    5, 12, 92, 93, 17, // hexahedron
    6, 13, 90, 91, 18, // prism
    7, 14, 118, 19, // pyramid
];
