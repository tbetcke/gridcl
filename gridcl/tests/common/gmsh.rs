//! Helpers of the M4 acceptance tests: the fixture manifest (shared with gridcl-io's tests),
//! reading fixtures, and host snapshots of meshes keyed by origin ids, so that two meshes
//! compare independently of their local numbering (ADR-0022, item 7).

#[path = "../../../gridcl-io/tests/common/mod.rs"]
pub mod manifest;

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_topology::decode_origin_ids;

#[allow(unused_imports)] // each test binary uses a subset
pub use manifest::{Expect, Fixture, manifest};

/// Reads `path` with default options on one rank.
pub fn read<F: Real>(client: &Client, path: &Path) -> Result<AnyMesh<F>> {
    gmsh::read::<F, Serial>(client, Serial, path, GmshOptions::default())
}

/// Reads fixture `f`, panicking with the error.
pub fn read_fixture<F: Real>(client: &Client, f: &Fixture) -> AnyMesh<F> {
    read::<F>(client, &f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file))
}

/// A path for a file written by a test, unique per process and `name`.
pub fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("gridcl-m4-{}-{name}", std::process::id()))
}

/// One cell: its type, degree, vertex origin ids in reference order, and node coordinates
/// `[node][gdim]` in internal (A.5) order, in f64 with the origin added back.
#[derive(Clone, Debug)]
pub struct CellSnap {
    pub cell: CellType,
    pub degree: u32,
    pub vertices: Vec<i64>,
    pub coords: Vec<f64>,
}

/// (dimension, sorted vertex origin ids of the entity, value).
pub type TagKey = (usize, Vec<i64>, i32);

/// A mesh on the host, keyed by origin ids.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub tdim: usize,
    pub gdim: usize,
    /// The local node count.
    pub num_nodes: usize,
    /// The cells by origin id.
    pub cells: BTreeMap<i64, CellSnap>,
    /// (name, dimension, value).
    pub names: BTreeSet<(String, usize, i32)>,
    pub physical: BTreeSet<TagKey>,
    pub elementary: BTreeSet<TagKey>,
    /// The largest absolute coordinate.
    pub max_abs: f64,
}

/// The sorted vertex origin ids of every entity of dimension `dim`, by flat index: cells over
/// the cell blocks in block order for dim = tdim, else the entity blocks.
pub fn entity_vertex_ids<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, dim: usize) -> Vec<Vec<i64>>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let ids = decode_origin_ids(&m.vertex_origin_ids().to_host().expect("vertex ids"));
    let rows = |data: &DeviceBuffer<u32>, stride: u32| -> Vec<Vec<i64>> {
        data.to_host()
            .expect("vertex rows")
            .chunks_exact(stride as usize)
            .map(|r| {
                let mut t: Vec<i64> = r.iter().map(|&v| ids[v as usize]).collect();
                t.sort_unstable();
                t
            })
            .collect()
    };
    if dim == m.tdim() {
        m.cell_blocks()
            .iter()
            .flat_map(|b| rows(&b.vertices.data, b.vertices.stride))
            .collect()
    } else {
        let blocks = m
            .entity_blocks(dim)
            .unwrap_or_else(|e| panic!("tags of dimension {dim} without entities: {e}"));
        let mut out = Vec::new();
        for b in blocks {
            assert_eq!(b.offset as usize, out.len(), "flat entity numbering");
            out.extend(rows(&b.vertices.data, b.vertices.stride));
        }
        out
    }
}

/// The entries of `tags` keyed by sorted vertex origin ids, after checking that the indices
/// are in range, never `u32::MAX`, and sorted by (index, value) without repeats.
pub fn tag_keys<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>, tags: &MeshTags) -> BTreeSet<TagKey>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let entities = entity_vertex_ids(m, tags.dim);
    let indices = tags.indices.to_host().expect("tag indices");
    let values = tags.values.to_host().expect("tag values");
    assert_eq!(indices.len(), values.len(), "MeshTags lengths");
    let pairs: Vec<(u32, i32)> = indices
        .iter()
        .copied()
        .zip(values.iter().copied())
        .collect();
    assert!(
        pairs.windows(2).all(|w| w[0] < w[1]),
        "dimension {}: entries not sorted by (index, value) or repeated",
        tags.dim
    );
    pairs
        .into_iter()
        .map(|(i, v)| {
            assert_ne!(i, u32::MAX, "dimension {}: unmatched entry", tags.dim);
            let key = entities
                .get(i as usize)
                .unwrap_or_else(|| panic!("dimension {}: index {i} out of range", tags.dim));
            (tags.dim, key.clone(), v)
        })
        .collect()
}

/// The snapshot of `m`.
pub fn snapshot<F, TD, GD, C>(m: &Mesh<F, TD, GD, C>) -> Snapshot
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = m.gdim();
    let ids = decode_origin_ids(&m.vertex_origin_ids().to_host().expect("vertex ids"));
    let nodes: Vec<f64> = m
        .nodes()
        .to_host()
        .expect("nodes")
        .iter()
        .map(|x| x.widen())
        .collect();
    let origin = m.origin();
    let origin = origin.as_ref();
    let mut cells = BTreeMap::new();
    for (b, block) in m.cell_blocks().iter().enumerate() {
        let geom = m.geometry_block(b);
        let cell: CellType = block.cell.into();
        let k = geom.dofmap.stride as usize;
        let nv = block.vertices.stride as usize;
        let verts = block.vertices.data.to_host().expect("cell vertices");
        let dofmap = geom.dofmap.data.to_host().expect("dofmap");
        let cell_ids = decode_origin_ids(&block.origin_ids.to_host().expect("cell ids"));
        for (c, &id) in cell_ids.iter().enumerate() {
            let coords = dofmap[c * k..(c + 1) * k]
                .iter()
                .flat_map(|&n| (0..gd).map(move |g| (n as usize * gd + g, g)))
                .map(|(i, g)| nodes[i] + origin[g])
                .collect();
            let snap = CellSnap {
                cell,
                degree: geom.element.degree,
                vertices: verts[c * nv..(c + 1) * nv]
                    .iter()
                    .map(|&v| ids[v as usize])
                    .collect(),
                coords,
            };
            assert!(cells.insert(id, snap).is_none(), "duplicate cell id {id}");
        }
    }
    let max_abs = cells
        .values()
        .flat_map(|c| c.coords.iter())
        .fold(0.0f64, |a, x| a.max(x.abs()));
    let tags = m.tags();
    let (mut physical, mut elementary) = (BTreeSet::new(), BTreeSet::new());
    for d in 0..=m.tdim() {
        if let Some(t) = tags.physical(d) {
            physical.extend(tag_keys(m, t));
        }
        if let Some(t) = tags.elementary(d) {
            elementary.extend(tag_keys(m, t));
        }
    }
    Snapshot {
        tdim: m.tdim(),
        gdim: gd,
        num_nodes: m.node_index_map().num_local(),
        cells,
        names: tags.names().map(|(n, d, v)| (n.to_owned(), d, v)).collect(),
        physical,
        elementary,
        max_abs,
    }
}

/// The snapshot of an erased mesh.
pub fn snapshot_any<F: Real>(any: &AnyMesh<F>) -> Snapshot {
    with_mesh!(any.as_ref(), m => snapshot(m))
}

/// Asserts identical topology (cell types, degrees, cell and vertex origin ids, names, tags
/// keyed by origin ids) and coordinates equal to `tol` relative to the largest absolute
/// coordinate (ADR-0022, items 7 and 8).
pub fn assert_same(a: &Snapshot, b: &Snapshot, tol: f64, what: &str) {
    assert_eq!((a.tdim, a.gdim), (b.tdim, b.gdim), "{what}: dimensions");
    assert_eq!(
        a.cells.keys().collect::<Vec<_>>(),
        b.cells.keys().collect::<Vec<_>>(),
        "{what}: cell origin ids"
    );
    let scale = a.max_abs.max(f64::MIN_POSITIVE);
    for ((id, x), y) in a.cells.iter().zip(b.cells.values()) {
        assert_eq!(
            (x.cell, x.degree, &x.vertices),
            (y.cell, y.degree, &y.vertices),
            "{what}: cell {id}"
        );
        assert_eq!(x.coords.len(), y.coords.len(), "{what}: cell {id} nodes");
        for (p, q) in x.coords.iter().zip(&y.coords) {
            assert!(
                (p - q).abs() <= tol * scale,
                "{what}: cell {id}: coordinate {p} vs {q} (tolerance {tol} x {scale})"
            );
        }
    }
    assert_eq!(a.names, b.names, "{what}: names");
    assert_eq!(a.physical, b.physical, "{what}: physical tags");
    assert_eq!(a.elementary, b.elementary, "{what}: elementary tags");
}
