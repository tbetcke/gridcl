//! Chunked reading of binary Gmsh files (docs/plans/M8.md, T1; ADR-0035, rule 1): the chunks
//! of `MshIndex` concatenate to what `read_file` parses, in either byte order and `size_t`
//! width, read only their byte ranges, and malformed headers give `Parse` with the offset.

use crate::common;
use crate::encode;

use std::ops::Range;
use std::path::{Path, PathBuf};

use common::{Fixture, manifest};
use gridcl_device::GridError;
use gridcl_io::gmsh::{self, ElementBlock, MshFile, MshIndex};

/// The even chunk r of P over n items.
fn chunk(n: usize, r: usize, p: usize) -> Range<usize> {
    r * n / p..(r + 1) * n / p
}

/// One element: (entity tag, element type, element tag, node tags).
type Record = (i32, i32, i64, Vec<i64>);

/// The elements of `dim` in `blocks`, in order.
fn records(blocks: &[ElementBlock], dim: usize) -> Vec<Record> {
    let mut out = Vec::new();
    for b in blocks.iter().filter(|b| b.dim == dim) {
        let k = b.nodes.len() / b.tags.len().max(1);
        for (e, &tag) in b.tags.iter().enumerate() {
            out.push((
                b.entity_tag,
                b.element_type,
                tag,
                b.nodes[e * k..(e + 1) * k].to_vec(),
            ));
        }
    }
    out
}

/// Asserts that the chunks of `index` over P = 1–8 concatenate to `model`.
fn assert_chunks_equal(index: &MshIndex, model: &MshFile, what: &str) {
    assert!(index.is_binary(), "{what}: binary");
    let tags: Vec<i64> = model
        .node_blocks
        .iter()
        .flat_map(|b| b.tags.clone())
        .collect();
    let coords: Vec<[f64; 3]> = model
        .node_blocks
        .iter()
        .flat_map(|b| b.coords.clone())
        .collect();
    assert_eq!(index.num_nodes(), tags.len(), "{what}: nodes");
    for dim in 0..=3 {
        let want = records(&model.element_blocks, dim);
        assert_eq!(
            index.num_elements(dim),
            want.len(),
            "{what}: elements of dimension {dim}"
        );
        for p in 1..=8 {
            let mut got = Vec::new();
            for r in 0..p {
                let range = chunk(want.len(), r, p);
                let c = index
                    .read_elements(dim, range.clone())
                    .unwrap_or_else(|e| panic!("{what}: elements {dim} {range:?}: {e}"));
                assert!(c.blocks.iter().all(|b| b.dim == dim && !b.tags.is_empty()));
                let part = records(&c.blocks, dim);
                assert_eq!(part.len(), range.len(), "{what}: elements {dim} {range:?}");
                got.extend(part);
            }
            assert_eq!(
                got, want,
                "{what}: elements of dimension {dim} in {p} chunks"
            );
        }
    }
    for p in 1..=8 {
        let (mut got_tags, mut got_coords) = (Vec::new(), Vec::new());
        for r in 0..p {
            let range = chunk(tags.len(), r, p);
            let c = index
                .read_nodes(range.clone())
                .unwrap_or_else(|e| panic!("{what}: nodes {range:?}: {e}"));
            assert_eq!(c.tags.len(), range.len(), "{what}: nodes {range:?}");
            assert_eq!(c.coords.len(), range.len(), "{what}: coordinates {range:?}");
            got_tags.extend(c.tags);
            got_coords.extend(c.coords);
        }
        assert_eq!(got_tags, tags, "{what}: node tags in {p} chunks");
        assert_eq!(got_coords, coords, "{what}: coordinates in {p} chunks");
    }
}

fn scan(path: &Path) -> MshIndex {
    MshIndex::scan(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// A path for a file written by a test, unique per process and `name`.
fn temp_path(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("gridcl-m8-index-{}-{name}", std::process::id()))
}

fn binary_fixtures() -> Vec<Fixture> {
    manifest().files.into_iter().filter(|f| f.binary).collect()
}

/// Every binary fixture, cut into P = 1–8 even chunks of nodes and of the elements of every
/// dimension, concatenates to `read_file`'s blocks.
#[test]
fn chunks_concatenate_to_read_file() {
    let fixtures = binary_fixtures();
    assert!(!fixtures.is_empty(), "no binary fixtures");
    for f in &fixtures {
        let model = gmsh::read_file(&f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        assert_chunks_equal(&scan(&f.path()), &model, &f.file);
    }
}

/// The binary fixtures re-encoded big-endian and with a 4-byte `size_t` read alike.
#[test]
fn chunks_in_both_byte_orders_and_size_t_widths() {
    for f in binary_fixtures().iter().step_by(3) {
        let model = gmsh::read_file(&f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        for (big_endian, size_t) in [(false, 4), (true, 4), (true, 8)] {
            let what = format!("{} (big-endian {big_endian}, size_t {size_t})", f.file);
            let path = temp_path(&format!("{big_endian}-{size_t}-{}", f.file));
            std::fs::write(&path, encode::encode_binary(&model, big_endian, size_t)).unwrap();
            assert_chunks_equal(&scan(&path), &model, &what);
            std::fs::remove_file(&path).ok();
        }
    }
}

/// An ASCII file scans as not binary; its chunks are refused with `Parse`.
#[test]
fn ascii_files_are_not_binary() {
    let path = common::fixtures_dir().join("triangle_p1.msh");
    let index = scan(&path);
    assert!(!index.is_binary());
    assert!(matches!(
        index.read_nodes(0..1),
        Err(GridError::Parse { .. })
    ));
    assert!(matches!(
        index.read_elements(2, 0..1),
        Err(GridError::Parse { .. })
    ));
}

/// A range beyond the counts is `IndexOutOfRange`.
#[test]
fn ranges_beyond_the_counts() {
    let index = scan(&common::fixtures_dir().join("triangle_p1-binary.msh"));
    let n = index.num_nodes();
    assert!(matches!(
        index.read_nodes(0..n + 1),
        Err(GridError::IndexOutOfRange { index, len }) if index == n + 1 && len == n
    ));
    let m = index.num_elements(2);
    assert!(matches!(
        index.read_elements(2, m..m + 1),
        Err(GridError::IndexOutOfRange { .. })
    ));
    assert!(index.read_elements(3, 0..0).unwrap().blocks.is_empty());
}

/// A chunk reads only its byte range: the bytes of one node are its tag and coordinates, and
/// of one element its tag and node tags.
#[test]
fn chunks_read_only_their_bytes() {
    let index = scan(&common::fixtures_dir().join("tetrahedron_p2-binary.msh"));
    let before = index.bytes_read();
    index.read_nodes(3..5).unwrap();
    assert_eq!(index.bytes_read() - before, 2 * (8 + 3 * 8));
    let before = index.bytes_read();
    let c = index.read_elements(3, 1..2).unwrap();
    assert_eq!(c.blocks[0].element_type, 11);
    assert_eq!(index.bytes_read() - before, 8 + 10 * 8);
}

/// The byte offset of the first occurrence of `marker` after `from`.
fn find(bytes: &[u8], marker: &[u8], from: usize) -> usize {
    from + bytes[from..]
        .windows(marker.len())
        .position(|w| w == marker)
        .expect("marker")
}

/// Truncated files and corrupt block headers give `Parse` with the byte offset of the fault.
#[test]
fn malformed_headers_give_parse_with_the_offset() {
    let path = common::fixtures_dir().join("two_regions_p1-binary.msh");
    let bytes = std::fs::read(&path).unwrap();
    let nodes = find(&bytes, b"$Nodes\n", 0) + 7;
    let elements = find(&bytes, b"$Elements\n", 0) + 10;
    let check = |name: &str, data: &[u8], want: Option<u64>| {
        let p = temp_path(name);
        std::fs::write(&p, data).unwrap();
        let r = MshIndex::scan(&p);
        std::fs::remove_file(&p).ok();
        match r {
            Err(GridError::Parse { offset, .. }) => {
                if let Some(want) = want {
                    assert_eq!(offset, want, "{name}");
                }
            }
            other => panic!("{name}: expected Parse, got {other:?}"),
        }
    };
    // Cut inside the first node block, after its header.
    check("cut-nodes", &bytes[..nodes + 4 * 8 + 20 + 8], None);
    // Cut inside the element section.
    check("cut-elements", &bytes[..elements + 4 * 8 + 10], None);
    // The first node block's dimension (after the four size_t of the section header) is 7.
    let mut bad = bytes.clone();
    let at = nodes + 4 * 8;
    bad[at..at + 4].copy_from_slice(&7i32.to_le_bytes());
    check("bad-dimension", &bad, Some(at as u64));
    // The first element block's parametric-free header with an element type in a block of the
    // wrong dimension: type 4 (tetrahedron) where the block says dimension 0.
    let mut bad = bytes.clone();
    let at = elements + 4 * 8 + 8;
    bad[at..at + 4].copy_from_slice(&4i32.to_le_bytes());
    check("type-in-wrong-dimension", &bad, Some(at as u64));
    // A node count larger than the file.
    let mut bad = bytes.clone();
    let at = nodes + 4 * 8 + 12;
    bad[at..at + 8].copy_from_slice(&(1u64 << 40).to_le_bytes());
    check("huge-count", &bad, Some(at as u64));
    // An unknown element type.
    let mut bad = bytes;
    let at = elements + 4 * 8 + 8;
    bad[at..at + 4].copy_from_slice(&999i32.to_le_bytes());
    let p = temp_path("unknown-type");
    std::fs::write(&p, &bad).unwrap();
    let r = MshIndex::scan(&p);
    std::fs::remove_file(&p).ok();
    assert!(
        matches!(
            r,
            Err(GridError::UnsupportedElementType { type_id: 999, .. })
        ),
        "{r:?}"
    );
}
