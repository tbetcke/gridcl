//! M4 acceptance tests at the file level (docs/design.md, "M4 Gmsh I/O (serial)"):
//! the committed corpus reads with Gmsh's node counts, covers every element type, and agrees
//! with mshio, the test oracle (ADR-0022).

mod common;
mod encode;

use std::collections::{BTreeMap, BTreeSet};

use common::{Expect, Fixture, REQUIRED_TYPES, manifest};
use gridcl_io::gmsh::{self, MshFile};

fn read(f: &Fixture) -> MshFile {
    gmsh::read_file(&f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file))
}

/// Whether the file's `$MeshFormat` line says binary (file-type 1).
fn header_says_binary(bytes: &[u8]) -> bool {
    let text = String::from_utf8_lossy(&bytes[..bytes.len().min(64)]);
    let line = text
        .lines()
        .nth(1)
        .expect("a $MeshFormat line after the section header");
    let fields: Vec<&str> = line.split_whitespace().collect();
    assert_eq!(fields[0], "4.1", "MSH version");
    fields[1] == "1"
}

/// The parsed file has Gmsh's node count and element counts per type, and every element has
/// its type's node count (ADR-0022, item 14).
#[test]
fn corpus_reads_with_gmsh_node_counts() {
    let manifest = manifest();
    assert!(!manifest.files.is_empty(), "empty manifest");
    for f in &manifest.files {
        let file = read(f);
        let nodes: usize = file.node_blocks.iter().map(|b| b.tags.len()).sum();
        assert_eq!(nodes, f.num_nodes, "{}: node count", f.file);
        for b in &file.node_blocks {
            assert_eq!(b.coords.len(), b.tags.len(), "{}: coordinates", f.file);
        }
        let mut counts = BTreeMap::new();
        for b in &file.element_blocks {
            let ty =
                gmsh::element_type(b.element_type).unwrap_or_else(|e| panic!("{}: {e}", f.file));
            assert_eq!(
                b.nodes.len(),
                b.tags.len() * ty.num_nodes,
                "{}: nodes of type {}",
                f.file,
                b.element_type
            );
            *counts.entry(b.element_type).or_insert(0) += b.tags.len();
        }
        assert_eq!(counts, f.element_counts, "{}: element counts", f.file);
        assert_eq!(
            f.partitioned,
            file.partitioned_entities.is_some(),
            "{}",
            f.file
        );
        assert_eq!(f.periodic, !file.periodic.is_empty(), "{}", f.file);
    }
}

/// Every A.7 type up to order 4 (pyramids up to order 3) and every serendipity type is in the
/// corpus, which has ASCII and binary files, a partitioned and a periodic file, and the
/// manifest's encodings match the files.
#[test]
fn corpus_covers_every_element_type() {
    let manifest = manifest();
    assert!(
        manifest.gmsh_version.starts_with("4.15."),
        "pinned Gmsh 4.15.x, found {}",
        manifest.gmsh_version
    );
    let mut types = BTreeSet::new();
    for f in &manifest.files {
        let bytes = std::fs::read(f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        assert_eq!(header_says_binary(&bytes), f.binary, "{}: encoding", f.file);
        for b in &read(f).element_blocks {
            types.insert(b.element_type);
        }
    }
    let missing: Vec<i32> = REQUIRED_TYPES
        .iter()
        .copied()
        .filter(|t| !types.contains(t))
        .collect();
    assert!(
        missing.is_empty(),
        "types missing from the corpus: {missing:?}"
    );
    assert!(manifest.files.iter().any(|f| f.binary), "no binary file");
    assert!(manifest.files.iter().any(|f| !f.binary), "no ASCII file");
    assert!(
        manifest.files.iter().any(|f| f.partitioned),
        "no partitioned file"
    );
    assert!(
        manifest.files.iter().any(|f| f.periodic),
        "no periodic file"
    );
}

/// The Gmsh type mshio 0.4.2 cannot read: it gives the 18-node prism 28 nodes (ADR-0023).
const MSHIO_UNREADABLE_TYPE: i32 = 13;

/// Nodes and elements agree with mshio on every fixture mshio can read: node blocks (entity,
/// coordinates and tags), element blocks (entity, type, tags and node tags). mshio must fail on
/// exactly the files with 18-node prisms (ADR-0023).
#[test]
fn nodes_and_elements_agree_with_mshio() {
    let manifest = manifest();
    let mut exempt = 0;
    for f in &manifest.files {
        let bytes = std::fs::read(f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        let ours = read(f);
        let parsed = mshio::parse_msh_bytes(&bytes);
        if f.element_counts.contains_key(&MSHIO_UNREADABLE_TYPE) {
            assert!(
                parsed.is_err(),
                "{}: mshio reads 18-node prisms now; remove the ADR-0023 exemption",
                f.file
            );
            exempt += 1;
            continue;
        }
        let theirs = parsed.unwrap_or_else(|e| panic!("{}: mshio: {e:?}", f.file));
        agree_with_mshio(&f.file, &ours, &theirs, f.expect == Expect::Mesh);
    }
    assert!(exempt > 0, "the corpus has no 18-node prisms");
}

/// Asserts that `ours` and mshio's parse of the same file agree: node blocks (entity,
/// coordinates and tags) and element blocks (entity, type, tags and node tags). With
/// `unique_tags`, dense node tags must be exactly min..=max.
fn agree_with_mshio(
    name: &str,
    ours: &MshFile,
    theirs: &mshio::MshFile<u64, i32, f64>,
    unique_tags: bool,
) {
    // Nodes. mshio keeps the tags of a block only when the file's tags are sparse; dense
    // tags are min..=max over the file.
    let nodes = theirs.data.nodes.as_ref().expect("mshio: $Nodes");
    assert_eq!(
        ours.node_blocks.len(),
        nodes.node_blocks.len(),
        "{name}: node blocks"
    );
    let mut all_tags = Vec::new();
    for (a, b) in ours.node_blocks.iter().zip(&nodes.node_blocks) {
        assert_eq!(a.dim as i32, b.entity_dim, "{name}: node block dim");
        assert_eq!(a.entity_tag, b.entity_tag, "{name}: node block entity");
        assert_eq!(a.coords.len(), b.nodes.len(), "{name}: node block length");
        for (x, y) in a.coords.iter().zip(&b.nodes) {
            assert_eq!(*x, [y.x, y.y, y.z], "{name}: node coordinates");
        }
        if let Some(map) = &b.node_tags {
            // Repeated tags (the duplicate-tag fixture) map to one of their positions.
            for &t in &a.tags {
                let i = map[&(t as u64)];
                assert_eq!(a.tags[i], t, "{name}: node tag {t}");
            }
        }
        all_tags.extend_from_slice(&a.tags);
    }
    let range = nodes.min_node_tag as i64..=nodes.max_node_tag as i64;
    assert!(
        all_tags.iter().all(|t| range.contains(t)),
        "{name}: node tag range"
    );
    let dense = nodes.node_blocks.iter().all(|b| b.node_tags.is_none());
    if dense && unique_tags {
        all_tags.sort_unstable();
        assert_eq!(
            all_tags,
            range.collect::<Vec<_>>(),
            "{name}: dense node tags"
        );
    }

    // Elements.
    let elements = theirs.data.elements.as_ref().expect("mshio: $Elements");
    assert_eq!(
        ours.element_blocks.len(),
        elements.element_blocks.len(),
        "{name}: element blocks"
    );
    for (a, b) in ours.element_blocks.iter().zip(&elements.element_blocks) {
        assert_eq!(a.dim as i32, b.entity_dim, "{name}: element block dim");
        assert_eq!(a.entity_tag, b.entity_tag, "{name}: element block entity");
        assert_eq!(
            a.element_type, b.element_type as i32,
            "{name}: element type"
        );
        assert_eq!(
            a.tags.len(),
            b.elements.len(),
            "{name}: element block length"
        );
        let k = a.nodes.len() / a.tags.len().max(1);
        for (i, e) in b.elements.iter().enumerate() {
            assert_eq!(a.tags[i], e.element_tag as i64, "{name}: element tag");
            let nodes: Vec<i64> = e.nodes.iter().map(|&n| n as i64).collect();
            assert_eq!(
                a.nodes[i * k..(i + 1) * k],
                nodes[..],
                "{name}: element nodes"
            );
        }
    }
}

/// Every binary fixture, re-encoded in both byte orders and with 4- and 8-byte `size_t`, parses
/// to the same model; re-encoded as Gmsh wrote it (little-endian, 8 bytes) it has Gmsh's bytes.
#[test]
fn binary_byte_orders_and_size_t_widths() {
    let manifest = manifest();
    let mut seen = 0;
    for f in manifest.files.iter().filter(|f| f.binary) {
        let bytes = std::fs::read(f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        let model = read(f);
        assert_eq!(
            encode::encode_binary(&model, false, 8),
            bytes,
            "{}: re-encoded bytes",
            f.file
        );
        for (big_endian, size_t) in [(false, 4), (true, 4), (true, 8)] {
            let copy = encode::encode_binary(&model, big_endian, size_t);
            let parsed = gmsh::parse(&copy).unwrap_or_else(|e| {
                panic!("{} (big-endian {big_endian}, size_t {size_t}): {e}", f.file)
            });
            assert_eq!(
                parsed, model,
                "{} (big-endian {big_endian}, size_t {size_t})",
                f.file
            );
        }
        seen += 1;
    }
    assert!(seen > 0, "no binary fixtures");
}

/// Truncated binary files never panic, and fail with `Parse` when cut inside `$Nodes` or
/// `$Elements`: every prefix of a small binary fixture, in both byte orders.
#[test]
fn binary_truncated_files() {
    let manifest = manifest();
    let f = manifest
        .files
        .iter()
        .find(|f| f.file == "two_regions_p1-binary.msh")
        .expect("two_regions_p1-binary.msh");
    let model = read(f);
    for big_endian in [false, true] {
        let bytes = encode::encode_binary(&model, big_endian, 8);
        let nodes = bytes
            .windows(7)
            .position(|w| w == b"$Nodes\n")
            .expect("$Nodes");
        let end = bytes
            .windows(13)
            .position(|w| w == b"$EndElements\n")
            .expect("$EndElements");
        // A prefix that ends right after `$EndNodes` is a file without elements.
        let between = bytes
            .windows(10)
            .position(|w| w == b"$EndNodes\n")
            .expect("$EndNodes")
            + 9;
        for len in 0..bytes.len() {
            let result = gmsh::parse(&bytes[..len]);
            if (nodes + 7..end).contains(&len) && len != between && len != between + 1 {
                assert!(
                    matches!(result, Err(gridcl_device::GridError::Parse { .. })),
                    "prefix of {len} bytes (big-endian {big_endian})"
                );
            }
        }
    }
}

/// Every corpus file, written in both encodings, parses back to the same model, and mshio reads
/// the written file with the same nodes and elements (except the files with 18-node prisms,
/// ADR-0023).
#[test]
fn written_files_parse_back_and_agree_with_mshio() {
    let manifest = manifest();
    for f in &manifest.files {
        let model = read(f);
        for encoding in [gmsh::Encoding::Ascii, gmsh::Encoding::Binary] {
            let what = format!("{} written as {encoding:?}", f.file);
            let mut bytes = Vec::new();
            model.write_to(&mut bytes, encoding).unwrap();
            let back = gmsh::parse(&bytes).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(back, model, "{what}");
            if !f.element_counts.contains_key(&MSHIO_UNREADABLE_TYPE) {
                let theirs = mshio::parse_msh_bytes(&bytes)
                    .unwrap_or_else(|e| panic!("{what}: mshio: {e:?}"));
                agree_with_mshio(&what, &model, &theirs, f.expect == Expect::Mesh);
            }
        }
    }
}
