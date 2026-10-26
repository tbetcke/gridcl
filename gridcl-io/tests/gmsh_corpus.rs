//! M4 acceptance tests at the file level (docs/design.md, "M4 Gmsh I/O (serial)"):
//! the committed corpus reads with Gmsh's node counts, covers every element type, and agrees
//! with mshio, the test oracle (ADR-0022).

mod common;

use std::collections::{BTreeMap, BTreeSet};

use common::{Fixture, REQUIRED_TYPES, manifest};
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

/// Nodes and elements agree with mshio on every fixture: node blocks (entity, coordinates and
/// tags), element blocks (entity, type, tags and node tags).
#[test]
fn nodes_and_elements_agree_with_mshio() {
    let manifest = manifest();
    for f in &manifest.files {
        let bytes = std::fs::read(f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
        let ours = read(f);
        let theirs =
            mshio::parse_msh_bytes(&bytes).unwrap_or_else(|e| panic!("{}: mshio: {e:?}", f.file));
        let name = &f.file;

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
                for (i, &t) in a.tags.iter().enumerate() {
                    assert_eq!(map.get(&(t as u64)), Some(&i), "{name}: node tag {t}");
                }
            }
            all_tags.extend_from_slice(&a.tags);
        }
        if nodes.node_blocks.iter().all(|b| b.node_tags.is_none()) && !all_tags.is_empty() {
            all_tags.sort_unstable();
            let dense: Vec<i64> = (nodes.min_node_tag as i64..=nodes.max_node_tag as i64).collect();
            assert_eq!(all_tags, dense, "{name}: dense node tags");
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
}
