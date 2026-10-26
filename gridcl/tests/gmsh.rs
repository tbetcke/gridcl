//! M4 acceptance tests at the mesh level (docs/design.md, "M4 Gmsh I/O (serial)"; ADR-0022):
//! the corpus builds meshes with the nodes of its D_TOP elements, tags match Gmsh's physical
//! groups, D_GEO is inferred, duplicate tags are rejected, and read → write → read is the
//! identity.

mod common;

use std::collections::BTreeSet;

use common::gmsh::{
    Expect, Fixture, assert_same, entity_vertex_ids, manifest, read, read_fixture, snapshot_any,
    temp_path,
};
use common::{client, name, tol};
use cubecl::prelude::Client;
use gridcl::prelude::*;

/// Every mesh fixture builds, with the nodes of its D_TOP elements (the complete-P2 twin's
/// count for serendipity files) and the cells of its D_TOP elements.
fn builds_with_node_counts<F: Real>(client: &Client) {
    let manifest = manifest();
    for f in manifest.meshes() {
        let any = read_fixture::<F>(client, f);
        let (nodes, cells) = with_mesh!(any.as_ref(), m => (
            m.node_index_map().num_local(),
            m.cell_blocks().iter().map(|b| b.index_map.num_local()).sum::<usize>(),
        ));
        assert_eq!(
            nodes,
            f.num_mesh_nodes,
            "{} ({}): mesh nodes",
            f.file,
            name::<F>()
        );
        let want_cells: usize = f
            .element_counts
            .iter()
            .filter(|(ty, _)| {
                gridcl_refcell::tdim(gridcl_io::gmsh::element_type(**ty).expect("A.7 type").cell)
                    == f.tdim
            })
            .map(|(_, n)| n)
            .sum();
        assert_eq!(cells, want_cells, "{} ({}): cells", f.file, name::<F>());
    }
}

#[test]
fn corpus_builds_meshes_with_gmsh_node_counts() {
    for_each_precision!(client(), builds_with_node_counts);
}

/// The physical tags of every dimension, read back, are in range, unique per value and never
/// `u32::MAX`; each group has Gmsh's size and the entities whose sorted vertex origin ids are
/// Gmsh's sorted corner node tags; named groups are found by name.
fn tags_match_groups<F: Real>(client: &Client) {
    let manifest = manifest();
    let mut groups_seen = 0;
    for f in manifest.meshes() {
        let any = read_fixture::<F>(client, f);
        let what = format!("{} ({})", f.file, name::<F>());
        with_mesh!(any.as_ref(), m => {
            let tags = m.tags();
            for g in &f.physical_groups {
                groups_seen += 1;
                let physical = tags
                    .physical(g.dim)
                    .unwrap_or_else(|| panic!("{what}: no physical tags of dimension {}", g.dim));
                let keys = common::gmsh::tag_keys(m, physical);
                let mut got: Vec<Vec<i64>> = keys
                    .into_iter()
                    .filter(|(_, _, v)| *v == g.tag)
                    .map(|(_, k, _)| k)
                    .collect();
                got.sort_unstable();
                assert_eq!(got.len(), g.size, "{what}: size of group {} (dim {})", g.tag, g.dim);
                assert_eq!(got, g.elements, "{what}: entities of group {} (dim {})", g.tag, g.dim);
                assert_eq!(tags.group_size(g.dim, g.tag), Some(g.size), "{what}: group_size");
                let selected = tags.group(g.dim, g.tag).expect("group");
                assert_eq!(selected.indices.len(), g.size, "{what}: group({})", g.tag);
                if let Some(n) = &g.name {
                    assert_eq!(tags.lookup(g.dim, n), Some(g.tag), "{what}: lookup {n}");
                    let codim = [Codim::Vol, Codim::Bnd, Codim::BBnd, Codim::BBBnd]
                        [f.tdim - g.dim];
                    let named = tags.named(codim, n).expect("named");
                    assert_eq!(named.dim, g.dim, "{what}: named {n}");
                    assert_eq!(named.indices.len(), g.size, "{what}: named {n}");
                    let ents = entity_vertex_ids(m, g.dim);
                    let mut via_name: Vec<Vec<i64>> = named
                        .indices
                        .to_host()
                        .expect("indices")
                        .iter()
                        .map(|&i| ents[i as usize].clone())
                        .collect();
                    via_name.sort_unstable();
                    assert_eq!(via_name, g.elements, "{what}: named {n}");
                }
            }
        });
    }
    assert!(groups_seen > 0, "no physical groups in the corpus");
}

#[test]
fn tag_counts_equal_physical_group_sizes() {
    for_each_precision!(client(), tags_match_groups);
}

/// (D_TOP, D_GEO) of every mesh fixture, covering 1D in 1D, 2D and 3D, 2D in 2D and 3D, and
/// 3D; a fixed `gdim` raises the embedding, and one below the inferred D_GEO or D_TOP fails.
fn gdim_inferred<F: Real>(client: &Client) {
    let manifest = manifest();
    let mut seen = BTreeSet::new();
    for f in manifest.meshes() {
        let any = read_fixture::<F>(client, f);
        assert_eq!(
            (any.tdim(), any.gdim()),
            (f.tdim, f.gdim),
            "{} ({}): (D_TOP, D_GEO)",
            f.file,
            name::<F>()
        );
        seen.insert((f.tdim, f.gdim));
    }
    for want in [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)] {
        assert!(
            seen.contains(&want),
            "no fixture with (D_TOP, D_GEO) = {want:?}"
        );
    }
    let opts = |g| GmshOptions::default().gdim(g);
    let planar: &Fixture = manifest
        .meshes()
        .find(|f| (f.tdim, f.gdim) == (2, 2))
        .expect("a planar fixture");
    let raised = gmsh::read::<F, Serial>(client, Serial, planar.path(), opts(3)).expect("gdim 3");
    assert_eq!(
        (raised.tdim(), raised.gdim()),
        (2, 3),
        "{}: gdim(3)",
        planar.file
    );
    for g in [0, 1, 4] {
        let r = gmsh::read::<F, Serial>(client, Serial, planar.path(), opts(g));
        assert!(
            matches!(r, Err(GridError::DimensionMismatch { .. })),
            "{}: gdim({g}) must fail with DimensionMismatch",
            planar.file
        );
    }
    let spatial = manifest
        .meshes()
        .find(|f| (f.tdim, f.gdim) == (2, 3))
        .expect("a surface fixture");
    let r = gmsh::read::<F, Serial>(client, Serial, spatial.path(), opts(2));
    assert!(
        matches!(r, Err(GridError::DimensionMismatch { .. })),
        "{}: gdim(2) below the inferred D_GEO must fail",
        spatial.file
    );
}

#[test]
fn gdim_inferred_on_1d_2d_3d_fixtures() {
    for_each_precision!(client(), gdim_inferred);
}

/// Duplicate node tags and duplicate element tags give `DuplicateOriginId` with the repeated
/// tag.
#[test]
fn duplicate_tags_rejected() {
    let client = client();
    let manifest = manifest();
    let (mut nodes, mut elements) = (0, 0);
    for f in &manifest.files {
        let Expect::DuplicateOriginId {
            id,
            nodes: of_nodes,
        } = f.expect
        else {
            continue;
        };
        if of_nodes {
            nodes += 1;
        } else {
            elements += 1;
        }
        match read::<f32>(&client, &f.path()) {
            Err(GridError::DuplicateOriginId(got)) => assert_eq!(got, id, "{}", f.file),
            Err(e) => panic!("{}: expected DuplicateOriginId({id}), got {e}", f.file),
            Ok(_) => panic!("{}: expected DuplicateOriginId({id}), got a mesh", f.file),
        }
    }
    assert!(nodes > 0, "no duplicate-node-tag fixture");
    assert!(elements > 0, "no duplicate-element-tag fixture");
}

/// read → write → read gives identical topology and coordinates to 10^−15 relative to the
/// largest absolute coordinate in f64 (10^−5 in f32), in both encodings (ADR-0022, items 7, 8).
fn round_trip<F: Real>(client: &Client) {
    let manifest = manifest();
    let tolerance = if size_of::<F>() == 8 {
        1e-15
    } else {
        tol::<F>()
    };
    for f in manifest.meshes() {
        let first = read_fixture::<F>(client, f);
        let a = snapshot_any(&first);
        for binary in [false, true] {
            let what = format!(
                "{} ({}, {})",
                f.file,
                name::<F>(),
                if binary { "binary" } else { "ASCII" }
            );
            let path = temp_path(&format!("{}-{}-{binary}", name::<F>(), f.file));
            gmsh::GmshWriter::new(&path)
                .binary(binary)
                .write_any(first.as_ref())
                .unwrap_or_else(|e| panic!("{what}: write: {e}"));
            let second =
                read::<F>(client, &path).unwrap_or_else(|e| panic!("{what}: re-read: {e}"));
            let _ = std::fs::remove_file(&path);
            assert_same(&a, &snapshot_any(&second), tolerance, &what);
        }
    }
}

#[test]
fn read_write_read_round_trip() {
    for_each_precision!(client(), round_trip);
}
