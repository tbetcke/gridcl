//! The chunk conversion (docs/plans/M8.md, T2; ADR-0035, rules 1, 2, 4 and 5): every corpus
//! file, cut into P = 1–8 even chunks of nodes and elements and converted chunk by chunk,
//! concatenates to what `to_gmsh_mesh` gives for the whole file. Binary files are cut by
//! `MshIndex`, the others (and serendipity files, after `complete`) in memory, as the rank-0
//! path does.

use crate::common;

use std::ops::Range;

use common::{Fixture, manifest};
use gridcl_device::GridError;
use gridcl_dims::CellType;
use gridcl_io::gmsh::{
    self, BlockHeader, CellChunk, ElementBlock, ElementChunk, GmshMesh, GmshTagGroup, Layout,
    MshFile, MshIndex, NodeChunk,
};
use gridcl_parallel::Serial;

/// The even chunk r of P over n items.
fn chunk(n: usize, r: usize, p: usize) -> Range<usize> {
    r * n / p..(r + 1) * n / p
}

fn num_elements(file: &MshFile, dim: usize) -> usize {
    file.element_blocks
        .iter()
        .filter(|b| b.dim == dim)
        .map(|b| b.tags.len())
        .sum()
}

/// Where the chunks come from: the file by `MshIndex`, or a file in memory.
enum Source<'a> {
    Index(&'a MshIndex),
    Memory(&'a MshFile),
}

impl Source<'_> {
    fn nodes(&self, range: Range<usize>) -> NodeChunk {
        match self {
            Source::Index(i) => i.read_nodes(range).unwrap(),
            Source::Memory(f) => gmsh::node_range(f, range).unwrap(),
        }
    }

    fn elements(&self, dim: usize, range: Range<usize>) -> ElementChunk {
        match self {
            Source::Index(i) => i.read_elements(dim, range).unwrap(),
            Source::Memory(f) => gmsh::element_range(f, dim, range).unwrap(),
        }
    }
}

/// The concatenation of the chunks of P ranks: cell chunks per layout block, tag groups per
/// cell type, nodes.
#[derive(Default)]
struct Joined {
    cells: Vec<CellChunk>,
    groups: Vec<GmshTagGroup>,
    tags: Vec<i64>,
    coords: Vec<[f64; 3]>,
}

fn join(layout: &Layout, source: &Source<'_>, counts: (usize, [usize; 4]), p: usize) -> Joined {
    let mut out = Joined::default();
    for r in 0..p {
        let nodes = source.nodes(chunk(counts.0, r, p));
        out.tags.extend(nodes.tags);
        out.coords.extend(nodes.coords);
        let top = source.elements(layout.tdim, chunk(counts.1[layout.tdim], r, p));
        let cells = layout.cells(&top.blocks).unwrap();
        assert_eq!(cells.len(), layout.blocks.len(), "every block is declared");
        if out.cells.is_empty() {
            out.cells = cells;
        } else {
            for (into, c) in out.cells.iter_mut().zip(cells) {
                assert_eq!((into.cell, into.degree), (c.cell, c.degree));
                let first = into.tags.len() as u32;
                into.nodes.extend(c.nodes);
                into.tags.extend(c.tags);
                into.elementary.extend(c.elementary);
                into.physical
                    .extend(c.physical.iter().map(|&(i, v)| (first + i, v)));
                into.partitions.extend(c.partitions);
            }
        }
        let lower: Vec<ElementBlock> = (0..layout.tdim)
            .flat_map(|d| source.elements(d, chunk(counts.1[d], r, p)).blocks)
            .collect();
        for g in layout.tag_groups(&lower).unwrap() {
            match out.groups.iter_mut().find(|h| h.cell == g.cell) {
                Some(into) => {
                    let first = into.tags.len() as u32;
                    into.corners.extend(g.corners);
                    into.tags.extend(g.tags);
                    into.elementary.extend(g.elementary);
                    into.physical
                        .extend(g.physical.iter().map(|&(i, v)| (first + i, v)));
                }
                None => out.groups.push(g),
            }
        }
    }
    out.groups.sort_by_key(|g| g.cell);
    out
}

/// Asserts that `joined` is `serial` with node indices replaced by node tags.
fn assert_joined_equal(joined: &Joined, serial: &GmshMesh, what: &str) {
    let gdim = serial.gdim;
    assert_eq!(joined.tags, serial.node_tags, "{what}: node tags");
    let coords: Vec<f64> = joined
        .coords
        .iter()
        .flat_map(|x| x[..gdim].to_vec())
        .collect();
    assert_eq!(coords, serial.nodes, "{what}: coordinates");
    assert_eq!(joined.cells.len(), serial.blocks.len(), "{what}: blocks");
    for (c, b) in joined.cells.iter().zip(&serial.blocks) {
        let w = format!("{what}: {:?} block", b.cell);
        assert_eq!((c.cell, c.degree), (b.cell, b.degree), "{w}");
        let nodes: Vec<i64> = b
            .nodes
            .iter()
            .map(|&i| serial.node_tags[i as usize])
            .collect();
        assert_eq!(c.nodes, nodes, "{w}: node tags");
        assert_eq!(c.tags, b.tags, "{w}: element tags");
        assert_eq!(c.elementary, b.elementary, "{w}: elementary tags");
        assert_eq!(c.physical, b.physical, "{w}: physical tags");
    }
    let serial_groups: Vec<&GmshTagGroup> = serial
        .tag_groups
        .iter()
        .filter(|g| !g.tags.is_empty())
        .collect();
    let groups: Vec<&GmshTagGroup> = joined.groups.iter().collect();
    assert_eq!(groups, serial_groups, "{what}: tag groups");
}

/// The source, layout and counts of fixture `f` for the parallel paths: binary files without
/// serendipity elements by `MshIndex`, the others in memory after `complete`.
fn check_fixture(f: &Fixture) {
    let file = gmsh::read_file(&f.path()).unwrap_or_else(|e| panic!("{}: {e}", f.file));
    let serial = gmsh::to_gmsh_mesh(&file, &Serial, None).unwrap();
    let headers: Vec<BlockHeader> = file.element_blocks.iter().map(BlockHeader::of).collect();
    let layout = Layout::new(&file, &headers).unwrap();
    assert_eq!(layout.tdim, serial.tdim, "{}: D_TOP", f.file);
    let index = MshIndex::scan(&f.path()).unwrap();
    if f.binary {
        let scanned = Layout::new(index.sections(), &index.block_headers()).unwrap();
        assert_eq!(index.block_headers(), headers, "{}: headers", f.file);
        assert_eq!(
            (
                scanned.tdim,
                &scanned.blocks,
                scanned.serendipity,
                scanned.num_partitions
            ),
            (
                layout.tdim,
                &layout.blocks,
                layout.serendipity,
                layout.num_partitions
            ),
            "{}: layout from the scan",
            f.file
        );
    }
    let has_serendipity = f.element_counts.keys().any(|t| (16..=19).contains(t));
    assert_eq!(
        layout.serendipity, has_serendipity,
        "{}: serendipity",
        f.file
    );
    let completed = gmsh::complete(&file, &layout).unwrap();
    let source_file = completed.as_ref().unwrap_or(&file);
    let source = if f.binary && completed.is_none() {
        Source::Index(&index)
    } else {
        Source::Memory(source_file)
    };
    let num_nodes: usize = source_file.node_blocks.iter().map(|b| b.tags.len()).sum();
    let counts = (
        num_nodes,
        std::array::from_fn(|d| num_elements(source_file, d)),
    );
    let gdim = gmsh::geometric_dim(
        layout.tdim,
        &source.nodes(0..num_nodes).coords,
        &Serial,
        None,
    );
    if completed.is_none() {
        assert_eq!(gdim.unwrap(), serial.gdim, "{}: D_GEO", f.file);
    }
    for p in 1..=8 {
        let joined = join(&layout, &source, counts, p);
        assert_joined_equal(&joined, &serial, &format!("{} in {p} chunks", f.file));
        check_partitions(f, &joined, p);
    }
}

/// The partition of every cell is the one Gmsh reports; unpartitioned files have none.
fn check_partitions(f: &Fixture, joined: &Joined, p: usize) {
    for c in &joined.cells {
        if f.num_partitions == 0 {
            assert!(c.partitions.is_empty(), "{}: no partitions", f.file);
            continue;
        }
        assert_eq!(c.partitions.len(), c.tags.len(), "{} in {p} chunks", f.file);
        for (&tag, &q) in c.tags.iter().zip(&c.partitions) {
            assert!(
                (1..=f.num_partitions as i32).contains(&q),
                "{}: cell {tag} in partition {q}",
                f.file
            );
            assert!(
                f.partition_elements[q as usize - 1]
                    .binary_search(&tag)
                    .is_ok(),
                "{}: cell {tag} is not in Gmsh's partition {q}",
                f.file
            );
        }
    }
}

/// Every corpus mesh, in P = 1–8 chunks, converts to `to_gmsh_mesh`'s mesh.
#[test]
fn chunks_convert_to_the_serial_mesh() {
    let m = manifest();
    let mut binary = 0;
    for f in m.meshes() {
        binary += usize::from(f.binary);
        check_fixture(f);
    }
    assert!(binary > 0, "no binary fixtures");
}

/// The in-memory chunks of a binary file equal `MshIndex`'s reads.
#[test]
fn memory_chunks_equal_index_reads() {
    for f in manifest().meshes().filter(|f| f.binary) {
        let file = gmsh::read_file(&f.path()).unwrap();
        let index = MshIndex::scan(&f.path()).unwrap();
        for p in [1, 3, 8] {
            for r in 0..p {
                let range = chunk(index.num_nodes(), r, p);
                assert_eq!(
                    gmsh::node_range(&file, range.clone()).unwrap(),
                    index.read_nodes(range).unwrap(),
                    "{}: nodes",
                    f.file
                );
                for dim in 0..=3 {
                    let range = chunk(index.num_elements(dim), r, p);
                    assert_eq!(
                        gmsh::element_range(&file, dim, range.clone()).unwrap(),
                        index.read_elements(dim, range).unwrap(),
                        "{}: elements of dimension {dim}",
                        f.file
                    );
                }
            }
        }
        let n = index.num_nodes();
        assert!(matches!(
            gmsh::node_range(&file, 0..n + 1),
            Err(GridError::IndexOutOfRange { .. })
        ));
    }
}

/// Serendipity cells are refused until completed; completion keeps the file order of the
/// cells and adds its nodes after the file's.
#[test]
fn serendipity_cells_need_completion() {
    let m = manifest();
    let f = m
        .meshes()
        .find(|f| f.file == "hexahedron_s2.msh")
        .expect("hexahedron_s2.msh in the corpus");
    let file = gmsh::read_file(&f.path()).unwrap();
    let headers: Vec<BlockHeader> = file.element_blocks.iter().map(BlockHeader::of).collect();
    let layout = Layout::new(&file, &headers).unwrap();
    assert_eq!(layout.blocks, [(CellType::Hexahedron, 2)]);
    assert!(matches!(
        layout.cells(&file.element_blocks),
        Err(GridError::Parse { .. })
    ));
    let completed = gmsh::complete(&file, &layout).unwrap().expect("an upgrade");
    let before: usize = file.node_blocks.iter().map(|b| b.tags.len()).sum();
    let after: usize = completed.node_blocks.iter().map(|b| b.tags.len()).sum();
    assert!(after > before);
    assert_eq!(
        completed.node_blocks[..file.node_blocks.len()],
        file.node_blocks[..]
    );
    let tags = |f: &MshFile| -> Vec<i64> {
        f.element_blocks
            .iter()
            .flat_map(|b| b.tags.iter().copied())
            .collect()
    };
    assert_eq!(tags(&completed), tags(&file));
    // A file without serendipity cells needs no completion.
    let twin = gmsh::read_file(&f.twin_path().unwrap()).unwrap();
    let headers: Vec<BlockHeader> = twin.element_blocks.iter().map(BlockHeader::of).collect();
    assert!(
        gmsh::complete(&twin, &Layout::new(&twin, &headers).unwrap())
            .unwrap()
            .is_none()
    );
}

/// A partitioned entity with no single valid partition gives its cells partition 0.
#[test]
fn cells_without_a_single_partition() {
    let m = manifest();
    let f = m
        .meshes()
        .find(|f| f.num_partitions > 0)
        .expect("a partitioned fixture");
    let mut file = gmsh::read_file(&f.path()).unwrap();
    let headers: Vec<BlockHeader> = file.element_blocks.iter().map(BlockHeader::of).collect();
    let tdim = Layout::new(&file, &headers).unwrap().tdim;
    let parts = file.partitioned_entities.as_mut().unwrap();
    let list = [&mut parts.curves, &mut parts.surfaces, &mut parts.volumes];
    let n = parts.num_partitions as i32;
    let top = list.into_iter().nth(tdim - 1).unwrap();
    top[0].partitions = vec![1, 2];
    top[1].partitions = vec![n + 1];
    let (first, second) = (top[0].entity.tag, top[1].entity.tag);
    let layout = Layout::new(&file, &headers).unwrap();
    let cells = layout.cells(&file.element_blocks).unwrap();
    let mut seen = 0;
    for b in file.element_blocks.iter().filter(|b| b.dim == tdim) {
        let zero = b.entity_tag == first || b.entity_tag == second;
        for t in &b.tags {
            let c = cells
                .iter()
                .find_map(|c| c.tags.iter().position(|x| x == t).map(|i| c.partitions[i]));
            assert_eq!(c == Some(0), zero, "cell {t}");
            seen += usize::from(zero);
        }
    }
    assert!(seen > 0);
}
