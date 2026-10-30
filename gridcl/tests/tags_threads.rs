//! Physical tags in parallel on `ThreadComm` (docs/plans/M6.md, T15; ADR-0026, items 3 and
//! 10): every holder of an entity gets the host tags' values, unmatched entries fail on every
//! rank, and the name tables are agreed.

use std::collections::BTreeMap;

use gridcl::prelude::*;
use gridcl::{CommDyn, GhostMode};
use gridcl_testing::{HostChunk, HostMesh, HostTags, ThreadComm};
use gridcl_topology::decode_origin_ids;

/// Sorted distinct tag values by sorted corner key.
type Reference = BTreeMap<Vec<i64>, Vec<i32>>;

fn add(reference: &mut Reference, t: &HostTags) {
    let nv = gridcl_refcell::vertices(t.entity).len();
    for (corners, &v) in t.corners.chunks_exact(nv).zip(&t.values) {
        let mut key = corners.to_vec();
        key.sort_unstable();
        let values = reference.entry(key).or_default();
        values.push(v);
        values.sort_unstable();
        values.dedup();
    }
}

/// The chunk of `rank` with one extra point tag: the first vertex of its first cell, with
/// value 100 + rank (a vertex in several groups when ranks share it).
fn chunk_with_point(h: &HostMesh, rank: usize, size: usize) -> HostChunk {
    let mut chunk = h.chunk(rank, size);
    if let Some(b) = chunk.blocks.first() {
        chunk.tags.push(HostTags {
            entity: CellType::Point,
            corners: vec![b.node_ids[0]],
            values: vec![100 + rank as i32],
        });
    }
    chunk
}

/// The tag values of every local entity and cell of `mesh`, by sorted vertex origin ids.
fn local_values<TD, GD, C>(mesh: &mut Mesh<f32, TD, GD, C>) -> Vec<(usize, Vec<i64>, Vec<i32>)>
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let ids = decode_origin_ids(&mesh.vertex_origin_ids().to_host().unwrap());
    let mut rows: Vec<(usize, usize, Vec<u32>)> = Vec::new();
    for d in 0..TD::N {
        mesh.create_entities(d).unwrap();
        for b in mesh.entity_blocks(d).unwrap() {
            let stride = b.vertices.stride as usize;
            let data = b.vertices.data.to_host().unwrap();
            for (e, row) in data.chunks_exact(stride).enumerate() {
                rows.push((d, b.offset as usize + e, row.to_vec()));
            }
        }
    }
    let mut first = 0;
    for b in mesh.cell_blocks() {
        let stride = b.vertices.stride as usize;
        let data = b.vertices.data.to_host().unwrap();
        for (c, row) in data.chunks_exact(stride).enumerate() {
            rows.push((TD::N, first + c, row.to_vec()));
        }
        first += b.index_map.num_local();
    }
    let mut values: BTreeMap<(usize, usize), Vec<i32>> = BTreeMap::new();
    for d in 0..=TD::N {
        if let Some(t) = mesh.tags().physical(d) {
            let indices = t.indices.to_host().unwrap();
            let v = t.values.to_host().unwrap();
            for (&i, &v) in indices.iter().zip(&v) {
                values.entry((d, i as usize)).or_default().push(v);
            }
        }
    }
    rows.into_iter()
        .map(|(d, i, row)| {
            let mut key: Vec<i64> = row.iter().map(|&v| ids[v as usize]).collect();
            key.sort_unstable();
            (d, key, values.remove(&(d, i)).unwrap_or_default())
        })
        .collect()
}

/// Every local entity, owned or ghost, has the values of the host tags, on 1–3 ranks and
/// in every ghost mode.
fn ghosts_like_owners<TD, GD>(name: &str, h: &HostMesh)
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let modes = [
        GhostMode::None,
        GhostMode::SharedFacet,
        GhostMode::SharedVertex { layers: 1 },
    ];
    for p in 1..=3 {
        let mut reference = Reference::new();
        for r in 0..p {
            chunk_with_point(h, r, p)
                .tags
                .iter()
                .for_each(|t| add(&mut reference, t));
        }
        for mode in modes {
            ThreadComm::run(p, |comm| {
                let client = gridcl_testing::test_client();
                let chunk = chunk_with_point(h, comm.rank(), p);
                let mut mesh = chunk
                    .builder::<f32, TD, GD>(&client)
                    .unwrap()
                    .ghosts(mode)
                    .build_on(comm.clone())
                    .unwrap();
                let names: Vec<_> = mesh
                    .tags()
                    .names()
                    .map(|(n, d, v)| (n.to_owned(), d, v))
                    .collect();
                assert_eq!(names, h.tag_names(), "{name}, {p} ranks");
                for (d, key, values) in local_values(&mut mesh) {
                    let want = reference.get(&key).cloned().unwrap_or_default();
                    assert_eq!(
                        values,
                        want,
                        "{name}, {p} ranks, {mode:?}, rank {}: dimension {d}, {key:?}",
                        comm.rank()
                    );
                }
            });
        }
    }
}

#[test]
fn tags_ghost_values_equal_the_owners() {
    ghosts_like_owners::<D2, D2>(
        "quadrilaterals",
        &gridcl_testing::square_box(Cell2::Quadrilateral, 3, 1),
    );
    ghosts_like_owners::<D3, D3>("mixed", &gridcl_testing::mixed_box(2, 1));
    ghosts_like_owners::<D3, D3>(
        "tetrahedra p2, shuffled ids",
        &gridcl_testing::cube_box(Cell3::Tetrahedron, 2, 2).with_shuffled_ids(5),
    );
    ghosts_like_owners::<D2, D3>("sphere", &gridcl_testing::sphere(1, 1));
}

/// A tag entry that matches nothing gives the same `UnmatchedTagElement` on every rank: a
/// key whose corners no rank holds together, an unknown id, and a diagonal of a quadrilateral
/// (corners held, no edge).
#[test]
fn tags_unmatched_key_on_every_rank() {
    let h = gridcl_testing::square_box(Cell2::Quadrilateral, 3, 1);
    let p = 3;
    let total: usize = (0..p)
        .flat_map(|r| h.chunk(r, p).tags)
        .map(|t| t.values.len())
        .sum();
    let first = h.chunk(0, p).blocks[0].node_ids.clone();
    let last = h.chunk(p - 1, p).blocks[0].node_ids.clone();
    for (what, corners) in [
        ("far apart", vec![first[0], *last.last().unwrap()]),
        ("unknown", vec![first[0], 1 << 40]),
        ("diagonal", vec![last[0], last[3]]),
    ] {
        let out = ThreadComm::run(p, |comm| {
            let client = gridcl_testing::test_client();
            let mut builder = h
                .chunk(comm.rank(), p)
                .builder::<f32, D2, D2>(&client)
                .unwrap();
            if comm.rank() == p - 1 {
                // The second entry is unmatched; the first matches an edge of the chunk.
                let corners = [&last[..2], &corners[..]].concat();
                builder = builder.physical_tags(CellType::Interval, &corners, &[8, 9]);
            }
            builder.build_on(comm).map(|_| ())
        });
        for (r, o) in out.iter().enumerate() {
            assert!(
                matches!(o, Err(GridError::UnmatchedTagElement { dim: 1, tag }) if *tag == total as i64 + 1),
                "{what}, rank {r}: {o:?}"
            );
        }
    }
}

/// The name tables of the ranks are agreed: rank 0's, then the new entries of the others.
#[test]
fn tags_names_are_agreed() {
    let out = ThreadComm::run(3, |comm| {
        let client = gridcl_testing::test_client();
        let tri = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        let names: &[(&str, usize, i32)] = match comm.rank() {
            0 => &[("a", 1, 1), ("b", 2, 2)],
            1 => &[("c", 1, 3), ("a", 1, 1), ("ä", 0, 4)],
            _ => &[],
        };
        let (coords, ids, cells): (&[f64], &[i64], &[i64]) = if comm.rank() == 0 {
            (&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], &[0, 1, 2], &[0, 1, 2])
        } else {
            (&[], &[], &[])
        };
        let mesh: Mesh<f32, D2, D2, ThreadComm> = MeshBuilder::new(&client)
            .nodes(coords, Some(ids))
            .cell_block_by_ids(tri, cells, None)
            .tag_names(names)
            .build_on(comm)
            .unwrap();
        let names: Vec<(String, usize, i32)> = mesh
            .tags()
            .names()
            .map(|(n, d, v)| (n.to_owned(), d, v))
            .collect();
        (names, mesh.tags().group_size(1, 3))
    });
    let want =
        [("a", 1, 1), ("b", 2, 2), ("c", 1, 3), ("ä", 0, 4)].map(|(n, d, v)| (n.to_owned(), d, v));
    for (names, size) in out {
        assert_eq!(names, want);
        assert_eq!(size, Some(0));
    }
}
