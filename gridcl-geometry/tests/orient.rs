//! `orient`, steps 1 to 3 (M9 T7, T8): the neighbour table of `orientation_neighbours` has
//! parity 0 between consistently oriented cells and 1 across every flipped cell, and its
//! records per (cell origin id, local facet) do not depend on the partition or the ghost mode;
//! label propagation (`orientation_components`) labels each component by its smallest cell
//! origin id, its signs satisfy every facet constraint on any partition, and pointer jumping
//! keeps the rounds logarithmic on a strip.

use std::collections::BTreeMap;

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_device::DeviceStrided;
use gridcl_geometry::{OrientationComponents, orientation_components, orientation_neighbours};
use gridcl_parallel::Comm;
use gridcl_testing::parallel::{self, Case, Coverage, Family};
use gridcl_testing::{HostMesh, ThreadComm};

const NONE: u32 = u32::MAX;

/// The mesh of `h` on one rank with the facet entities and (tdim − 1, tdim) connectivity.
fn build<TD, GD>(client: &Client, h: &HostMesh) -> Mesh<f32, TD, GD>
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let mut mesh = h.build::<f32, TD, GD>(client).unwrap();
    mesh.create_connectivity(TD::N - 1, TD::N).unwrap();
    mesh
}

/// The orientation sign of every local cell, flat over the blocks, from its vertices: the
/// sign of (x1 − x0) × (x2 − x0) · x0 for surfaces in 3D (the normal of the boundary cycle,
/// against a point of the cell), of x0 × x1 for curves in 2D. On a surface or curve around
/// the origin, consistently oriented cells have equal signs.
fn signs<TD, GD, C>(m: &Mesh<f32, TD, GD, C>) -> Vec<bool>
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let gd = GD::N;
    let nodes = m.nodes().to_host().unwrap();
    let origin = m.origin();
    let x = |n: u32| -> Vec<f64> {
        (0..gd)
            .map(|g| f64::from(nodes[n as usize * gd + g]) + origin.as_ref()[g])
            .collect()
    };
    let mut out = Vec::new();
    for b in 0..m.cell_blocks().len() {
        let dofmap = &m.geometry_block(b).dofmap;
        for row in dofmap
            .data
            .to_host()
            .unwrap()
            .chunks_exact(dofmap.stride as usize)
        {
            let (x0, x1) = (x(row[0]), x(row[1]));
            let s = if gd == 2 {
                x0[0] * x1[1] - x0[1] * x1[0]
            } else {
                let x2 = x(row[2]);
                let (a, b): (Vec<f64>, Vec<f64>) =
                    (0..3).map(|g| (x1[g] - x0[g], x2[g] - x0[g])).unzip();
                let n = [
                    a[1] * b[2] - a[2] * b[1],
                    a[2] * b[0] - a[0] * b[2],
                    a[0] * b[1] - a[1] * b[0],
                ];
                (0..3).map(|g| n[g] * x0[g]).sum()
            };
            out.push(s > 0.0);
        }
    }
    out
}

/// Every record of `table`, flat: (cell, local facet, record).
fn records(table: &[DeviceStrided<u32>]) -> Vec<(usize, usize, u32)> {
    let mut out = Vec::new();
    let mut cell = 0;
    for block in table {
        let k = block.stride as usize;
        for row in block.data.to_host().unwrap().chunks_exact(k) {
            for (e, &r) in row.iter().enumerate() {
                out.push((cell, e, r));
            }
            cell += 1;
        }
    }
    out
}

/// Checks the parity of every neighbour record of `h` against the cells' signs, and that a
/// closed surface or curve has no facet without a neighbour.
fn check_parity<TD, GD>(client: &Client, h: &HostMesh, what: &str)
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    let mesh = build::<TD, GD>(client, h);
    let s = signs(&mesh);
    let table = orientation_neighbours(&mesh).unwrap();
    for (cell, e, r) in records(&table) {
        assert_ne!(r, NONE, "{what}: cell {cell} facet {e} has no neighbour");
        let other = (r >> 1) as usize;
        assert!(
            other < s.len(),
            "{what}: cell {cell} facet {e}: remote on one rank"
        );
        assert_eq!(
            r & 1 == 1,
            s[cell] != s[other],
            "{what}: parity between cells {cell} and {other}"
        );
    }
}

#[test]
fn parity_is_zero_between_consistent_cells_and_one_across_flips() {
    let client = gridcl_testing::test_client();
    for p in 1..=3 {
        let surfaces = [
            ("sphere", gridcl_testing::sphere(1, p)),
            ("mixed surface", gridcl_testing::mixed_surface(1, p)),
        ];
        for (what, h) in surfaces {
            check_parity::<D2, D3>(&client, &h, &format!("{what} p={p}"));
            let f = h.with_flipped_cells(3 + u64::from(p), 0.5);
            assert_ne!(f, h);
            check_parity::<D2, D3>(&client, &f, &format!("flipped {what} p={p}"));
        }
        let h = gridcl_testing::circle(12, p);
        check_parity::<D1, D2>(&client, &h, &format!("circle p={p}"));
        let f = h.with_flipped_cells(5, 0.5);
        assert_ne!(f, h);
        check_parity::<D1, D2>(&client, &f, &format!("flipped circle p={p}"));
    }
}

/// The interleaved (hi ^ 0x8000_0000, lo) origin ids of every local cell, flat.
fn cell_ids<TD, GD, C>(m: &Mesh<f32, TD, GD, C>) -> Vec<i64>
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let mut out = Vec::new();
    for b in m.cell_blocks() {
        for [hi, lo] in b.origin_ids.to_host().unwrap().as_chunks::<2>().0 {
            out.push(((u64::from(hi ^ 0x8000_0000) << 32) | u64::from(*lo)) as i64);
        }
    }
    out
}

/// (origin id, local facet) → (parity, remote) of the owned cells of a mesh of `h` under
/// `case` on `comm`; every owned cell's facets must have a neighbour (closed surfaces).
fn owned_parities<TD, GD, C>(
    comm: C,
    client: &Client,
    h: &HostMesh,
    case: &Case,
) -> BTreeMap<(i64, usize), (u32, bool)>
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let mut mesh: Mesh<f32, TD, GD, C> =
        parallel::build(comm, client, h, case, CellOrdering::Input).unwrap();
    mesh.create_connectivity(TD::N - 1, TD::N).unwrap();
    let ids = cell_ids(&mesh);
    let mut owned = vec![false; ids.len()];
    let mut offset = 0;
    for b in mesh.cell_blocks() {
        owned[offset..offset + b.index_map.num_owned()].fill(true);
        offset += b.index_map.num_local();
    }
    let table = orientation_neighbours(&mesh).unwrap();
    let mut out = BTreeMap::new();
    for (cell, e, r) in records(&table) {
        if owned[cell] {
            assert_ne!(r, NONE, "{case}: owned cell {} facet {e}", ids[cell]);
            out.insert((ids[cell], e), (r & 1, (r >> 1) as usize >= ids.len()));
        }
    }
    out
}

/// The parities of a flipped closed surface or curve on 1–4 threads, in every ghost mode,
/// equal one rank's; process-boundary facets get theirs from the side exchange.
#[test]
fn parities_match_one_rank() {
    let client = gridcl_testing::test_client();
    let cases: Vec<Case> = parallel::orient_cases(Coverage::Full)
        .into_iter()
        .filter(|c| [Family::Sphere, Family::MixedSurface, Family::Circle].contains(&c.family))
        .collect();
    for case in &cases {
        let h = case.family.host(case.degree).with_flipped_cells(23, 0.5);
        macro_rules! run {
            ($td:ty, $gd:ty) => {{
                let one = owned_parities::<$td, $gd, _>(Serial, &client, &h, case);
                let mut remote = 0;
                for n in 1..=4 {
                    let parts = ThreadComm::run(n, |comm| {
                        let client = gridcl_testing::test_client();
                        owned_parities::<$td, $gd, _>(comm, &client, &h, case)
                    });
                    let mut all = BTreeMap::new();
                    for part in parts {
                        for (key, (parity, r)) in part {
                            remote += usize::from(r);
                            assert!(all.insert(key, parity).is_none(), "{case}: {key:?} twice");
                        }
                    }
                    let want: BTreeMap<_, _> = one.iter().map(|(k, (p, _))| (*k, *p)).collect();
                    assert_eq!(all, want, "{case}, {n} ranks");
                }
                assert!(
                    remote > 0 || case.ghosts != GhostMode::None,
                    "{case}: no remote facet"
                );
            }};
        }
        match case.family {
            Family::Circle => run!(D1, D2),
            _ => run!(D2, D3),
        }
    }
}

/// (label, sign) of every local cell, flat, and the sign of every slot (cells, then remote
/// slots) of a propagation state.
fn decode(c: &OrientationComponents, num_cells: usize) -> (Vec<(i64, bool)>, Vec<u32>) {
    let state = c.state.to_host().unwrap();
    let words = state.as_chunks::<4>().0;
    let cells = words[..num_cells]
        .iter()
        .map(|[hi, lo, s, _]| {
            let id = ((u64::from(hi ^ 0x8000_0000) << 32) | u64::from(*lo)) as i64;
            (id, s & 1 == 1)
        })
        .collect();
    (cells, words.iter().map(|w| w[2] & 1).collect())
}

/// Checks that the signs of the owned cells of `mesh` satisfy every facet constraint of the
/// neighbour table, remote slots included, and returns (origin id, label, sign) per owned cell
/// and the number of rounds.
fn owned_components<TD, GD, C>(
    mesh: &Mesh<f32, TD, GD, C>,
    what: &str,
) -> (Vec<(i64, i64, bool)>, usize)
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let ids = cell_ids(mesh);
    let table = orientation_neighbours(mesh).unwrap();
    let components = orientation_components(mesh).unwrap();
    let (cells, slot_signs) = decode(&components, ids.len());
    let mut owned = vec![false; ids.len()];
    let mut offset = 0;
    for b in mesh.cell_blocks() {
        owned[offset..offset + b.index_map.num_owned()].fill(true);
        offset += b.index_map.num_local();
    }
    for (cell, e, r) in records(&table) {
        if owned[cell] && r != NONE {
            let other = (r >> 1) as usize;
            assert_eq!(
                u32::from(cells[cell].1) ^ slot_signs[other],
                r & 1,
                "{what}: cell {} facet {e}: s_a ⊕ s_b ≠ parity",
                ids[cell]
            );
        }
    }
    let out = (0..ids.len())
        .filter(|&c| owned[c])
        .map(|c| (ids[c], cells[c].0, cells[c].1))
        .collect();
    (out, components.rounds)
}

/// The offset of the second copy's ids in [`twice`].
const OFFSET: i64 = 1 << 40;

/// Two copies of `h`, the second shifted by 10 in x, with distinct node and cell ids (the
/// second copy's are offset by [`OFFSET`]).
fn twice(h: &HostMesh) -> HostMesh {
    let n = h.num_nodes();
    let gd = h.gdim;
    let mut nodes = h.nodes.clone();
    nodes.extend(h.nodes.chunks_exact(gd).flat_map(|x| {
        let mut x = x.to_vec();
        x[0] += 10.0;
        x
    }));
    let ids: Vec<i64> = h
        .node_ids
        .clone()
        .unwrap_or_else(|| (0..n as i64).collect());
    let node_ids = ids
        .iter()
        .copied()
        .chain(ids.iter().map(|i| i + OFFSET))
        .collect();
    let blocks = h
        .blocks
        .iter()
        .map(|b| {
            let mut b = b.clone();
            let count = b.cells.len() / b.nodes_per_cell();
            let ids = b
                .cell_ids
                .clone()
                .unwrap_or_else(|| (0..count as i64).collect());
            let shifted: Vec<u32> = b.cells.iter().map(|&v| v + n as u32).collect();
            b.cells.extend(shifted);
            b.cell_ids = Some(
                ids.iter()
                    .copied()
                    .chain(ids.iter().map(|i| i + OFFSET))
                    .collect(),
            );
            b
        })
        .collect();
    HostMesh {
        gdim: gd,
        nodes,
        node_ids: Some(node_ids),
        blocks,
    }
}

/// One rank: every cell of each closed surface or curve carries the smallest cell origin id
/// of its component as label, and its sign is its flip relative to that cell: consistent with
/// the geometric orientation of the cells.
fn check_labels<TD, GD>(client: &Client, h: &HostMesh, what: &str)
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
{
    // One component.
    let mesh = build::<TD, GD>(client, h);
    let s = signs(&mesh);
    let (cells, _) = owned_components(&mesh, what);
    let min = cells.iter().map(|c| c.0).min().unwrap();
    let root = cells.iter().position(|c| c.0 == min).unwrap();
    for (c, &(id, label, sign)) in cells.iter().enumerate() {
        assert_eq!(label, min, "{what}: label of cell {id}");
        assert_eq!(sign, s[c] != s[root], "{what}: sign of cell {id}");
    }
    // Two components.
    let mesh = build::<TD, GD>(client, &twice(h));
    let (cells, _) = owned_components(&mesh, what);
    let min = |second: bool| {
        cells
            .iter()
            .filter(|c| (c.0 >= OFFSET) == second)
            .map(|c| c.0)
            .min()
            .unwrap()
    };
    let (first, second) = (min(false), min(true));
    for &(id, label, _) in &cells {
        let want = if id >= OFFSET { second } else { first };
        assert_eq!(label, want, "{what}, twice: label of cell {id}");
    }
}

#[test]
fn labels_are_the_smallest_origin_id_of_each_component() {
    let client = gridcl_testing::test_client();
    for p in 1..=3 {
        let surfaces = [
            ("sphere", gridcl_testing::sphere(1, p)),
            ("mixed surface", gridcl_testing::mixed_surface(1, p)),
        ];
        for (what, h) in surfaces {
            let f = h.with_flipped_cells(7 + u64::from(p), 0.5);
            check_labels::<D2, D3>(&client, &f, &format!("flipped {what} p={p}"));
        }
        let f = gridcl_testing::circle(12, p).with_flipped_cells(9, 0.5);
        check_labels::<D1, D2>(&client, &f, &format!("flipped circle p={p}"));
    }
}

/// (origin id, label, sign) of the owned cells of a mesh of `h` under `case` on `comm`.
fn case_components<TD, GD, C>(
    comm: C,
    client: &Client,
    h: &HostMesh,
    case: &Case,
) -> Vec<(i64, i64, bool)>
where
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    let mut mesh: Mesh<f32, TD, GD, C> =
        parallel::build(comm, client, h, case, CellOrdering::Input).unwrap();
    mesh.create_connectivity(TD::N - 1, TD::N).unwrap();
    owned_components(&mesh, &format!("{case}")).0
}

/// Labels and signs per cell origin id on 1–4 threads, in every ghost mode, equal one
/// rank's, and the signs of every rank satisfy its facet constraints.
#[test]
fn signs_satisfy_every_facet_constraint_on_threads() {
    let client = gridcl_testing::test_client();
    let cases: Vec<Case> = parallel::orient_cases(Coverage::Full)
        .into_iter()
        .filter(|c| [Family::Sphere, Family::MixedSurface, Family::Circle].contains(&c.family))
        .collect();
    for case in &cases {
        let h = case.family.host(case.degree).with_flipped_cells(29, 0.5);
        macro_rules! run {
            ($td:ty, $gd:ty) => {{
                let one: BTreeMap<i64, (i64, bool)> =
                    case_components::<$td, $gd, _>(Serial, &client, &h, case)
                        .into_iter()
                        .map(|(id, l, s)| (id, (l, s)))
                        .collect();
                for n in 1..=4 {
                    let parts = ThreadComm::run(n, |comm| {
                        let client = gridcl_testing::test_client();
                        case_components::<$td, $gd, _>(comm, &client, &h, case)
                    });
                    let mut all = BTreeMap::new();
                    for (id, l, s) in parts.into_iter().flatten() {
                        assert!(all.insert(id, (l, s)).is_none(), "{case}: cell {id} twice");
                    }
                    assert_eq!(all, one, "{case}, {n} ranks");
                }
            }};
        }
        match case.family {
            Family::Circle => run!(D1, D2),
            _ => run!(D2, D3),
        }
    }
}

/// On a strip whose cell ids increase along it, pointer jumping needs O(log n) rounds, not
/// O(n).
#[test]
fn rounds_are_logarithmic_on_a_strip() {
    let client = gridcl_testing::test_client();
    for n in [64usize, 1024] {
        let h = gridcl_testing::triangle_strip(n)
            .mapped(3, |x| vec![x[0], x[1], 0.0])
            .with_flipped_cells(31, 0.5);
        let mesh = build::<D2, D3>(&client, &h);
        let (cells, rounds) = owned_components(&mesh, &format!("strip {n}"));
        assert!(cells.iter().all(|c| c.1 == 0), "strip {n}: labels");
        let bound = 2 * n.ilog2() as usize + 4;
        assert!(rounds <= bound, "strip {n}: {rounds} rounds > {bound}");
    }
}
