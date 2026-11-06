//! `orient`, steps 1 and 2 (M9 T7): the neighbour table of `orientation_neighbours` has
//! parity 0 between consistently oriented cells and 1 across every flipped cell, and its
//! records per (cell origin id, local facet) do not depend on the partition or the ghost mode.

use std::collections::BTreeMap;

use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_device::DeviceStrided;
use gridcl_geometry::orientation_neighbours;
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
