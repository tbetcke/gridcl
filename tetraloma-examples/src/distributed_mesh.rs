//! A distributed mesh: partitioners, ghost modes and cell ordering (the book, tutorial
//! "Distributed meshes").
//!
//! The mesh is the unit square as 8 × 8 quadrilaterals. Every rank passes its share of the
//! nodes and cells to a [`MeshBuilder`], as a program that reads its own input format would,
//! and [`MeshBuilder::build_on`] distributes them: it partitions the cells, adds ghost cells,
//! orders the owned cells and numbers everything.

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Cells per side of the unit square.
const N: usize = 8;

/// Builds a distributed mesh in `F` on `client` over `comm` and prints its partition, ghosts
/// and ordering. Collective over `comm`.
///
/// The mesh is the unit square as 8 × 8 quadrilaterals, built several times: with the default
/// options, with the [`Block`] partitioner and with a round-robin partitioner of its own, with
/// each [`GhostMode`], and with each [`CellOrdering`]. Every time `run` checks that the ranks
/// own the 64 cells and 81 vertices once between them and that the area is 1. It also checks
/// that `GhostMode::None` gives no ghost cells and that each further mode gives at least as
/// many as the one before, and that the cell ordering changes the order of a rank's owned
/// cells but not which cells it owns.
///
/// # Errors
///
/// Allocation and device errors, and the errors of the calls it makes.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real, C: Comm>(client: &Client, comm: C) -> tetraloma::Result<()> {
    let (rank, size) = (comm.rank(), comm.size());
    if rank == 0 {
        println!(
            "the unit square as {N} × {N} quadrilaterals on {size} rank(s), on {} in {}",
            client.name(),
            crate::scalar_name::<F>()
        );
    }

    // ANCHOR: build_on
    // Collective: every rank passes its share and gets its part of the mesh.
    let mesh: PlanarMesh<F, C> = square::<F>(client, &comm).build_on(comm.clone())?;
    // ANCHOR_END: build_on
    let (owned, ghosts) = check(&mesh)?;
    let vertices = mesh.vertex_index_map();
    println!(
        "rank {rank}: {owned} owned and {ghosts} ghost cells, {} owned and {} ghost vertices",
        vertices.num_owned(),
        vertices.num_ghosts()
    );

    // ANCHOR: partitioners
    let block: PlanarMesh<F, C> = square::<F>(client, &comm)
        .partitioner(Block) // the default
        .ghosts(GhostMode::SharedFacet)
        .build_on(comm.clone())?;
    let round_robin: PlanarMesh<F, C> = square::<F>(client, &comm)
        .partitioner(RoundRobin)
        .ghosts(GhostMode::SharedFacet)
        .build_on(comm.clone())?;
    // ANCHOR_END: partitioners
    for (name, mesh) in [("block", &block), ("round-robin", &round_robin)] {
        let (owned, ghosts) = check(mesh)?;
        println!("rank {rank}: {name}: {owned} owned and {ghosts} ghost cells");
    }

    // ANCHOR: ghosts
    let modes = [
        GhostMode::None,
        GhostMode::SharedFacet,
        GhostMode::SharedVertex { layers: 1 },
        GhostMode::SharedVertex { layers: 2 },
    ];
    let mut ghost_cells = Vec::new();
    for mode in modes {
        let mesh: PlanarMesh<F, C> = square::<F>(client, &comm)
            .ghosts(mode)
            .build_on(comm.clone())?;
        let (owned, ghosts) = check(&mesh)?;
        println!("rank {rank}: {mode:?}: {owned} owned and {ghosts} ghost cells");
        ghost_cells.push(ghosts);
    }
    // ANCHOR_END: ghosts
    assert_eq!(ghost_cells[0], 0, "GhostMode::None gave ghost cells");
    assert!(
        ghost_cells.windows(2).all(|w| w[0] <= w[1]),
        "a wider ghost mode gave fewer ghost cells: {ghost_cells:?}"
    );
    let facet_ghosts = comm.sum(ghost_cells[1] as u64);
    assert_eq!(
        facet_ghosts > 0,
        size > 1,
        "ghost cells across facets on {size} rank(s): {facet_ghosts}"
    );

    // ANCHOR: ordering
    let mut owned_sets = Vec::new();
    for ordering in [
        CellOrdering::Input,
        CellOrdering::Hilbert,
        CellOrdering::ReverseCuthillMcKee,
    ] {
        let mesh: PlanarMesh<F, C> = square::<F>(client, &comm)
            .ordering(ordering)
            .build_on(comm.clone())?;
        let ids = owned_cell_ids(&mesh)?; // a host read, for inspection
        // How far apart consecutive cells lie, in cell widths: smaller is more local.
        let path: f64 = ids
            .windows(2)
            .map(|w| {
                let (a, b) = (w[0], w[1]);
                let dx = (a % N) as f64 - (b % N) as f64;
                let dy = (a / N) as f64 - (b / N) as f64;
                dx.hypot(dy)
            })
            .sum();
        let first = &ids[..ids.len().min(8)];
        println!("rank {rank}: {ordering:?}: path {path:.1}, first cells {first:?}");
        if ordering == CellOrdering::Input {
            // Source rank, then input position: here the ids in ascending order.
            assert!(ids.windows(2).all(|w| w[0] < w[1]), "input order {ids:?}");
        }
        let mut sorted = ids;
        sorted.sort_unstable();
        owned_sets.push(sorted);
    }
    // ANCHOR_END: ordering
    assert!(
        owned_sets.windows(2).all(|w| w[0] == w[1]),
        "the cell ordering changed which cells rank {rank} owns"
    );
    if rank == 0 {
        println!("distributed mesh ok on {size} rank(s)");
    }
    Ok(())
}

// ANCHOR: chunk
/// This rank's share of the input: the unit square as N × N quadrilaterals. Rank r of p
/// passes the r-th of p even shares of the nodes and of the cells, each with its origin id.
/// The cells name their corners by node id, so a cell may use the nodes of any rank.
fn square<F: Real>(client: &Client, comm: &dyn CommDyn) -> MeshBuilder<F, D2, D2> {
    let (rank, size) = (comm.rank(), comm.size());
    let share = |n: usize| rank * n / size..(rank + 1) * n / size;

    // Node id = row · (N + 1) + column, at (column / N, row / N).
    let node_ids: Vec<i64> = share((N + 1) * (N + 1)).map(|id| id as i64).collect();
    let coords: Vec<f64> = node_ids
        .iter()
        .flat_map(|&id| {
            let (column, row) = (id as usize % (N + 1), id as usize / (N + 1));
            [column as f64 / N as f64, row as f64 / N as f64]
        })
        .collect();

    // Cell id = row · N + column; corners in tensor order: lower left, lower right, upper
    // left, upper right.
    let cell_ids: Vec<i64> = share(N * N).map(|id| id as i64).collect();
    let corners: Vec<i64> = cell_ids
        .iter()
        .flat_map(|&id| {
            let (column, row) = (id as usize % N, id as usize / N);
            let v = (row * (N + 1) + column) as i64;
            let up = (N + 1) as i64;
            [v, v + 1, v + up, v + up + 1]
        })
        .collect();

    let quad = CoordinateElement::<D2> {
        cell: Cell2::Quadrilateral,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, Some(&node_ids))
        .cell_block_by_ids(quad, &corners, Some(&cell_ids))
}
// ANCHOR_END: chunk

// ANCHOR: round_robin
/// A partitioner of its own: graph vertex (cell) i goes to rank i mod p. It ignores the
/// graph's edges and so cuts many of them: it shows the trait, not a good partition.
struct RoundRobin;

impl Partitioner for RoundRobin {
    fn name(&self) -> &str {
        "round-robin"
    }

    fn partition(
        &self,
        comm: &dyn CommDyn,
        g: &DistGraph,
        nparts: usize,
    ) -> tetraloma::Result<Vec<i32>> {
        // This rank holds the graph vertices vtxdist[rank]..vtxdist[rank + 1].
        let rank = comm.rank();
        let nparts = i64::try_from(nparts).map_err(|_| GridError::TooLarge)?;
        Ok((g.vtxdist[rank]..g.vtxdist[rank + 1])
            .map(|i| (i % nparts) as i32)
            .collect())
    }
}
// ANCHOR_END: round_robin

/// Checks that `mesh` covers the unit square once: the ranks own the N² cells and (N + 1)²
/// vertices once between them, and the area is 1. Returns this rank's owned and ghost cells.
fn check<F: Real, C: Comm>(mesh: &PlanarMesh<F, C>) -> tetraloma::Result<(usize, usize)> {
    let blocks = mesh.cell_blocks();
    let owned: usize = blocks.iter().map(|b| b.index_map.num_owned()).sum();
    let ghosts: usize = blocks.iter().map(|b| b.index_map.num_ghosts()).sum();
    let comm = mesh.comm();
    assert_eq!(
        comm.sum(owned as u64),
        (N * N) as u64,
        "the ranks do not own each cell once"
    );
    assert_eq!(
        comm.sum(mesh.vertex_index_map().num_owned() as u64),
        ((N + 1) * (N + 1)) as u64,
        "the ranks do not own each vertex once"
    );
    let area = mesh.measure(1)?; // owned cells only, summed over the ranks
    assert!(
        (area - 1.0).abs() <= 10.0 * crate::tol::<F>(),
        "area {area}, expected 1"
    );
    Ok((owned, ghosts))
}

/// The id (row · N + column) of every owned cell of `mesh`, in local order, found from the
/// cell's centre. It reads the physical points back to the host, for inspection only.
fn owned_cell_ids<F: Real, C: Comm>(mesh: &PlanarMesh<F, C>) -> tetraloma::Result<Vec<usize>> {
    let map = &mesh.cell_blocks()[0].index_map;
    if map.num_local() == 0 {
        return Ok(Vec::new());
    }
    let rule = QuadRule::for_grid(mesh, 1)?;
    let x = mesh.physical_points(&rule)?.blocks[0].to_host_cell_major()?;
    let per_cell = x.len() / map.num_local();
    let mut ids = Vec::new();
    for points in x.chunks_exact(per_cell).take(map.num_owned()) {
        // The mean of the cell's points is its centre.
        let mut centre = [0.0; 2];
        for p in points {
            centre[0] += p[0].widen() / points.len() as f64;
            centre[1] += p[1].widen() / points.len() as f64;
        }
        let column = (centre[0] * N as f64) as usize;
        let row = (centre[1] * N as f64) as usize;
        ids.push(row * N + column);
    }
    Ok(ids)
}
