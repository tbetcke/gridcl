//! `forward` and `reverse` on a mesh's node index map (Listing B13).

use cubecl::prelude::Client;
use tetraloma::prelude::*;

/// Runs the halo exchange of a distributed mesh on `comm`: one value per node and component
/// (block 3), owner to ghosts and back with `Add`.
///
/// The mesh is the unit sphere as 320 triangles of degree 2, in even chunks of the ranks, with
/// ghost cells across shared edges, in f64 where the device supports it and f32 otherwise.
/// `run` checks that `forward` gives every ghost node its owner's coordinates, and that
/// `reverse` with `Add` of ones adds one to each owned value per ghost copy, over all ranks.
/// Collective over `comm`; with one rank there are no ghosts, and both exchanges leave the
/// values unchanged.
///
/// # Errors
///
/// The errors of mesh creation and of the scatterer; allocation, device and communication
/// errors.
///
/// # Panics
///
/// If a check fails.
pub fn run<C: Comm>(client: &Client, comm: C) -> tetraloma::Result<()> {
    if crate::has_f64(client) {
        exchange::<f64, C>(client, comm)
    } else {
        exchange::<f32, C>(client, comm)
    }
}

fn exchange<F: Real + ScatterNumeric, C: Comm>(client: &Client, comm: C) -> tetraloma::Result<()> {
    let (rank, size) = (comm.rank(), comm.size());
    let scalp: SurfaceMesh<F, C> = tetraloma_testing::sphere(2, 2)
        .chunk(rank, size)
        .builder::<F, D2, D3>(client)?
        .partitioner(Block)
        .ghosts(GhostMode::SharedFacet)
        .build_on(comm)?;
    // ANCHOR: index_map
    let nodes = scalp.node_index_map();
    let (owned, local) = (nodes.num_owned(), nodes.num_local());
    // ANCHOR_END: index_map
    println!(
        "rank {rank} of {size}: {owned} owned and {} ghost nodes, on {} in {}",
        nodes.num_ghosts(),
        client.name(),
        crate::scalar_name::<F>()
    );

    // ANCHOR: forward
    // Owner -> ghosts: the owned nodes' coordinates, zero on the ghosts.
    let coordinates = scalp.nodes().to_host()?; // [node][3], relative to the mesh's origin
    let mut values = coordinates.clone();
    values[3 * owned..].fill(F::narrow(0.0));
    let mut u = DeviceBuffer::from_host(scalp.client(), &values)?;
    let sc = nodes.scatterer(scalp.client(), 3)?; // block size 3: a vector per node
    sc.forward(&mut u)?; // owner -> ghosts (insert)
    // ANCHOR_END: forward
    let tol = 10.0 * crate::tol::<F>();
    for (i, (got, want)) in u.to_host()?.iter().zip(&coordinates).enumerate() {
        assert!(
            (got.widen() - want.widen()).abs() <= tol,
            "rank {rank}: forward gave node {} component {} the value {got:?}, not {want:?}",
            i / 3,
            i % 3
        );
    }

    // ANCHOR: reverse
    // Ghosts -> owner: ones everywhere, summed. Each owned value becomes 1 plus the number of
    // ranks that ghost its node.
    let mut u = DeviceBuffer::from_host(scalp.client(), &vec![F::narrow(1.0); 3 * local])?;
    sc.reverse(&mut u, ReverseOp::Add)?; // ghosts -> owner, deterministic, no atomics
    // ANCHOR_END: reverse
    let summed = u.to_host()?;
    let owned_sum: f64 = summed[..3 * owned].iter().map(|v| v.widen()).sum();
    assert!(
        summed[3 * owned..].iter().all(|v| v.widen() == 1.0),
        "rank {rank}: reverse changed a ghost value"
    );
    let comm = scalp.comm();
    let total = comm.sum(owned_sum.round() as u64);
    let nodes_global = u64::try_from(nodes.size_global()).expect("a node count fits u64");
    let ghosts_global = comm.sum(nodes.num_ghosts() as u64);
    assert_eq!(
        total,
        3 * (nodes_global + ghosts_global),
        "reverse with Add: the owned sums do not count every ghost copy"
    );
    if rank == 0 {
        println!(
            "forward and reverse ok: {nodes_global} nodes, {ghosts_global} ghost copies on {size} rank(s)"
        );
    }
    Ok(())
}
