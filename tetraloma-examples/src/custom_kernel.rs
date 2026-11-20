//! A downstream CubeCL kernel over a mesh (the book, tutorial "Writing kernels").
//!
//! The kernel integrates 1 and |x|² over the owned cells of a mesh. It reads only what the
//! public API exposes: the node coordinates and the origin of the `DeviceGrid`, the dofmap and
//! coordinate element of each `GeometryBlock`, the basis values of the quadrature rule, its
//! weights and the det J field. `integrate` launches it on any grid; `integrals` is generic
//! over every `Mesh<F, TD, GD, C>` and checks ∫ 1 against `measure`; `run` calls it through
//! `with_mesh!` on three meshes whose integrals are known.

use cubecl::prelude::{Client, CubeCount, CubeDim};
use tetraloma::detj_block;
use tetraloma::prelude::*;

use crate::build_mesh::{quarter_annulus, two_blocks};

/// Runs the program on `client` in `F`.
///
/// It integrates f = 1 and f = |x|² with [`integrate`] over three meshes: the rectangle
/// [0, 2] × [0, 1] as a quadrilateral and two triangles, the quarter annulus 1 ≤ r ≤ 2 as two
/// curved quadrilaterals of degree 4, and the box [0, 1] × [0, 1] × [0, 2] as two hexahedra.
/// It checks ∫ 1 against `measure` (within 10 times the rounding tolerance of `F`) and both
/// integrals against their exact values: 2 and 10/3, 3π/4 and 15π/8, 2 and 4. On the annulus
/// the cells only approximate the circles, which adds a relative error of about 2 · 10⁻⁶.
///
/// # Errors
///
/// The errors of mesh creation and of [`integrate`]; allocation and device errors.
///
/// # Panics
///
/// If a check fails.
pub fn run<F: Real>(client: &Client) -> tetraloma::Result<()> {
    println!(
        "integrals with a custom kernel on {} in {}",
        client.name(),
        crate::scalar_name::<F>()
    );
    let tol = 10.0 * crate::tol::<F>();
    let pi = std::f64::consts::PI;
    let cases = [
        Case {
            name: "rectangle [0, 2] × [0, 1]: one quadrilateral, two triangles",
            mesh: AnyMesh::Planar(two_blocks(client)?),
            degree: 2,
            exact: [2.0, 10.0 / 3.0],
            tol,
        },
        Case {
            name: "quarter annulus 1 ≤ r ≤ 2: two quadrilaterals of degree 4",
            mesh: AnyMesh::Planar(quarter_annulus(client, 4, 1, 2)?),
            degree: 16,
            exact: [3.0 * pi / 4.0, 15.0 * pi / 8.0],
            // The error of the geometry, about 2.4 · 10⁻⁶ for ∫ |x|².
            tol: tol.max(1e-5),
        },
        Case {
            name: "box [0, 1] × [0, 1] × [0, 2]: two hexahedra",
            mesh: AnyMesh::Volume(two_hexahedra(client)?),
            degree: 2,
            exact: [2.0, 4.0],
            tol,
        },
    ];
    for case in &cases {
        println!("  {}", case.name);
        // ANCHOR: with_mesh
        // The dimensions are known only at run time: with_mesh! has one arm per variant, and
        // each arm calls the generic function with a typed mesh.
        let [one, r2] = with_mesh!(case.mesh.as_ref(), m => integrals(m, case.degree))?;
        // ANCHOR_END: with_mesh
        let pairs = [("∫ 1", one, case.exact[0]), ("∫ |x|²", r2, case.exact[1])];
        for (what, value, exact) in pairs {
            let error = (value - exact).abs() / exact;
            println!("    {what} = {value:.12}, exact {exact:.12}, relative error {error:.1e}");
            assert!(error <= case.tol, "{what} = {value}, exact {exact}");
        }
    }
    Ok(())
}

/// One mesh of the program: its rule degree and the exact ∫ 1 and ∫ |x|².
struct Case<F: Real> {
    /// What the mesh is.
    name: &'static str,
    /// The mesh, with its dimensions erased as a file reader returns it.
    mesh: AnyMesh<F>,
    /// The degree of the Gauss rules.
    degree: u32,
    /// The exact ∫ 1 and ∫ |x|².
    exact: [f64; 2],
    /// The relative tolerance against `exact`.
    tol: f64,
}

// ANCHOR: generic
/// ∫ 1 and ∫ |x|² over `mesh` with [`integrate`], at Gauss rules of `degree`. Prints them and
/// checks ∫ 1 against `measure`. Generic over every mesh type, so `with_mesh!` can call it.
///
/// # Errors
///
/// The errors of [`integrate`] and `measure`.
///
/// # Panics
///
/// If ∫ 1 and `measure` differ by more than 10 times the rounding tolerance of `F`.
pub fn integrals<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    degree: u32,
) -> tetraloma::Result<[f64; 2]>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let [one, r2] = integrate(mesh, degree)?;
    let measure = mesh.measure(degree)?;
    let cells: usize = mesh
        .cell_blocks()
        .iter()
        .map(|block| block.index_map.num_owned())
        .sum();
    println!(
        "    {cells} cells of dimension {} in {}D: measure {measure:.12}",
        mesh.tdim(),
        mesh.gdim()
    );
    let tol = 10.0 * crate::tol::<F>();
    assert!(
        (one - measure).abs() <= tol * measure.abs(),
        "∫ 1 = {one}, measure {measure}"
    );
    Ok([one, r2])
}
// ANCHOR_END: generic

// ANCHOR: integrate
/// ∫ 1 and ∫ |x|² over the owned cells of `grid`, with Gauss rules of `degree`, summed over
/// the ranks of the grid's communicator. Collective on a distributed grid.
///
/// Per block it launches the kernel once per chunk of the det J field. Each unit writes the
/// integrals of its own cells, so no two units write the same entry and no atomics are needed.
/// The per-cell values are read back once per block and added in cell order in f64.
///
/// # Errors
///
/// `TooLarge` if a block has too many cells for one launch; allocation and device errors.
pub fn integrate<G: DeviceGrid>(grid: &G, degree: u32) -> tetraloma::Result<[f64; 2]> {
    let client = grid.client();
    let gdim = <G::GD as Dim>::N;
    let rule = QuadRule::<G::F, G::TD>::for_grid(grid, degree)?;
    let origin = grid.origin();
    let mut local = [0.0f64; 2];
    for (b, block) in grid.cell_blocks().iter().enumerate() {
        // Owned cells come first in a block; ghost cells belong to another rank's sum.
        let owned = block.index_map.num_owned();
        if owned == 0 {
            continue;
        }
        let geometry = grid.geometry_block(b);
        let points = rule.get(block.cell)?;
        // The basis values [q][k] and det J are computed on the device and stay there.
        let phi = points.tabulate(&geometry.element)?;
        let detj = detj_block(grid, b, points)?;
        // The weights [q], then the origin [g]: one small upload per block.
        let consts: Vec<G::F> = points
            .weights()
            .iter()
            .chain(origin.as_ref())
            .map(|&v| <G::F as Real>::narrow(v))
            .collect();
        let consts = DeviceBuffer::from_host(client, &consts)?;
        // The result: ∫ 1 and ∫ |x|² of every owned cell, [cell][2].
        let sums = DeviceBuffer::<G::F>::zeros(client, 2 * owned)?;
        for chunk in detj.chunks() {
            let cells = chunk.num_cells.min(owned.saturating_sub(chunk.first_cell));
            if cells == 0 {
                continue;
            }
            let (cube_count, cube_dim, cells_per_unit) = launch_shape(client, cells)?;
            let (first, num_cells) = (fit(chunk.first_cell)?, fit(cells)?);
            // SAFETY: cell c < num_cells of the chunk is cell first + c < owned of the block. It
            // reads its dofmap row (owned ≤ the block's rows), the nodes the row names (local
            // nodes), phi below num_points · nodes_per_cell, det J inside the chunk's whole
            // tiles, consts below num_points + gdim, and writes sums[2 (first + c) + 1], below
            // 2 · owned.
            unsafe {
                kernel::integrate_kernel::launch_unchecked::<G::F>(
                    client,
                    cube_count,
                    cube_dim,
                    grid.nodes().arg(),
                    geometry.dofmap.data.arg(),
                    phi.arg(),
                    chunk.data.arg(),
                    consts.arg(),
                    sums.arg(),
                    first,
                    num_cells,
                    cells_per_unit,
                    gdim,
                    geometry.nodes_per_cell(),
                    points.len(),
                    detj.tile(),
                );
            }
        }
        // One read of the per-cell integrals, added in cell order in f64: deterministic.
        for cell in sums.to_host()?.chunks_exact(2) {
            local[0] += cell[0].widen();
            local[1] += cell[1].widen();
        }
    }
    let comm = grid.comm();
    Ok([comm.sum(local[0]), comm.sum(local[1])])
}
// ANCHOR_END: integrate

// ANCHOR: launch_shape
/// A one-dimensional launch over `cells` cells: the cube count, the cube dimension and the
/// cells per unit. A cube has up to 64 units. On a GPU each unit integrates one cell. On the
/// CPU runtime, which reports its core count, every unit is a worker thread, and each one
/// integrates a contiguous run of cells.
///
/// # Errors
///
/// `TooLarge` if the cubes exceed the device's largest cube count.
fn launch_shape(client: &Client, cells: usize) -> tetraloma::Result<(CubeCount, CubeDim, u32)> {
    let hardware = &client.properties().hardware;
    let units = 64u32
        .min(hardware.max_units_per_cube)
        .min(hardware.max_cube_dim.0);
    let cells_per_unit = if hardware.num_cpu_cores.is_some() {
        cells.div_ceil(units as usize)
    } else {
        1
    };
    let cubes = fit(cells.div_ceil(cells_per_unit * units as usize))?;
    if cubes > hardware.max_cube_count.0 {
        return Err(GridError::TooLarge);
    }
    Ok((
        CubeCount::new_1d(cubes),
        CubeDim::new_1d(units),
        fit(cells_per_unit)?,
    ))
}
// ANCHOR_END: launch_shape

/// `x` as u32, or `TooLarge`.
fn fit(x: usize) -> tetraloma::Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

// ANCHOR: kernel
/// The kernel, in a module of its own so that CubeCL's prelude and tetraloma's do not meet.
mod kernel {
    use cubecl::prelude::*;

    /// ∫ 1 and ∫ |x|² over the cells `first + c`, `c < num_cells`, of one cell block:
    ///
    /// - `nodes`: the node coordinates `[node][gdim]`, relative to the mesh's origin;
    /// - `dofmap`: the node indices `[cell][k]` of the block;
    /// - `phi`: the basis values `[q][k]` at the rule's points;
    /// - `detj`: one chunk of the det J field, in tiles of `tile` cells; cell `c` of the chunk
    ///   is cell `first + c` of the block;
    /// - `consts`: the rule's weights `[q]`, then the origin `[g]`;
    /// - `sums`: `[cell][2]`, the two integrals of every cell.
    ///
    /// Each unit integrates `cells_per_unit` consecutive cells.
    #[cube(launch_unchecked)]
    #[allow(clippy::too_many_arguments, clippy::needless_range_loop)]
    pub fn integrate_kernel<F: Float>(
        nodes: &[F],
        dofmap: &[u32],
        phi: &[F],
        detj: &[F],
        consts: &[F],
        sums: &mut [F],
        first: u32,
        num_cells: u32,
        cells_per_unit: u32,
        #[comptime] gdim: usize,
        #[comptime] nodes_per_cell: usize,
        #[comptime] num_points: usize,
        #[comptime] tile: usize,
    ) {
        let begin = ABSOLUTE_POS * cells_per_unit as usize;
        for i in 0..cells_per_unit as usize {
            let c = begin + i;
            if c < num_cells as usize {
                let cell = first as usize + c;
                // Point q of chunk cell c: tile c / tile, lane c % tile.
                let t = c / tile;
                let base = t * num_points * tile + (c - t * tile);
                let mut one = F::from_int(0);
                let mut r2 = F::from_int(0);
                for q in 0..num_points {
                    let w = consts[q] * detj[base + q * tile];
                    // |X(ξ_q)|², with X_g = origin_g + Σ_k φ_k(ξ_q) x_{k,g}.
                    let mut xx = F::from_int(0);
                    for g in 0..gdim {
                        let mut x = consts[g + num_points];
                        for k in 0..nodes_per_cell {
                            let node = dofmap[cell * nodes_per_cell + k] as usize;
                            x += phi[q * nodes_per_cell + k] * nodes[node * gdim + g];
                        }
                        xx += x * x;
                    }
                    one += w;
                    r2 += w * xx;
                }
                sums[cell * 2] = one;
                sums[cell * 2 + 1] = r2;
            }
        }
    }
}
// ANCHOR_END: kernel

/// The box [0, 1] × [0, 1] × [0, 2] as two unit cubes, stacked in z.
///
/// # Errors
///
/// The errors of `MeshBuilder::build`.
fn two_hexahedra<F: Real>(client: &Client) -> tetraloma::Result<VolumeMesh<F>> {
    // Twelve nodes: the four corners of the unit square at z = 0, 1 and 2.
    #[rustfmt::skip]
    let coords: [f64; 36] = [
        0.0, 0.0, 0.0,   1.0, 0.0, 0.0,   0.0, 1.0, 0.0,   1.0, 1.0, 0.0,
        0.0, 0.0, 1.0,   1.0, 0.0, 1.0,   0.0, 1.0, 1.0,   1.0, 1.0, 1.0,
        0.0, 0.0, 2.0,   1.0, 0.0, 2.0,   0.0, 1.0, 2.0,   1.0, 1.0, 2.0,
    ];
    // Tensor order of the reference cube: x fastest, then y, then z.
    #[rustfmt::skip]
    let cells: [u32; 16] = [
        0, 1, 2, 3, 4, 5, 6, 7,
        4, 5, 6, 7, 8, 9, 10, 11,
    ];
    let hexahedron = CoordinateElement::<D3> {
        cell: Cell3::Hexahedron,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(hexahedron, &cells, None)
        .build()
}
