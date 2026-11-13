//! Shared helpers of the M3 acceptance tests: test meshes and their closed-form entity counts,
//! host f64 references, precision and vector-size helpers, and the orientation-fixture reader.
#![allow(dead_code)]

pub mod gmsh;
pub mod reference;
pub mod vtk;

use cubecl::prelude::*;
use gridcl::prelude::*;
use gridcl::{FieldShape, VECTOR_PAD};
use gridcl_device::testing::{record_plans, with_vector_size};
use gridcl_device::{KernelKind, LaunchPlan, VectorSizeOverride};
use gridcl_refcell::{LagrangeElement, NodeFamily as RefFamily};
use gridcl_testing::{HostMesh, TOL_F32, TOL_F64};

/// The test device.
pub fn client() -> Client {
    gridcl_testing::test_client()
}

/// Runs `$f::<f32>(&client)` and, where the device supports f64, `$f::<f64>(&client)`.
#[macro_export]
macro_rules! for_each_precision {
    ($client:expr, $f:ident) => {{
        let client = $client;
        $f::<f32>(&client);
        if gridcl_testing::supports::<f64>(&client) {
            $f::<f64>(&client);
        } else {
            eprintln!(
                "{}: f64 is not supported on this device; skipped",
                stringify!($f)
            );
        }
    }};
}

/// The relative tolerance of `F`: 1e-5 for f32, 1e-12 for f64.
pub fn tol<F: Real>() -> f64 {
    if size_of::<F>() == 4 {
        TOL_F32
    } else {
        TOL_F64
    }
}

/// The name of `F` for messages.
pub fn name<F: Real>() -> &'static str {
    if size_of::<F>() == 4 { "f32" } else { "f64" }
}

/// SplitMix64, as in gridcl-device's tests.
#[derive(Clone, Debug)]
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------------------------
// Test meshes and closed-form counts.

/// The structured boxes of "Testing and validation".
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BoxKind {
    Interval,
    Triangle,
    Quadrilateral,
    Tetrahedron,
    Hexahedron,
    Prism,
    Pyramid,
    Mixed,
}

impl BoxKind {
    pub const ALL: [BoxKind; 8] = [
        BoxKind::Interval,
        BoxKind::Triangle,
        BoxKind::Quadrilateral,
        BoxKind::Tetrahedron,
        BoxKind::Hexahedron,
        BoxKind::Prism,
        BoxKind::Pyramid,
        BoxKind::Mixed,
    ];

    pub fn tdim(self) -> usize {
        match self {
            BoxKind::Interval => 1,
            BoxKind::Triangle | BoxKind::Quadrilateral => 2,
            _ => 3,
        }
    }

    /// The box with n cells (or hexahedral subcells) per direction, of degree `p`.
    pub fn host(self, n: usize, p: u32) -> HostMesh {
        use gridcl_testing::*;
        match self {
            BoxKind::Interval => interval_box(n, p),
            BoxKind::Triangle => square_box(Cell2::Triangle, n, p),
            BoxKind::Quadrilateral => square_box(Cell2::Quadrilateral, n, p),
            BoxKind::Tetrahedron => cube_box(Cell3::Tetrahedron, n, p),
            BoxKind::Hexahedron => cube_box(Cell3::Hexahedron, n, p),
            BoxKind::Prism => cube_box(Cell3::Prism, n, p),
            BoxKind::Pyramid => cube_box(Cell3::Pyramid, n, p),
            BoxKind::Mixed => mixed_box(n, p),
        }
    }
}

/// Entity counts of a box: entities per dimension 0..=tdim, and per type for each dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Counts {
    /// Entities of dimension 0, 1, ..., tdim (cells last).
    pub per_dim: Vec<usize>,
    /// (dimension, entity type, count) for every entity type of dimensions 1..tdim.
    pub per_type: Vec<(usize, CellType, usize)>,
}

/// The closed-form counts of `kind` with n subcells per direction.
pub fn box_counts(kind: BoxKind, n: usize) -> Counts {
    use CellType::*;
    let v1 = n + 1;
    // Lattice counts of the hexahedral grid.
    let hex_v = v1 * v1 * v1;
    let hex_e = 3 * n * v1 * v1;
    let hex_f = 3 * n * n * v1;
    let n3 = n * n * n;
    match kind {
        BoxKind::Interval => Counts {
            per_dim: vec![v1, n],
            per_type: vec![],
        },
        BoxKind::Quadrilateral => {
            let e = 2 * n * v1;
            Counts {
                per_dim: vec![v1 * v1, e, n * n],
                per_type: vec![(1, Interval, e)],
            }
        }
        BoxKind::Triangle => {
            let e = 2 * n * v1 + n * n;
            Counts {
                per_dim: vec![v1 * v1, e, 2 * n * n],
                per_type: vec![(1, Interval, e)],
            }
        }
        BoxKind::Hexahedron => Counts {
            per_dim: vec![hex_v, hex_e, hex_f, n3],
            per_type: vec![(1, Interval, hex_e), (2, Quadrilateral, hex_f)],
        },
        BoxKind::Tetrahedron => {
            // Every square face gets a diagonal, every hexahedron its main diagonal; every
            // square face splits into two triangles and every hexahedron has six inner faces.
            let e = hex_e + hex_f + n3;
            let f = 2 * hex_f + 6 * n3;
            Counts {
                per_dim: vec![hex_v, e, f, 6 * n3],
                per_type: vec![(1, Interval, e), (2, Triangle, f)],
            }
        }
        BoxKind::Prism => {
            // z-faces split into two triangles; one diagonal quadrilateral per hexahedron.
            let z_faces = n * n * v1;
            let e = hex_e + z_faces;
            let tri = 2 * z_faces;
            let quad = 2 * n * n * v1 + n3;
            Counts {
                per_dim: vec![hex_v, e, tri + quad, 2 * n3],
                per_type: vec![
                    (1, Interval, e),
                    (2, Triangle, tri),
                    (2, Quadrilateral, quad),
                ],
            }
        }
        BoxKind::Pyramid => {
            // A centre per hexahedron, joined to its 8 corners; 12 inner triangles each.
            let e = hex_e + 8 * n3;
            let tri = 12 * n3;
            Counts {
                per_dim: vec![hex_v + n3, e, hex_f + tri, 6 * n3],
                per_type: vec![
                    (1, Interval, e),
                    (2, Triangle, tri),
                    (2, Quadrilateral, hex_f),
                ],
            }
        }
        BoxKind::Mixed => {
            // Split subcells (even index sum) and those of them on the face x = 0.
            let split = n3.div_ceil(2);
            let at_x0 = (n * n).div_ceil(2);
            let e = hex_e + 8 * split + at_x0;
            let tri = 12 * split + 3 * at_x0;
            let quad = hex_f - at_x0;
            let cells = n3 + 5 * split + at_x0;
            Counts {
                per_dim: vec![hex_v + split, e, tri + quad, cells],
                per_type: vec![
                    (1, Interval, e),
                    (2, Triangle, tri),
                    (2, Quadrilateral, quad),
                ],
            }
        }
    }
}

/// Builds `h` on `client`, panicking with the error.
pub fn build<F, TD, GD>(h: &HostMesh, client: &Client) -> Mesh<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    h.build::<F, TD, GD>(client)
        .unwrap_or_else(|e| panic!("build {}: {e}", name::<F>()))
}

/// `h` with its gdim-1 curve lifted into `gdim` dimensions along a smooth, non-degenerate
/// curve: x -> (x, 0.2 sin(pi x)) or (x, 0.2 sin(pi x), 0.1 cos(pi x)).
pub fn lift_curve(h: &HostMesh, gdim: usize) -> HostMesh {
    assert_eq!(h.gdim, 1);
    let pi = std::f64::consts::PI;
    let mut nodes = Vec::with_capacity(h.num_nodes() * gdim);
    for &x in &h.nodes {
        nodes.push(x);
        nodes.push(0.2 * (pi * x).sin());
        if gdim == 3 {
            nodes.push(0.1 * (pi * x).cos());
        }
    }
    HostMesh {
        gdim,
        nodes,
        ..h.clone()
    }
}

// ---------------------------------------------------------------------------------------------
// Host f64 references.

/// The Lagrange element of a host block.
pub fn element(cell: CellType, degree: u32) -> LagrangeElement {
    LagrangeElement::new(cell, degree, RefFamily::Equispaced).expect("element")
}

/// The physical coordinates of the nodes of cell `c` of block `b`, [k][gdim].
pub fn cell_nodes(h: &HostMesh, b: usize, c: usize) -> Vec<f64> {
    let block = &h.blocks[b];
    let k = block.nodes_per_cell();
    block.cells[c * k..(c + 1) * k]
        .iter()
        .flat_map(|&n| h.nodes[n as usize * h.gdim..(n as usize + 1) * h.gdim].to_vec())
        .collect()
}

/// Host Jacobians of block `b` at `points` (reference coordinates, [q][tdim]):
/// [cell][q][d * gdim + g], column-major, in f64.
pub fn host_jacobians(h: &HostMesh, b: usize, points: &[f64]) -> Vec<f64> {
    let block = &h.blocks[b];
    let td = gridcl_refcell::tdim(block.cell);
    let gd = h.gdim;
    let e = element(block.cell, block.degree);
    let tab = e.tabulate(points, 1).expect("tabulate");
    let k = block.nodes_per_cell();
    let nq = tab.num_points;
    let mut out = Vec::with_capacity(block.num_cells() * nq * td * gd);
    for c in 0..block.num_cells() {
        let x = cell_nodes(h, b, c);
        for q in 0..nq {
            for d in 0..td {
                for g in 0..gd {
                    out.push((0..k).map(|i| x[i * gd + g] * tab.get(1 + d, q, i)).sum());
                }
            }
        }
    }
    out
}

/// The pseudo-determinant sqrt(det(JᵀJ)) of a column-major gdim × tdim matrix, in f64.
pub fn host_detj(j: &[f64], td: usize, gd: usize) -> f64 {
    let col = |d: usize| &j[d * gd..(d + 1) * gd];
    let dot = |a: &[f64], b: &[f64]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    match (td, gd) {
        (1, 1) => j[0],
        (2, 2) => j[0] * j[3] - j[2] * j[1],
        (3, 3) => {
            let m = |g: usize, d: usize| j[d * 3 + g];
            m(0, 0) * (m(1, 1) * m(2, 2) - m(1, 2) * m(2, 1))
                - m(0, 1) * (m(1, 0) * m(2, 2) - m(1, 2) * m(2, 0))
                + m(0, 2) * (m(1, 0) * m(2, 1) - m(1, 1) * m(2, 0))
        }
        (1, _) => dot(col(0), col(0)).sqrt(),
        (2, 3) => {
            let (a, b, c) = (
                dot(col(0), col(0)),
                dot(col(0), col(1)),
                dot(col(1), col(1)),
            );
            (a * c - b * b).sqrt()
        }
        _ => unreachable!(),
    }
}

/// max |got − want| / max |want| (or the absolute error if `want` is zero).
pub fn rel_max_err(got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "lengths");
    let scale = want.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    let err = got
        .iter()
        .zip(want)
        .fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
    if scale > 0.0 { err / scale } else { err }
}

// ---------------------------------------------------------------------------------------------
// Fields on the host.

/// A field read back in plain [cell][point][component] order, as a flat vector.
pub fn flat<F: Real, S: FieldShape>(field: &QuadField<F, S>) -> Vec<F> {
    let host = field.to_host_cell_major().expect("to_host_cell_major");
    assert_eq!(size_of::<S::Host<F>>(), S::LEN * size_of::<F>());
    // SAFETY: the host value of every gridcl shape is an array, possibly nested, of exactly
    // S::LEN values of F (checked by size above), so the vector's memory is host.len() · LEN
    // initialised values of F, aligned for F.
    unsafe { core::slice::from_raw_parts(host.as_ptr().cast::<F>(), host.len() * S::LEN) }.to_vec()
}

/// [`flat`] of every block, widened to f64.
pub fn flat_f64<F: Real, S: FieldShape>(fields: &BlockFields<F, S>) -> Vec<Vec<f64>> {
    fields
        .blocks
        .iter()
        .map(|f| flat(f).into_iter().map(Real::widen).collect())
        .collect()
}

/// Asserts `got` equals `want` bitwise up to the sign of zero on the CPU runtime, and within
/// the tolerance of `F` (relative to the largest |want|) elsewhere. A separately written
/// reference kernel may fuse multiply-adds differently, which can flip the sign of a zero
/// (ADR-0013); every nonzero value must match in every bit. Under a `GRIDCL_GEOM_STRATEGY`
/// that reorders the sums of the node-order references ([`environment_reorders_sums`]), the
/// CPU runtime is held to the tolerance too.
pub fn assert_matches<F: Real>(got: &[F], want: &[F], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if gridcl_testing::bitwise_device() && !environment_reorders_sums() {
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            let (a, b) = (a.widen(), b.widen());
            assert!(
                a.to_bits() == b.to_bits() || (a == 0.0 && b == 0.0),
                "{what}: entry {i}: {a} vs {b} (bitwise up to the sign of zero)"
            );
        }
    } else {
        let g: Vec<f64> = got.iter().map(|x| x.widen()).collect();
        let w: Vec<f64> = want.iter().map(|x| x.widen()).collect();
        let err = rel_max_err(&g, &w);
        assert!(err <= tol::<F>(), "{what}: relative error {err:e}");
    }
}

/// Asserts bitwise equality on every device.
pub fn assert_bits<F: Real>(got: &[F], want: &[F], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    for (i, (a, b)) in got.iter().zip(want).enumerate() {
        assert!(
            a.widen().to_bits() == b.widen().to_bits(),
            "{what}: entry {i}: {} vs {}",
            a.widen(),
            b.widen()
        );
    }
}

// ---------------------------------------------------------------------------------------------
// Vector sizes.

/// 1 plus every power of two from 2 to `max` no wider than the device's widest IO-optimized
/// size for `elem_bytes`.
pub fn vector_sizes(client: &Client, elem_bytes: usize, max: usize) -> Vec<usize> {
    let widest = client
        .io_optimized_vector_sizes(elem_bytes)
        .next()
        .unwrap_or(1);
    let mut sizes = vec![1];
    sizes.extend(
        [2, 4, 8, 16]
            .into_iter()
            .filter(|&n| n <= widest && n <= max),
    );
    sizes
}

/// Runs `f` at vector size `n`; unless `expect_launch` is false, asserts that the first plan of
/// `kind` ran at exactly `n`.
pub fn at_vector_size<R>(
    n: usize,
    kind: KernelKind,
    expect_launch: bool,
    f: impl FnOnce() -> R,
) -> R {
    at_vector_size_with_plans(n, kind, expect_launch, f).0
}

/// [`at_vector_size`], also returning every plan made inside `f`, in order.
pub fn at_vector_size_with_plans<R>(
    n: usize,
    kind: KernelKind,
    expect_launch: bool,
    f: impl FnOnce() -> R,
) -> (R, Vec<(KernelKind, LaunchPlan)>) {
    let (result, plans): (R, Vec<(KernelKind, LaunchPlan)>) =
        record_plans(|| with_vector_size(VectorSizeOverride::Fixed(n), f));
    let first = plans.iter().find(|(k, _)| *k == kind);
    if expect_launch {
        let (_, plan) = first.unwrap_or_else(|| panic!("no {kind:?} plan was made"));
        assert_eq!(
            plan.vector_size, n,
            "the {kind:?} kernel ran at vector size {}, not {n}",
            plan.vector_size
        );
    } else {
        assert!(first.is_none(), "an empty block made a {kind:?} plan");
    }
    (result, plans)
}

/// The cell counts of the invariance tests (docs/plans/M3.md, question 14), besides 10^6 + 3.
pub const INVARIANCE_CELLS: [usize; 5] = [0, 1, 15, 16, 17];

/// The large invariance case: 10^6 + 3 P1 triangles.
pub const LARGE_CELLS: usize = 1_000_003;

/// `VECTOR_PAD`, for test arithmetic.
pub const PAD: usize = VECTOR_PAD;

// ---------------------------------------------------------------------------------------------
// The orientation fixture of gridcl-refcell.

/// The six cells with D_TOP >= 2 and the bits of every permutation of their vertices, in the
/// lexicographic order of [`next_permutation`].
pub fn orientation_fixture() -> Vec<(CellType, usize, Vec<u32>)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../gridcl-refcell/tests/fixtures/orientation.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    let value: serde_json::Value = serde_json::from_str(&text).expect("parse fixture");
    let cells = value.as_object().expect("object");
    let mut out = Vec::new();
    for (name, f) in cells {
        let cell = [
            CellType::Triangle,
            CellType::Quadrilateral,
            CellType::Tetrahedron,
            CellType::Hexahedron,
            CellType::Prism,
            CellType::Pyramid,
        ]
        .into_iter()
        .find(|c| format!("{c:?}") == *name)
        .unwrap_or_else(|| panic!("unknown cell {name}"));
        let nv = f["num_vertices"].as_u64().unwrap() as usize;
        let bits = f["bits"]
            .as_array()
            .unwrap()
            .iter()
            .map(|b| u32::try_from(b.as_u64().unwrap()).unwrap())
            .collect();
        out.push((cell, nv, bits));
    }
    out
}

/// Advances `p` to the next permutation in lexicographic order; false after the last one.
/// A copy of gridcl-refcell/tests/fixtures.rs.
pub fn next_permutation(p: &mut [i64]) -> bool {
    let Some(i) = (1..p.len()).rev().find(|&i| p[i - 1] < p[i]) else {
        return false;
    };
    let j = (i..p.len()).rev().find(|&j| p[j] > p[i - 1]).unwrap();
    p.swap(i - 1, j);
    p[i..].reverse();
    true
}

/// Every permutation of 0..n in lexicographic order.
pub fn permutations(n: usize) -> Vec<Vec<i64>> {
    let mut p: Vec<i64> = (0..n as i64).collect();
    let mut out = vec![p.clone()];
    while next_permutation(&mut p) {
        out.push(p.clone());
    }
    out
}

// ---------------------------------------------------------------------------------------------
// M9h: strategies, degrees, degree-scaled tolerances and the high-order test meshes
// (docs/plans/M9h.md, T0b; ADR-0042, items 2, 3 and 8).

pub use gridcl_geometry::testing::{Strategy, StrategyFamily, StrategyOverride};

/// The environment variable of the degree policy's override (ADR-0042, item 3).
pub const GEOM_STRATEGY_VAR: &str = "GRIDCL_GEOM_STRATEGY";

/// Whether `GRIDCL_GEOM_STRATEGY` forces, for some family, a strategy that sums in another
/// order than the scalar node-order references: `sumfact` (one direction at a time), `shared`
/// (`sumfact`'s kernels on tensor rules, T9) or `cooperative` (trees over units). `direct` and `loops` sum in node order, and the policy
/// stays bitwise M9's kernels (ADR-0042, item 11), so the references' bitwise comparisons hold
/// under them (docs/plans/M9h.md, Deviations log, T7b).
pub fn environment_reorders_sums() -> bool {
    let Some(value) = std::env::var_os(GEOM_STRATEGY_VAR) else {
        return false;
    };
    value.to_string_lossy().split(',').any(|entry| {
        let name = entry.rsplit('=').next().unwrap_or_default().trim();
        [
            Strategy::SumFactorised,
            Strategy::Shared,
            Strategy::Cooperative,
        ]
        .iter()
        .any(|s| s.name() == name)
    })
}

/// One setting an M9h test runs under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StrategyRun {
    /// No scoped hook: `GRIDCL_GEOM_STRATEGY` as set for the whole run (the kernel-task check
    /// set of docs/plans/M9h.md).
    Environment,
    /// Every family overridden by the scoped hook.
    Hook(StrategyOverride),
}

impl std::fmt::Display for StrategyRun {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StrategyRun::Environment => write!(f, "{GEOM_STRATEGY_VAR} from the environment"),
            StrategyRun::Hook(StrategyOverride::Policy) => write!(f, "the degree policy"),
            StrategyRun::Hook(StrategyOverride::Fixed(s)) => write!(f, "strategy {}", s.name()),
        }
    }
}

/// The settings of [`for_each_strategy!`]: the environment's alone if `GRIDCL_GEOM_STRATEGY`
/// is set, else the policy and then every strategy forced.
pub fn strategy_runs() -> Vec<StrategyRun> {
    if std::env::var_os(GEOM_STRATEGY_VAR).is_some() {
        return vec![StrategyRun::Environment];
    }
    let mut runs = vec![StrategyRun::Hook(StrategyOverride::Policy)];
    runs.extend(
        Strategy::ALL
            .into_iter()
            .map(|s| StrategyRun::Hook(StrategyOverride::Fixed(s))),
    );
    runs
}

/// Runs `f` with every family's strategy overridden by `s`.
pub fn with_every_family<R>(s: StrategyOverride, f: impl FnOnce() -> R) -> R {
    use StrategyFamily::*;
    use gridcl_geometry::testing::with_strategy;
    with_strategy(EVector, s, || {
        with_strategy(Points, s, || {
            with_strategy(Jacobian, s, || {
                with_strategy(Normals, s, || {
                    with_strategy(FacetNormals, s, || with_strategy(PullBack, s, f))
                })
            })
        })
    })
}

/// Runs `f` under `run` and checks what ran (`record_strategies`): under a forced strategy,
/// that strategy or its fallback `Loops` (ADR-0042, item 3).
pub fn under_strategy<R>(run: StrategyRun, f: impl FnOnce() -> R) -> R {
    let (result, ran) = gridcl_geometry::testing::record_strategies(|| match run {
        StrategyRun::Environment => f(),
        StrategyRun::Hook(s) => with_every_family(s, f),
    });
    if let StrategyRun::Hook(StrategyOverride::Fixed(s)) = run {
        for (family, got) in &ran {
            assert!(
                *got == s || *got == Strategy::Loops,
                "under {run}: {family:?} ran {}",
                got.name()
            );
        }
    }
    result
}

/// Runs `$f(run)` under every [`StrategyRun`] of [`strategy_runs`], `$f` taking the run.
#[macro_export]
macro_rules! for_each_strategy {
    ($f:expr) => {{
        #[allow(unused_mut)] // FnMut or Fn.
        let mut f = $f;
        for run in $crate::common::strategy_runs() {
            $crate::common::under_strategy(run, || f(run));
        }
    }};
}

/// The highest geometry degree of the M9h tests: 8, and 4 in debug builds (p ≥ 5 is
/// release-only, docs/plans/M9h.md, T0b).
pub const fn max_degree() -> u32 {
    if cfg!(debug_assertions) { 4 } else { 8 }
}

/// The degrees of the M9h tests for meshes with or without pyramids: 1 to [`max_degree`], and
/// at most 3 with pyramids (ADR-0041).
pub fn degrees(has_pyramids: bool) -> std::ops::RangeInclusive<u32> {
    1..=if has_pyramids { 3 } else { max_degree() }
}

pub use gridcl_testing::reference::Amplification;

/// max(1, A / A_ref) of every block of `h` (with its node family) at `points` of each block's
/// cell, the largest over blocks (ADR-0042, item 2).
pub fn block_scale(
    h: &HostMesh,
    points: impl Fn(CellType) -> Vec<f64>,
    kind: Amplification,
) -> f64 {
    h.blocks
        .iter()
        .filter(|b| b.num_cells() > 0)
        .map(|b| {
            gridcl_testing::reference::tolerance_scale(
                b.cell,
                b.degree,
                b.family,
                &points(b.cell),
                kind,
            )
        })
        .fold(1.0, f64::max)
}

/// Asserts the relative error of `got` against `want` (per quantity over the block, ADR-0042,
/// item 2) is within the tolerance of `F` times `scale`.
pub fn assert_scaled<F: Real>(got: &[f64], want: &[f64], scale: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    let err = gridcl_testing::reference::rel_err(got, want);
    let tol = gridcl_testing::reference::tolerance::<F>(scale);
    assert!(
        err <= tol,
        "{what} ({}): relative error {err:e} > {tol:e} (scale {scale})",
        name::<F>()
    );
}

/// Asserts `got` equals `want` bitwise on the CPU runtime (up to the sign of zero) and within
/// the tolerance of `F` times `scale` elsewhere, both read back and widened to f64.
pub fn assert_invariant<F: Real>(got: &[f64], want: &[f64], scale: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: lengths");
    if gridcl_testing::bitwise_device() {
        for (i, (a, b)) in got.iter().zip(want).enumerate() {
            assert!(
                a.to_bits() == b.to_bits() || (*a == 0.0 && *b == 0.0),
                "{what} ({}): entry {i}: {a} vs {b} (bitwise up to the sign of zero)",
                name::<F>()
            );
        }
    } else {
        assert_scaled::<F>(got, want, scale, what);
    }
}

/// The node families of a mesh whose blocks all have the cell types `cells`: equispaced, and
/// GLL where every cell offers it (intervals, quadrilaterals, hexahedra).
pub fn node_families(cells: &[CellType]) -> Vec<NodeFamily> {
    let gll = cells.iter().all(|c| {
        matches!(
            c,
            CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron
        )
    });
    if gll {
        vec![NodeFamily::Equispaced, NodeFamily::Gll]
    } else {
        vec![NodeFamily::Equispaced]
    }
}

/// A check on the M9h test meshes; [`visit_high_order_meshes`] calls the methods that apply to
/// each mesh's dimensions.
pub trait MeshVisitor {
    /// X, J, det J, K and the pull-back, on every mesh.
    fn common<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim;
    /// Facet normals, in codimension 0.
    fn facets<F: Real, TD: TopoDim + Codim0<TD>>(&mut self, what: &str, h: &HostMesh);
    /// Normals, in codimension 1.
    fn normals<F, TD, GD>(&mut self, what: &str, h: &HostMesh)
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim;
}

/// Which M9h meshes [`visit_high_order_meshes`] gives.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshSet {
    /// Small meshes of every (TD, GD) pair and cell type, curved and perturbed, for the
    /// accuracy tests.
    Accuracy,
    /// Boxes with at least 17 cells of each cell type, a curve in 2D and 3D, and the mixed
    /// surface, for the invariance tests ([`INVARIANCE_CELLS`]).
    Invariance,
}

/// Calls `v` on every mesh of `set` at degree `p`, in each node family it offers ([`node_families`]),
/// with every (TD, GD) pair, skipping meshes with pyramids above p = 3.
pub fn visit_high_order_meshes<F: Real>(set: MeshSet, p: u32, v: &mut impl MeshVisitor) {
    use gridcl_testing::{ball, cube_box, interval_box, mixed_surface, sphere, square_box};
    let accuracy = set == MeshSet::Accuracy;
    let named = |what: &str, f: NodeFamily| format!("{what} p={p} {f:?}");
    let interval = interval_box(if accuracy { 5 } else { 17 }, p).perturbed(0.05);
    for f in node_families(&[CellType::Interval]) {
        let h = interval.with_node_family(f);
        v.common::<F, D1, D1>(&named("intervals", f), &h);
        v.facets::<F, D1>(&named("intervals", f), &h);
        let curve = lift_curve(&h, 2);
        v.common::<F, D1, D2>(&named("curve in 2D", f), &curve);
        v.normals::<F, D1, D2>(&named("curve in 2D", f), &curve);
        let curve = lift_curve(&h, 3);
        v.common::<F, D1, D3>(&named("curve in 3D", f), &curve);
    }
    for (cell, n) in [
        (Cell2::Triangle, 3),
        (Cell2::Quadrilateral, if accuracy { 3 } else { 5 }),
    ] {
        let base = square_box(cell, n, p).perturbed(0.05);
        for f in node_families(&[cell.into()]) {
            let h = base.with_node_family(f);
            let what = named(&format!("{cell:?} box"), f);
            v.common::<F, D2, D2>(&what, &h);
            v.facets::<F, D2>(&what, &h);
        }
    }
    let surfaces = if accuracy {
        vec![
            ("sphere", sphere(1, p)),
            ("mixed surface", mixed_surface(1, p)),
        ]
    } else {
        vec![("mixed surface", mixed_surface(1, p))]
    };
    for (what, h) in &surfaces {
        let what = named(what, NodeFamily::Equispaced);
        v.common::<F, D2, D3>(&what, h);
        v.normals::<F, D2, D3>(&what, h);
    }
    for cell in [
        Cell3::Tetrahedron,
        Cell3::Hexahedron,
        Cell3::Prism,
        Cell3::Pyramid,
    ] {
        if cell == Cell3::Pyramid && p > 3 {
            continue;
        }
        let n = if !accuracy && matches!(cell, Cell3::Hexahedron | Cell3::Prism) {
            3
        } else {
            2
        };
        let base = cube_box(cell, n, p).perturbed(0.05);
        for f in node_families(&[cell.into()]) {
            let h = base.with_node_family(f);
            let what = named(&format!("{cell:?} box"), f);
            v.common::<F, D3, D3>(&what, &h);
            v.facets::<F, D3>(&what, &h);
        }
    }
    if accuracy {
        let base = ball(0, p);
        for f in node_families(&[CellType::Hexahedron]) {
            let h = base.with_node_family(f);
            v.common::<F, D3, D3>(&named("ball", f), &h);
        }
    }
}
