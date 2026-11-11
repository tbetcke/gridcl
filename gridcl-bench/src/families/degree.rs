//! The degree sweep of M9h (docs/plans/M9h.md, T1; ADR-0042, items 1, 6, 7 and 12): every
//! degree-dependent kernel family at p = 1–8 (pyramids 1–3) on every cell type, in both node
//! families where offered, with the measurement protocol of "Benchmarks" (warm-up checked
//! against a host reference, calibration call, ten timed runs), at the vector size, tile and CPU
//! block of the vector policy (rule 12).
//!
//! - **Families** ([`DegreeFamily`]): the E-vector; X, J, det J and K; normals (on a curve in 2D
//!   and on surfaces in 3D); facet normals; the pull-back at 1 and 9 Newton iterations (rule 1:
//!   step = (t(9) − t(1)) / 8, set-up = t(1) − step); and the device basis evaluator on its own
//!   (`gridcl_geometry::bench::Interpolant`).
//! - **Sizes** count quadrature points: the Gauss rule of degree 2p (the facet rule of degree 2p
//!   for facet normals) on the box whose cell count is nearest to the size over the points per
//!   cell. Records hold the cells and points per cell, so time per point is exact.
//! - **Meshes:** the unit boxes of gridcl-testing (the codimension-1 meshes are the interval and
//!   square boxes lifted onto a line in 2D and a plane in 3D), in the node family of the
//!   record, unperturbed. Their cells are affine, so the pull-back converges in one step and is
//!   checked at 1 iteration too; the costs do not depend on the values, since every lane runs
//!   every iteration. The E-vector (except for the E-vector family itself) and the pull-back's
//!   input X are made outside the timing.
//! - **Checks:** each warm-up is compared on a sample of cells with an f64 host evaluation from
//!   the stored nodes (gridcl-refcell's tabulation), by ADR-0042, item 2: the relative error
//!   per quantity within 10^−5 (f32) or 10^−12 (f64) times max(1, A / A_ref). The J-derived
//!   quantities and the pull-back are held to further factors of max(1, n / n_ref) and of the
//!   box side, because coordinates of size 1 sum into J ~ 1 / side (ADR-0046's R);
//!   the basis evaluator to the absolute amplification of its coefficients. The E-vector is
//!   compared bitwise. Records keep the error and the tolerance (T3's error survey).
//! - **Compile time** (rule 6): the warm-up minus the median. Sizes run in ascending order with
//!   the same kernels, so the smallest size's records carry the first-use time; the warm-up
//!   launches exactly one new kernel (the family's), everything else is made beforehand. The
//!   mesh's E-vector is made only for the families that read it, after the E-vector family.
//! - **Status:** `ok`; `too slow to compile` where the warm-up exceeds the calibration call by
//!   more than 300 s; on a CPU `too slow` where one call takes over 600 s, measured or
//!   extrapolated linearly from the previous size; `too slow` and `too slow to compile` carry
//!   over to the larger sizes without running; `failed: <message>` for errors, failed checks
//!   and panics. A compile that never finishes cannot be timed in-process, so
//!   `scripts/bench/degree-sweep.sh` runs each (cell, degree) as one process under a time limit
//!   (and after a stop each of its configurations without a record) and records what a stopped
//!   process left as [`stopped_status`].
//! - **Launch-only** (rule 7): the warm-up, its check and one more call, at vector size 1 and at
//!   the policy's; no timed runs.
//!
//! Strategies: `policy` runs the degree policy; a strategy's name forces it on every family
//! (`gridcl_geometry::testing::with_strategy`), and each record names the strategy that ran
//! (`record_strategies`: the requested one or its fallback `loops`). A strategy that does not
//! run yet makes a `failed` record.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::rc::Rc;
use std::time::{Duration, Instant};

use cubecl::prelude::Client;
use gridcl::{
    Cell2, Cell3, Codim0, Codim1, D1, D2, D3, DeviceBuffer, DeviceGrid, Dim, EmbeddedIn,
    FacetQuadRule, FieldShape, Mesh, NodeFamily, PullBackOptions, QuadField, QuadRule, Real,
    TopoDim, VECTOR_PAD, detj_block, facet_normals_block, jacobians_block, normals_block,
    physical_points_block, pseudo_inverse_block, pull_back_block,
};
use gridcl_device::testing::{record_plans, with_kind_vector_size};
use gridcl_device::{KernelKind, VectorPolicy, VectorSizeOverride, check_precision};
use gridcl_dims::CellType;
use gridcl_geometry::bench::Interpolant;
use gridcl_geometry::testing::{
    Strategy, StrategyFamily, StrategyOverride, record_strategies, with_strategy,
};
use gridcl_refcell::{LagrangeElement, Tabulation, gauss_rule, tdim};
use gridcl_testing::HostMesh;
use gridcl_testing::reference::{
    Amplification, amplification, conditioned_scale, detj_of, node_ratio, pinv_of, rel_err,
    tolerance, tolerance_scale,
};

use super::Machine;
use super::mesh::nearest_side;
use crate::protocol::{calls_per_run, is_cpu, median, profile, sync};
use crate::record::{DegreeRecord, Scalar};

/// The protocol's sizes, in quadrature points (rule 12).
pub const DEGREE_SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
/// The size of a launch-only run, in quadrature points.
pub const LAUNCH_SIZE: usize = 10_000;
/// A call slower than this on a CPU is `too slow` (rule 12).
pub const MAX_CALL_S: f64 = 600.0;
/// A compile and load slower than this is `too slow to compile` (rule 12).
pub const MAX_COMPILE_S: f64 = 300.0;
/// The Newton iterations of the pull-back records (rule 1).
pub const PULL_BACK_ITERATIONS: [u32; 2] = [1, 9];
/// Cells a check compares: all up to this many, else about this many at an odd stride.
pub const CHECK_CELLS: usize = 64;

/// Every cell type, in the order of the records.
pub const CELLS: [CellType; 7] = [
    CellType::Interval,
    CellType::Triangle,
    CellType::Quadrilateral,
    CellType::Tetrahedron,
    CellType::Hexahedron,
    CellType::Prism,
    CellType::Pyramid,
];

/// The bench families of the degree sweep, in the order of `record::DEGREE_FAMILIES`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum DegreeFamily {
    EVector,
    Points,
    Jacobian,
    DetJ,
    Pinv,
    Normals,
    FacetNormals,
    PullBack,
    BasisEval,
}

impl DegreeFamily {
    pub const ALL: [DegreeFamily; 9] = [
        DegreeFamily::EVector,
        DegreeFamily::Points,
        DegreeFamily::Jacobian,
        DegreeFamily::DetJ,
        DegreeFamily::Pinv,
        DegreeFamily::Normals,
        DegreeFamily::FacetNormals,
        DegreeFamily::PullBack,
        DegreeFamily::BasisEval,
    ];

    /// The name in records: `EVector`, `Points`, ..., `BasisEval`.
    pub fn name(self) -> &'static str {
        match self {
            DegreeFamily::EVector => "EVector",
            DegreeFamily::Points => "Points",
            DegreeFamily::Jacobian => "Jacobian",
            DegreeFamily::DetJ => "DetJ",
            DegreeFamily::Pinv => "Pinv",
            DegreeFamily::Normals => "Normals",
            DegreeFamily::FacetNormals => "FacetNormals",
            DegreeFamily::PullBack => "PullBack",
            DegreeFamily::BasisEval => "BasisEval",
        }
    }

    /// The family named `name`, ignoring case.
    pub fn named(name: &str) -> Option<DegreeFamily> {
        DegreeFamily::ALL
            .into_iter()
            .find(|f| f.name().eq_ignore_ascii_case(name))
    }

    /// The kind of the family's plans.
    pub fn kind(self) -> KernelKind {
        match self {
            DegreeFamily::EVector => KernelKind::EVector,
            DegreeFamily::PullBack | DegreeFamily::BasisEval => KernelKind::PullBack,
            _ => KernelKind::GeomFactors,
        }
    }

    /// Whether the family is measured on `cell`: normals on intervals, triangles and
    /// quadrilaterals (one dimension up), the others on every cell type.
    pub fn applies_to(self, cell: CellType) -> bool {
        self != DegreeFamily::Normals || tdim(cell) < 3
    }

    /// gdim of the family's mesh on `cell`: tdim, one more for normals.
    pub fn gdim(self, cell: CellType) -> usize {
        tdim(cell) + usize::from(self == DegreeFamily::Normals)
    }

    /// The Newton iterations of its records: 1 and 9 for the pull-back.
    fn iterations(self) -> Vec<Option<u32>> {
        if self == DegreeFamily::PullBack {
            PULL_BACK_ITERATIONS.map(Some).to_vec()
        } else {
            vec![None]
        }
    }
}

/// The highest degree measured on `cell`: 8, and 3 on pyramids (ADR-0041).
pub fn max_degree(cell: CellType) -> u32 {
    if cell == CellType::Pyramid { 3 } else { 8 }
}

/// Whether `cell` offers `family`: GLL nodes on intervals, quadrilaterals and hexahedra.
pub fn offers(cell: CellType, family: NodeFamily) -> bool {
    family == NodeFamily::Equispaced
        || matches!(
            cell,
            CellType::Interval | CellType::Quadrilateral | CellType::Hexahedron
        )
}

/// The node family's name in records: `equispaced` or `gll`.
pub fn node_family_name(family: NodeFamily) -> &'static str {
    match family {
        NodeFamily::Equispaced => "equispaced",
        NodeFamily::Gll => "gll",
    }
}

/// The node family named `name` (`equispaced` or `gll`).
pub fn node_family_named(name: &str) -> Option<NodeFamily> {
    [NodeFamily::Equispaced, NodeFamily::Gll]
        .into_iter()
        .find(|&f| node_family_name(f) == name)
}

/// Cells of the box with `side` cells (or hexahedral subcells) per direction.
pub fn box_cells(cell: CellType, side: usize) -> usize {
    match cell {
        CellType::Interval => side,
        CellType::Triangle => 2 * side * side,
        CellType::Quadrilateral => side * side,
        CellType::Hexahedron => side * side * side,
        CellType::Prism => 2 * side * side * side,
        _ => 6 * side * side * side,
    }
}

/// Points per cell of the Gauss rule of degree 2p on `cell`.
pub fn points_per_cell(cell: CellType, degree: u32) -> usize {
    gauss_rule(cell, 2 * degree).points.len() / tdim(cell)
}

/// What the sweep runs.
#[derive(Clone, Debug)]
pub struct DegreeOptions {
    /// The machine: `macmini` (the local Mac) or a target of scripts/hpc/remote-test.sh.
    pub target: String,
    /// The job or run the records belong to (rule 11's paired runs share it).
    pub run_id: String,
    pub families: Vec<DegreeFamily>,
    pub cells: Vec<CellType>,
    pub degrees: Vec<u32>,
    pub node_families: Vec<NodeFamily>,
    pub scalars: Vec<Scalar>,
    /// Quadrature points, ascending.
    pub sizes: Vec<usize>,
    /// Strategy names: [`STRATEGIES`](crate::record::STRATEGIES) or `policy`.
    pub strategies: Vec<String>,
    pub launch_only: bool,
    /// The vector sizes of a launch-only run.
    pub launch_vector_sizes: LaunchVectorSizes,
    pub runs: usize,
    pub fallback_reason: Option<String>,
    /// Records of an earlier run of the sweep (`degree_sweep --resume`): their configurations
    /// are not run again, and their outcomes carry over to larger sizes as measured ones do.
    pub done: Vec<DegreeRecord>,
    /// `degree_sweep --stopped-after S`: run nothing, and record each configuration without a
    /// record in `done` as stopped by a process time limit of S seconds (see
    /// [`stopped_status`]).
    pub stopped_after: Option<f64>,
}

/// The status of a configuration whose process was stopped after `limit` seconds: `too slow to
/// compile` if its kernel has no record at a smaller size (it was still compiling, or never
/// started), else `too slow`; both with the limit, and carried over to larger sizes.
pub fn stopped_status(compiled: bool, limit: f64) -> String {
    let what = if compiled {
        "too slow"
    } else {
        "too slow to compile"
    };
    format!("{what} (process stopped after {limit:.0} s)")
}

/// The vector sizes a launch-only run launches each configuration at.
///
/// `scripts/bench/cuda-resources.sh` runs one at a time: cubecl-cuda names each dumped PTX file
/// by the tail of its kernel id, which kernels that differ only in comptime settings share
/// (docs/cubecl-upstream.md, C19).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LaunchVectorSizes {
    /// 1 and the policy's (once if the policy's is 1).
    #[default]
    Both,
    /// 1 only.
    Scalar,
    /// The policy's only.
    Policy,
}

/// Runs `f` with every family under the strategy `name` (`policy` or a strategy's name) and
/// returns the strategy that ran: the last one `record_strategies` saw, else (nothing
/// launched) the requested one.
fn under_strategy<R>(name: &str, f: impl FnOnce() -> R) -> (R, &'static str) {
    let forced = Strategy::ALL.into_iter().find(|s| s.name() == name);
    let s = forced.map_or(StrategyOverride::Policy, StrategyOverride::Fixed);
    let every = StrategyFamily::ALL
        .into_iter()
        .rev()
        .fold(Box::new(f) as Box<dyn FnOnce() -> R>, |f, family| {
            Box::new(move || with_strategy(family, s, f))
        });
    let (result, ran) = record_strategies(every);
    let ran = ran.last().map(|(_, s)| *s).or(forced);
    (result, ran.map_or("policy", Strategy::name))
}

/// Runs the sweep and passes each record to `on_record` as soon as it is written.
///
/// Configurations run in the order cell, degree, node family, scalar type, size, family,
/// strategy, iterations and vector size; one mesh per (cell, degree, node family, scalar,
/// size) serves every family that uses its point count. Failures are records, not errors.
///
/// # Errors
///
/// An unknown strategy name, or sizes not in ascending order.
pub fn degree_sweep(
    client: &Client,
    machine: &Machine,
    options: &DegreeOptions,
    on_record: &mut dyn FnMut(&DegreeRecord),
) -> Result<Vec<DegreeRecord>, String> {
    if let Some(s) = options
        .strategies
        .iter()
        .find(|s| s.as_str() != "policy" && !Strategy::ALL.iter().any(|t| t.name() == *s))
    {
        return Err(format!("unknown strategy {s}"));
    }
    if !options.sizes.is_sorted() {
        return Err("sizes must ascend (compile times land on the smallest)".to_owned());
    }
    let has_f64 = check_precision::<f64>(client).is_ok();
    let mut records = Vec::new();
    for &cell in &options.cells {
        for &degree in options.degrees.iter().filter(|&&p| p <= max_degree(cell)) {
            for &node_family in options.node_families.iter().filter(|&&f| offers(cell, f)) {
                let mut hosts = HostMeshes::new();
                for &scalar in &options.scalars {
                    if scalar == Scalar::F64 && !has_f64 {
                        eprintln!("degree_sweep: f64 skipped: the device has no f64");
                        continue;
                    }
                    let mut history = History::new();
                    for &nominal in &options.sizes {
                        let point = Point {
                            cell,
                            degree,
                            node_family,
                            scalar,
                            nominal,
                        };
                        let mut sweep = PointSweep {
                            client,
                            machine,
                            options,
                            point,
                            hosts: &mut hosts,
                            meshes: HashMap::new(),
                            history: &mut history,
                        };
                        let new = match scalar {
                            Scalar::F32 => sweep.run::<f32>(on_record),
                            Scalar::F64 => sweep.run::<f64>(on_record),
                            Scalar::U32 => return Err("the degree sweep has no u32".to_owned()),
                        };
                        records.extend(new);
                    }
                }
            }
        }
    }
    Ok(records)
}

/// One (cell, degree, node family, scalar, size) of the sweep.
#[derive(Clone, Copy, Debug)]
struct Point {
    cell: CellType,
    degree: u32,
    node_family: NodeFamily,
    scalar: Scalar,
    nominal: usize,
}

/// The previous size's outcome of a (family, strategy, iterations, vector size), for the
/// carried-over and extrapolated statuses.
type History = HashMap<(DegreeFamily, String, Option<u32>, usize), (usize, f64, String)>;

/// Host meshes of one (cell, degree, node family), by (lifted, side).
type HostMeshes = HashMap<(bool, usize), Rc<HostMesh>>;

struct PointSweep<'a> {
    client: &'a Client,
    machine: &'a Machine,
    options: &'a DegreeOptions,
    point: Point,
    hosts: &'a mut HostMeshes,
    /// Device meshes of this point (`MeshData`), by (lifted, side).
    meshes: HashMap<(bool, usize), Rc<dyn Any>>,
    history: &'a mut History,
}

impl PointSweep<'_> {
    fn run<F: Real>(&mut self, on_record: &mut dyn FnMut(&DegreeRecord)) -> Vec<DegreeRecord> {
        let p = self.point;
        let options = self.options;
        let mut out = Vec::new();
        for &family in options.families.iter().filter(|f| f.applies_to(p.cell)) {
            let kind = family.kind();
            let bytes = p.scalar.bytes();
            let policy = VectorPolicy::for_client(self.client);
            let policy_v = policy.vector_size(kind, bytes);
            let vector_sizes = match (options.launch_only, options.launch_vector_sizes) {
                (false, _) | (true, LaunchVectorSizes::Policy) => vec![policy_v],
                (true, LaunchVectorSizes::Both) if policy_v != 1 => vec![1, policy_v],
                (true, _) => vec![1],
            };
            for iterations in family.iterations() {
                // Built on the first configuration that runs, then shared.
                let mut work: Option<Result<Work, String>> = None;
                for strategy in &options.strategies {
                    for &v in &vector_sizes {
                        let key = (family, strategy.clone(), iterations, v);
                        if let Some(r) = self.done(family, strategy, iterations, v) {
                            let time = if r.status == "ok" {
                                r.median_s
                            } else {
                                r.calibration_s
                            };
                            self.history
                                .insert(key, (p.nominal, time, r.status.clone()));
                            continue;
                        }
                        let carried = self
                            .carried(&key, is_cpu(self.machine.class))
                            .or_else(|| self.stopped(&key));
                        let (outcome, ran, shape) = match carried {
                            Some(outcome) => (outcome, "direct", None),
                            None => {
                                let w =
                                    work.get_or_insert_with(|| self.work::<F>(family, iterations));
                                match w {
                                    Ok(w) => {
                                        let (outcome, ran) = under_strategy(strategy, || {
                                            with_kind_vector_size(
                                                kind,
                                                VectorSizeOverride::Fixed(v),
                                                || self.measure(kind, w),
                                            )
                                        });
                                        (outcome, ran, Some(w.shape))
                                    }
                                    Err(e) => (Outcome::failed(e.clone()), "direct", None),
                                }
                            }
                        };
                        let shape = shape.unwrap_or_else(|| self.shape_estimate());
                        let time = if outcome.status == "ok" {
                            outcome.median_s
                        } else {
                            outcome.calibration_s
                        };
                        self.history
                            .insert(key, (p.nominal, time, outcome.status.clone()));
                        let record = self.record(
                            family, strategy, ran, iterations, v, policy_v, shape, outcome,
                        );
                        on_record(&record);
                        out.push(record);
                    }
                }
            }
        }
        out
    }

    /// The outcome of a configuration that is not run: a smaller size was too slow or too slow
    /// to compile, or on a CPU its time extrapolates beyond [`MAX_CALL_S`].
    fn carried(
        &self,
        key: &(DegreeFamily, String, Option<u32>, usize),
        cpu: bool,
    ) -> Option<Outcome> {
        let (n, t, status) = self.history.get(key)?;
        let estimate = t * self.point.nominal as f64 / *n as f64;
        if status.starts_with("too slow") {
            let mut o = Outcome::failed(String::new());
            o.status = status.clone();
            o.calibration_s = estimate;
            Some(o)
        } else if cpu && status == "ok" && estimate > MAX_CALL_S {
            let mut o = Outcome::failed(String::new());
            o.status = "too slow".to_owned();
            o.calibration_s = estimate;
            Some(o)
        } else {
            None
        }
    }

    /// The record of `options.done` with this configuration, if any.
    fn done(
        &self,
        family: DegreeFamily,
        strategy: &str,
        iterations: Option<u32>,
        vector_size: usize,
    ) -> Option<&DegreeRecord> {
        let p = self.point;
        let cell = format!("{:?}", p.cell);
        self.options.done.iter().find(|r| {
            r.family == family.name()
                && r.cell == cell
                && r.gdim == family.gdim(p.cell)
                && r.degree == p.degree
                && r.node_family == node_family_name(p.node_family)
                && r.scalar == p.scalar
                && r.strategy == strategy
                && r.nominal_points == p.nominal
                && r.iterations == iterations
                && r.vector_size == vector_size
                && r.launch_only == self.options.launch_only
        })
    }

    /// The outcome of a configuration that is not run because its process was stopped
    /// (`options.stopped_after`); `None` in a normal run.
    fn stopped(&self, key: &(DegreeFamily, String, Option<u32>, usize)) -> Option<Outcome> {
        let limit = self.options.stopped_after?;
        let mut o = Outcome::failed(String::new());
        o.status = stopped_status(self.history.contains_key(key), limit);
        o.calibration_s = limit;
        Some(o)
    }

    /// The cells and points of a configuration that was not built.
    fn shape_estimate(&self) -> Shape {
        let p = self.point;
        let q = points_per_cell(p.cell, p.degree);
        Shape {
            cells: (p.nominal / q).max(1),
            points_per_cell: q,
        }
    }

    /// The warm-up with its check, then the calibration call and the timed runs (or, in a
    /// launch-only run, one more call), as an outcome; errors and panics become `failed`.
    fn measure(&self, kind: KernelKind, work: &mut Work) -> Outcome {
        let mut out = Outcome::failed(String::new());
        out.status = "ok".to_owned();
        let result = catch_unwind(AssertUnwindSafe(|| {
            measure_into(
                self.client,
                kind,
                self.options,
                is_cpu(self.machine.class),
                work,
                &mut out,
            )
        }));
        match result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => out.status = format!("failed: {e}"),
            Err(panic) => {
                let message = panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_else(|| "a panic".to_owned());
                out.status = format!("failed: panic: {message}");
            }
        }
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        family: DegreeFamily,
        strategy: &str,
        ran: &str,
        iterations: Option<u32>,
        vector_size: usize,
        policy_vector_size: usize,
        shape: Shape,
        outcome: Outcome,
    ) -> DegreeRecord {
        let p = self.point;
        let m = self.machine;
        let bytes = p.scalar.bytes();
        let policy = VectorPolicy::for_client(self.client);
        DegreeRecord {
            backend: m.backend.clone(),
            device: m.device.clone(),
            device_class: m.class,
            target: self.options.target.clone(),
            run_id: self.options.run_id.clone(),
            family: family.name().to_owned(),
            cell: format!("{:?}", p.cell),
            gdim: family.gdim(p.cell),
            degree: p.degree,
            node_family: node_family_name(p.node_family).to_owned(),
            scalar: p.scalar,
            strategy: strategy.to_owned(),
            strategy_ran: ran.to_owned(),
            nominal_points: p.nominal,
            cells: shape.cells,
            points_per_cell: shape.points_per_cell,
            iterations,
            vector_size,
            vector_size_ran: outcome.vector_size_ran,
            policy_vector_size,
            tile: policy.quad_tile(bytes),
            block_bytes: policy.cpu_block_bytes(family.kind(), bytes),
            units_per_group: outcome.units_per_group,
            status: outcome.status,
            launch_only: self.options.launch_only,
            check_error: outcome.check.map(|c| c.error),
            check_tolerance: outcome.check.map(|c| c.tolerance),
            times_s: outcome.times_s,
            median_s: outcome.median_s,
            calls_per_run: outcome.calls_per_run,
            calibration_s: outcome.calibration_s,
            compile_s: outcome.compile_s,
            kernel_cache: kernel_cache(m),
            commit: m.commit.clone(),
            profile: profile().to_owned(),
            cores: m.cores.clone(),
            fallback_reason: self.options.fallback_reason.clone(),
            node_load: None,
        }
    }

    // -----------------------------------------------------------------------------------------
    // Workloads.

    /// The work of `family` (at `iterations` for the pull-back) at this point.
    fn work<F: Real>(
        &mut self,
        family: DegreeFamily,
        iterations: Option<u32>,
    ) -> Result<Work, String> {
        let cell = self.point.cell;
        match (family, tdim(cell)) {
            (DegreeFamily::BasisEval, _) => basis_eval_work::<F>(self.client, self.point),
            (DegreeFamily::Normals, 1) => self.normals_work::<F, D1, D2>(),
            (DegreeFamily::Normals, _) => self.normals_work::<F, D2, D3>(),
            (DegreeFamily::FacetNormals, 1) => self.facet_work::<F, D1>(),
            (DegreeFamily::FacetNormals, 2) => self.facet_work::<F, D2>(),
            (DegreeFamily::FacetNormals, _) => self.facet_work::<F, D3>(),
            (_, 1) => self.codim0_work::<F, D1>(family, iterations),
            (_, 2) => self.codim0_work::<F, D2>(family, iterations),
            _ => self.codim0_work::<F, D3>(family, iterations),
        }
    }

    /// The box side whose cell count is nearest to the size over `q` points per cell.
    fn side(&self, q: usize) -> usize {
        let cell = self.point.cell;
        nearest_side((self.point.nominal / q).max(1), |s| box_cells(cell, s))
    }

    /// The host box of this point at `side`, lifted to one dimension more if `lifted`.
    fn host(&mut self, lifted: bool, side: usize) -> Rc<HostMesh> {
        let p = self.point;
        self.hosts
            .entry((lifted, side))
            .or_insert_with(|| {
                use gridcl_testing::{cube_box, interval_box, square_box};
                let h = match p.cell {
                    CellType::Interval => interval_box(side, p.degree),
                    CellType::Triangle => square_box(Cell2::Triangle, side, p.degree),
                    CellType::Quadrilateral => square_box(Cell2::Quadrilateral, side, p.degree),
                    CellType::Tetrahedron => cube_box(Cell3::Tetrahedron, side, p.degree),
                    CellType::Hexahedron => cube_box(Cell3::Hexahedron, side, p.degree),
                    CellType::Prism => cube_box(Cell3::Prism, side, p.degree),
                    _ => cube_box(Cell3::Pyramid, side, p.degree),
                };
                let h = h.with_node_family(p.node_family);
                let h = match (lifted, h.gdim) {
                    (false, _) => h,
                    // The line y = x / 2 and the plane z = (x + 2y) / 4: no entry of J is 0.
                    (true, 1) => h.mapped(2, |x| vec![x[0], 0.5 * x[0]]),
                    (true, _) => h.mapped(3, |x| vec![x[0], x[1], 0.25 * (x[0] + 2.0 * x[1])]),
                };
                Rc::new(h)
            })
            .clone()
    }

    /// The device mesh of this point at `side` with its host data, built once, and with its
    /// E-vector (made at vector size 1) if `evector`.
    ///
    /// The E-vector family asks without: its warm-up then launches the E-vector kernel first,
    /// so its records carry that kernel's compile time (`EVector` runs first in a point).
    fn mesh<F, TD, GD>(
        &mut self,
        lifted: bool,
        side: usize,
        evector: bool,
    ) -> Result<Rc<MeshData<F, TD, GD>>, String>
    where
        F: Real,
        TD: TopoDim + EmbeddedIn<GD>,
        GD: Dim,
    {
        let data = match self.meshes.get(&(lifted, side)) {
            Some(m) => m
                .clone()
                .downcast::<MeshData<F, TD, GD>>()
                .map_err(|_| "a cached mesh of other dimensions".to_owned())?,
            None => {
                let h = self.host(lifted, side);
                let mesh: Mesh<F, TD, GD> = h.build(self.client).map_err(error)?;
                let host = HostCells::new(&mesh)?;
                let data = Rc::new(MeshData { mesh, side, host });
                self.meshes.insert((lifted, side), data.clone());
                data
            }
        };
        if evector {
            // At vector size 1: a gather, so the same values at every size, and its kernel
            // compiles in seconds where the policy's size can take minutes at high degree
            // (docs/cubecl-upstream.md, C20).
            with_kind_vector_size(KernelKind::EVector, VectorSizeOverride::Fixed(1), || {
                data.mesh.evector(0)
            })
            .map_err(error)?;
        }
        Ok(data)
    }

    /// The E-vector, X, J, det J, K and the pull-back on the codimension-0 box.
    fn codim0_work<F: Real, TD: TopoDim + Codim0<TD>>(
        &mut self,
        family: DegreeFamily,
        iterations: Option<u32>,
    ) -> Result<Work, String> {
        let p = self.point;
        let q = points_per_cell(p.cell, p.degree);
        let data = self.mesh::<F, TD, TD>(false, self.side(q), family != DegreeFamily::EVector)?;
        let rule = Rc::new(QuadRule::for_grid(&data.mesh, 2 * p.degree).map_err(error)?);
        let cell = data.mesh.cell_blocks()[0].cell;
        let shape = Shape {
            cells: data.host.cells,
            points_per_cell: q,
        };
        let refs = Rc::new(References::new(p, rule.get(cell).map_err(error)?.points()));
        let side = data.side as f64;
        let (d, r) = (data.clone(), rule.clone());
        Ok(match family {
            DegreeFamily::EVector => {
                let fresh: Rc<RefCell<Option<Mesh<F, TD, TD>>>> = Rc::new(RefCell::new(None));
                let (f1, f2, f3) = (fresh.clone(), fresh.clone(), fresh);
                let (d1, d2) = (d.clone(), d);
                Work {
                    shape,
                    prepare: Box::new(move || {
                        *f1.borrow_mut() = None;
                        *f1.borrow_mut() = Some(d1.mesh.try_clone().map_err(error)?);
                        Ok(())
                    }),
                    run: Box::new(move || {
                        let fresh = f2.borrow();
                        fresh
                            .as_ref()
                            .ok_or("no fresh mesh")?
                            .evector(0)
                            .map_err(error)?;
                        Ok(Box::new(()))
                    }),
                    check: Box::new(move |_| {
                        let fresh = f3.borrow();
                        let mesh = fresh.as_ref().ok_or("no fresh mesh")?;
                        check_evector(&d2.host, mesh)
                    }),
                }
            }
            DegreeFamily::PullBack => {
                let pts = rule.get(cell).map_err(error)?;
                let x = Rc::new(physical_points_block(&d.mesh, 0, pts).map_err(error)?);
                let opts = PullBackOptions {
                    tolerance: None,
                    max_iterations: iterations.expect("pull-back iterations"),
                };
                let (d1, x1) = (d.clone(), x);
                let scale = refs.pull_back_scale(&d.host, &sample(d.host.cells));
                let want = refs.points.clone();
                Work {
                    shape,
                    prepare: no_prepare(),
                    run: Box::new(move || {
                        let out = pull_back_block(&d1.mesh, 0, &x1, &opts).map_err(error)?;
                        Ok(Box::new(out))
                    }),
                    check: Box::new(move |out| {
                        let (xi, _) = *out
                            .downcast::<(
                                QuadField<F, gridcl::VecShape<TD>>,
                                QuadField<F, gridcl::ScalarShape>,
                            )>()
                            .map_err(|_| "pull-back: unexpected output")?;
                        let cells = sample(d.host.cells);
                        let got = sampled(&xi, &cells)?;
                        let want: Vec<f64> =
                            cells.iter().flat_map(|_| want.iter().copied()).collect();
                        Ok(Check::new(
                            rel_err(&got, &want),
                            tolerance::<F>(scale) * side,
                        ))
                    }),
                }
            }
            _ => {
                let d1 = d.clone();
                Work {
                    shape,
                    prepare: no_prepare(),
                    run: Box::new(move || -> Result<Box<dyn Any>, String> {
                        let pts = r.get(cell).map_err(error)?;
                        let m = &d1.mesh;
                        Ok(match family {
                            DegreeFamily::Points => {
                                Box::new(physical_points_block(m, 0, pts).map_err(error)?)
                            }
                            DegreeFamily::Jacobian => {
                                Box::new(jacobians_block(m, 0, pts).map_err(error)?)
                            }
                            DegreeFamily::DetJ => Box::new(detj_block(m, 0, pts).map_err(error)?),
                            _ => Box::new(pseudo_inverse_block(m, 0, pts).map_err(error)?),
                        })
                    }),
                    check: Box::new(move |out| {
                        let cells = sample(d.host.cells);
                        let h = &d.host;
                        let (td, gd) = (TD::N, TD::N);
                        match family {
                            DegreeFamily::Points => {
                                let field = downcast::<QuadField<F, gridcl::VecShape<TD>>>(out)?;
                                let want = refs.points_at(h, &cells);
                                let tol = tolerance::<F>(refs.scale_values);
                                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
                            }
                            DegreeFamily::Jacobian => {
                                let field =
                                    downcast::<QuadField<F, gridcl::JacShape<TD, TD>>>(out)?;
                                let want = refs.jacobians_at(h, &cells);
                                let tol = tolerance::<F>(refs.scale_derivatives) * side;
                                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
                            }
                            DegreeFamily::DetJ => {
                                let field = downcast::<QuadField<F, gridcl::ScalarShape>>(out)?;
                                let want: Vec<f64> = refs
                                    .jacobians_at(h, &cells)
                                    .chunks_exact(td * gd)
                                    .map(|j| detj_of(j, td, gd))
                                    .collect();
                                let tol = tolerance::<F>(refs.scale_derivatives) * side;
                                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
                            }
                            _ => {
                                let field =
                                    downcast::<QuadField<F, gridcl::PinvShape<TD, TD>>>(out)?;
                                let want: Vec<f64> = refs
                                    .jacobians_at(h, &cells)
                                    .chunks_exact(td * gd)
                                    .flat_map(|j| pinv_of(j, td, gd))
                                    .collect();
                                let scale =
                                    refs.conditioned_scale(h, &cells, Amplification::Derivatives);
                                let tol = tolerance::<F>(scale) * side;
                                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
                            }
                        }
                    }),
                }
            }
        })
    }

    /// Normals on the lifted box.
    fn normals_work<F, TD, GD>(&mut self) -> Result<Work, String>
    where
        F: Real,
        TD: TopoDim + Codim1<GD>,
        GD: Dim,
    {
        let p = self.point;
        let q = points_per_cell(p.cell, p.degree);
        let data = self.mesh::<F, TD, GD>(true, self.side(q), true)?;
        let rule = Rc::new(QuadRule::for_grid(&data.mesh, 2 * p.degree).map_err(error)?);
        let cell = data.mesh.cell_blocks()[0].cell;
        let refs = References::new(p, rule.get(cell).map_err(error)?.points());
        let shape = Shape {
            cells: data.host.cells,
            points_per_cell: q,
        };
        let side = data.side as f64;
        let (d, r) = (data.clone(), rule);
        Ok(Work {
            shape,
            prepare: no_prepare(),
            run: Box::new(move || {
                let pts = r.get(cell).map_err(error)?;
                Ok(Box::new(normals_block(&d.mesh, 0, pts).map_err(error)?))
            }),
            check: Box::new(move |out| {
                let field = downcast::<QuadField<F, gridcl::VecShape<GD>>>(out)?;
                let cells = sample(data.host.cells);
                let (td, gd) = (TD::N, GD::N);
                let nq = refs.num_points();
                let want: Vec<f64> = refs
                    .jacobians_at(&data.host, &cells)
                    .chunks_exact(td * gd)
                    .enumerate()
                    .flat_map(|(i, j)| {
                        let flipped = data.host.flags[cells[i / nq]] & 1 == 1;
                        unit_normal(j, gd, flipped)
                    })
                    .collect();
                let tol = tolerance::<F>(refs.scale_derivatives) * side;
                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
            }),
        })
    }

    /// Facet normals on the codimension-0 box, at the facet rule of degree 2p.
    fn facet_work<F: Real, TD: TopoDim + Codim0<TD>>(&mut self) -> Result<Work, String> {
        let p = self.point;
        let td_cell = TD::Cell::try_from(p.cell).map_err(|_| "facet normals: cell dimension")?;
        let rule = Rc::new(
            FacetQuadRule::<F, TD>::gauss(self.client, &[td_cell], 2 * p.degree).map_err(error)?,
        );
        let fps = rule.get(td_cell).map_err(error)?;
        let q = fps.points().len();
        let refs = References::new(p, fps.points().points());
        let normals: Vec<f64> = (0..q)
            .flat_map(|i| fps.reference_normal(i).to_vec())
            .collect();
        let data = self.mesh::<F, TD, TD>(false, self.side(q), true)?;
        let shape = Shape {
            cells: data.host.cells,
            points_per_cell: q,
        };
        let side = data.side as f64;
        let (d, r) = (data.clone(), rule);
        Ok(Work {
            shape,
            prepare: no_prepare(),
            run: Box::new(move || {
                let fps = r.get(td_cell).map_err(error)?;
                Ok(Box::new(
                    facet_normals_block(&d.mesh, 0, fps).map_err(error)?,
                ))
            }),
            check: Box::new(move |out| {
                let field = downcast::<QuadField<F, gridcl::VecShape<TD>>>(out)?;
                let cells = sample(data.host.cells);
                let td = TD::N;
                let want: Vec<f64> = refs
                    .jacobians_at(&data.host, &cells)
                    .chunks_exact(td * td)
                    .enumerate()
                    .flat_map(|(i, j)| {
                        let k = pinv_of(j, td, td);
                        let n_ref = &normals[(i % q) * td..(i % q + 1) * td];
                        let m: Vec<f64> = (0..td)
                            .map(|g| (0..td).map(|t| k[g * td + t] * n_ref[t]).sum())
                            .collect();
                        let len = m.iter().map(|x| x * x).sum::<f64>().sqrt();
                        m.into_iter().map(move |x| x / len)
                    })
                    .collect();
                let tol = tolerance::<F>(refs.scale_derivatives) * side;
                Ok(Check::new(rel_err(&sampled(&field, &cells)?, &want), tol))
            }),
        })
    }
}

/// The device basis evaluator at the Gauss points of degree 2p, repeated for about size / q
/// cells: the interpolant of node values in [1, 2).
fn basis_eval_work<F: Real>(client: &Client, p: Point) -> Result<Work, String> {
    let td = tdim(p.cell);
    let rule = gauss_rule(p.cell, 2 * p.degree);
    let q = rule.points.len() / td;
    let cells = (p.nominal / q).max(1);
    let count = cells
        .checked_mul(q)
        .ok_or("basis evaluator: too many points")?;
    let stride = count.next_multiple_of(VECTOR_PAD);
    let mut rows = vec![F::narrow(0.0); td * stride];
    for i in 0..stride {
        // Padding points repeat the last point.
        let at = i.min(count - 1) % q;
        for d in 0..td {
            rows[d * stride + i] = F::narrow(rule.points[at * td + d]);
        }
    }
    let element =
        LagrangeElement::new(p.cell, p.degree, p.node_family).map_err(|e| e.to_string())?;
    let values: Vec<f64> = (0..element.dim())
        .map(|j| 1.0 + (j % 8) as f64 / 8.0)
        .collect();
    let f =
        Interpolant::<F>::new(client, p.cell, p.degree, p.node_family, &values).map_err(error)?;
    let rows = DeviceBuffer::from_host(client, &rows).map_err(error)?;
    let tab = element
        .tabulate(&rule.points, 1)
        .map_err(|e| e.to_string())?;
    // Rounding through the coefficients: about ε · Σ_k |φ_k| Σ_j |C'_kj| · max |w| for values,
    // and up to 2p² times more for derivatives (Markov).
    let a = amplification(
        p.cell,
        p.degree,
        p.node_family,
        &rule.points,
        Amplification::Coefficients,
    )
    .into_iter()
    .fold(1.0f64, f64::max);
    let tol_values = tolerance::<F>(2.0 * a);
    let tol_derivatives = tol_values * f64::from(2 * p.degree * p.degree);
    Ok(Work {
        shape: Shape {
            cells,
            points_per_cell: q,
        },
        prepare: no_prepare(),
        run: Box::new(move || Ok(Box::new(f.evaluate(&rows, stride).map_err(error)?))),
        check: Box::new(move |out| {
            let out = downcast::<DeviceBuffer<F>>(out)?.to_host().map_err(error)?;
            let cells = sample(cells);
            let mut got = vec![Vec::new(); 1 + td];
            let mut want = vec![Vec::new(); 1 + td];
            for &c in &cells {
                for i in 0..q {
                    for d in 0..=td {
                        got[d].push(out[d * stride + c * q + i].widen());
                        want[d].push(
                            (0..tab.num_basis)
                                .map(|j| tab.get(d, i, j) * values[j])
                                .sum(),
                        );
                    }
                }
            }
            let e_values = rel_err(&got[0], &want[0]);
            let e_derivatives = (1..=td)
                .map(|d| rel_err(&got[d], &want[d]))
                .fold(0.0f64, f64::max);
            // The larger of the two errors relative to its own tolerance, reported against the
            // value tolerance.
            let ratio = (e_values / tol_values).max(e_derivatives / tol_derivatives);
            Ok(Check::new(ratio * tol_values, tol_values))
        }),
    })
}

/// The cells and points of a configuration.
#[derive(Clone, Copy, Debug)]
struct Shape {
    cells: usize,
    points_per_cell: usize,
}

/// One family's work at one point: an untimed `prepare` before every call, the timed `run`
/// (its output lives until the sync), and the warm-up's `check`.
struct Work {
    shape: Shape,
    prepare: Box<dyn FnMut() -> Result<(), String>>,
    run: Box<dyn FnMut() -> Result<Box<dyn Any>, String>>,
    check: CheckFn,
}

/// A warm-up check: the run's output to its error and tolerance.
type CheckFn = Box<dyn FnMut(Box<dyn Any>) -> Result<Check, String>>;

fn no_prepare() -> Box<dyn FnMut() -> Result<(), String>> {
    Box::new(|| Ok(()))
}

/// A check's relative error and the tolerance it is held to.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Check {
    error: f64,
    tolerance: f64,
}

impl Check {
    fn new(error: f64, tolerance: f64) -> Check {
        Check { error, tolerance }
    }

    /// Within the tolerance (a NaN error is not).
    fn passed(self) -> bool {
        self.error <= self.tolerance
    }
}

/// What one configuration measured.
#[derive(Clone, Debug)]
struct Outcome {
    status: String,
    vector_size_ran: usize,
    units_per_group: u32,
    check: Option<Check>,
    times_s: Vec<f64>,
    median_s: f64,
    calls_per_run: usize,
    calibration_s: f64,
    compile_s: f64,
}

impl Outcome {
    fn failed(message: String) -> Outcome {
        Outcome {
            status: format!("failed: {message}"),
            vector_size_ran: 0,
            units_per_group: 0,
            check: None,
            times_s: Vec::new(),
            median_s: 0.0,
            calls_per_run: 0,
            calibration_s: 0.0,
            compile_s: 0.0,
        }
    }
}

/// The measurement of [`PointSweep::measure`], filling `out` as it goes.
fn measure_into(
    client: &Client,
    kind: KernelKind,
    options: &DegreeOptions,
    cpu: bool,
    work: &mut Work,
    out: &mut Outcome,
) -> Result<(), String> {
    (work.prepare)()?;
    sync(client)?;
    let start = Instant::now();
    let (result, plans) = record_plans(|| -> Result<Box<dyn Any>, String> {
        let o = (work.run)()?;
        sync(client)?;
        Ok(o)
    });
    let warmup = start.elapsed().as_secs_f64();
    let output = result?;
    let plans: Vec<_> = plans.into_iter().filter(|(k, _)| *k == kind).collect();
    if plans.is_empty() {
        return Err(format!("the warm-up made no {kind:?} plan"));
    }
    out.vector_size_ran = plans.iter().map(|(_, p)| p.vector_size).max().unwrap_or(1);
    out.units_per_group = plans
        .iter()
        .map(|(_, p)| p.units_per_group)
        .max()
        .unwrap_or(1);
    let check = (work.check)(output)?;
    out.check = Some(check);
    if !check.passed() {
        return Err(format!(
            "check: relative error {:e} > {:e}",
            check.error, check.tolerance
        ));
    }
    let calibration = timed_call(client, work)?.as_secs_f64();
    out.calibration_s = calibration;
    out.median_s = calibration;
    out.calls_per_run = 1;
    out.compile_s = (warmup - calibration).max(0.0);
    if options.launch_only {
        return Ok(());
    }
    if out.compile_s > MAX_COMPILE_S {
        out.status = "too slow to compile".to_owned();
        return Ok(());
    }
    if cpu && calibration > MAX_CALL_S {
        out.status = "too slow".to_owned();
        return Ok(());
    }
    let calls = calls_per_run(Duration::from_secs_f64(calibration));
    let mut times = Vec::with_capacity(options.runs);
    for _ in 0..options.runs {
        let mut run = Duration::ZERO;
        for _ in 0..calls {
            run += timed_call(client, work)?;
        }
        times.push(run.as_secs_f64() / calls as f64);
    }
    out.median_s = median(&times);
    out.compile_s = (warmup - out.median_s).max(0.0);
    out.calls_per_run = calls;
    out.times_s = times;
    Ok(())
}

/// One `run` after an untimed `prepare` and sync: the time of the `run` and its sync.
fn timed_call(client: &Client, work: &mut Work) -> Result<Duration, String> {
    (work.prepare)()?;
    sync(client)?;
    let start = Instant::now();
    let out = (work.run)()?;
    sync(client)?;
    let elapsed = start.elapsed();
    drop(out);
    Ok(elapsed)
}

/// `disabled` where `CUDA_CACHE_DISABLE=1` (rule 6), else CubeCL's kernel cache state.
fn kernel_cache(machine: &Machine) -> String {
    if std::env::var("CUDA_CACHE_DISABLE").is_ok_and(|v| v == "1") {
        "disabled".to_owned()
    } else {
        machine.kernel_cache.clone()
    }
}

fn error(e: gridcl::GridError) -> String {
    e.to_string()
}

fn downcast<T: 'static>(out: Box<dyn Any>) -> Result<T, String> {
    out.downcast::<T>()
        .map(|b| *b)
        .map_err(|_| "unexpected output".to_owned())
}

// ---------------------------------------------------------------------------------------------
// Host references.

/// A device mesh with its E-vector, its box side and the host data of its checks.
struct MeshData<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim> {
    mesh: Mesh<F, TD, GD>,
    side: usize,
    host: HostCells<F>,
}

/// The stored nodes, dofmap, origin and flags of block 0, read once for the checks.
struct HostCells<F> {
    nodes: Vec<F>,
    dofmap: Vec<u32>,
    /// Nodes per cell.
    k: usize,
    gd: usize,
    origin: Vec<f64>,
    flags: Vec<u32>,
    cells: usize,
}

impl<F: Real> HostCells<F> {
    fn new<TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(mesh: &Mesh<F, TD, GD>) -> Result<Self, String> {
        let gb = mesh.geometry_block(0);
        let block = &mesh.cell_blocks()[0];
        Ok(HostCells {
            nodes: mesh.nodes().to_host().map_err(error)?,
            dofmap: gb.dofmap.data.to_host().map_err(error)?,
            k: gb.dofmap.stride as usize,
            gd: GD::N,
            origin: mesh.origin().as_ref().to_vec(),
            flags: block.flags.to_host().map_err(error)?,
            cells: block.index_map.num_local(),
        })
    }

    /// Coordinate g of node i of cell c as stored (relative to the origin).
    fn stored(&self, c: usize, i: usize, g: usize) -> F {
        self.nodes[self.dofmap[c * self.k + i] as usize * self.gd + g]
    }

    /// The absolute coordinates of cell c's nodes, `[node][g]`.
    fn coords(&self, c: usize) -> Vec<f64> {
        (0..self.k)
            .flat_map(|i| (0..self.gd).map(move |g| (i, g)))
            .map(|(i, g)| self.stored(c, i, g).widen() + self.origin[g])
            .collect()
    }
}

/// The element's tabulation at a point set and the tolerance scales of ADR-0042, item 2.
struct References {
    point: Point,
    points: Vec<f64>,
    tab: Tabulation,
    scale_values: f64,
    scale_derivatives: f64,
}

impl References {
    fn new(p: Point, points: &[f64]) -> References {
        let (cell, degree, family) = (p.cell, p.degree, p.node_family);
        let element = LagrangeElement::new(cell, degree, family).expect("element");
        References {
            point: p,
            points: points.to_vec(),
            tab: element.tabulate(points, 1).expect("tabulate"),
            scale_values: tolerance_scale(cell, degree, family, points, Amplification::Values),
            // J and what derives from it also scale with the node count (ADR-0046); the box
            // side stands in for R where the checks use the scale.
            scale_derivatives: tolerance_scale(
                cell,
                degree,
                family,
                points,
                Amplification::Derivatives,
            ) * node_ratio(cell, degree),
        }
    }

    fn num_points(&self) -> usize {
        self.tab.num_points
    }

    /// X at the cells, `[cell][q][g]`.
    fn points_at<F: Real>(&self, h: &HostCells<F>, cells: &[usize]) -> Vec<f64> {
        let (gd, nq, k) = (h.gd, self.num_points(), h.k);
        let mut out = Vec::with_capacity(cells.len() * nq * gd);
        for &c in cells {
            let x = h.coords(c);
            for q in 0..nq {
                for g in 0..gd {
                    out.push((0..k).map(|i| x[i * gd + g] * self.tab.get(0, q, i)).sum());
                }
            }
        }
        out
    }

    /// J at the cells, `[cell][q][d · GD + g]`.
    fn jacobians_at<F: Real>(&self, h: &HostCells<F>, cells: &[usize]) -> Vec<f64> {
        let (gd, nq, k) = (h.gd, self.num_points(), h.k);
        let td = tdim(self.point.cell);
        let mut out = Vec::with_capacity(cells.len() * nq * td * gd);
        for &c in cells {
            let x = h.coords(c);
            for q in 0..nq {
                for d in 0..td {
                    for g in 0..gd {
                        out.push(
                            (0..k)
                                .map(|i| x[i * gd + g] * self.tab.get(1 + d, q, i))
                                .sum(),
                        );
                    }
                }
            }
        }
        out
    }

    /// The scale of a quantity that solves with J (K, the pull-back) on the sampled cells:
    /// `reference::conditioned_scale` with the factor `kind` and the cells' J (ADR-0044).
    fn conditioned_scale<F: Real>(
        &self,
        h: &HostCells<F>,
        cells: &[usize],
        kind: Amplification,
    ) -> f64 {
        let p = self.point;
        conditioned_scale(
            p.cell,
            p.degree,
            p.node_family,
            &self.points,
            kind,
            &self.jacobians_at(h, cells),
            h.gd,
        )
    }

    /// The pull-back's scale on the sampled cells (`reference::pull_back_scale`).
    fn pull_back_scale<F: Real>(&self, h: &HostCells<F>, cells: &[usize]) -> f64 {
        self.conditioned_scale(h, cells, Amplification::Coefficients)
    }
}

/// The unit normal of a codimension-1 J (column-major gd × (gd − 1)), flipped if `flipped`.
fn unit_normal(j: &[f64], gd: usize, flipped: bool) -> Vec<f64> {
    let v = match gd {
        2 => vec![j[1], -j[0]],
        _ => vec![
            j[1] * j[5] - j[2] * j[4],
            j[2] * j[3] - j[0] * j[5],
            j[0] * j[4] - j[1] * j[3],
        ],
    };
    let s = if flipped { -1.0 } else { 1.0 };
    let len = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter().map(|x| s * x / len).collect()
}

/// The E-vector of `fresh` against the stored nodes of the sampled cells, bitwise: error 0 or
/// the largest relative difference, tolerance 0.
fn check_evector<F, TD, GD>(h: &HostCells<F>, fresh: &Mesh<F, TD, GD>) -> Result<Check, String>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let xe = fresh.evector(0).map_err(error)?;
    let cells = sample(h.cells);
    let mut got = Vec::new();
    let mut want = Vec::new();
    for chunk in xe.chunks() {
        let range = chunk.first_cell..chunk.first_cell + chunk.num_cells;
        let here: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|c| range.contains(c))
            .collect();
        if here.is_empty() {
            continue;
        }
        let data = chunk.data.to_host().map_err(error)?;
        let stride = chunk.num_cells.next_multiple_of(VECTOR_PAD);
        for c in here {
            for i in 0..h.k {
                for g in 0..h.gd {
                    got.push(data[(i * h.gd + g) * stride + c - chunk.first_cell].widen());
                    want.push(h.stored(c, i, g).widen());
                }
            }
        }
    }
    if got.len() != cells.len() * h.k * h.gd {
        return Err(format!("evector: {} of {} values", got.len(), want.len()));
    }
    let exact = got
        .iter()
        .zip(&want)
        .all(|(a, b)| a.to_bits() == b.to_bits());
    let error = if exact {
        0.0
    } else {
        rel_err(&got, &want).max(f64::MIN_POSITIVE)
    };
    Ok(Check::new(error, 0.0))
}

/// The values of `field` at the ascending `cells`, `[cell][point][component]`, widened.
fn sampled<F: Real, S: FieldShape>(
    field: &QuadField<F, S>,
    cells: &[usize],
) -> Result<Vec<f64>, String> {
    let (tile, nq, len) = (field.tile(), field.num_points(), S::LEN);
    let mut out = Vec::with_capacity(cells.len() * nq * len);
    for chunk in field.chunks() {
        let range = chunk.first_cell..chunk.first_cell + chunk.num_cells;
        let here: Vec<usize> = cells
            .iter()
            .copied()
            .filter(|c| range.contains(c))
            .collect();
        if here.is_empty() {
            continue;
        }
        let data = chunk.data.to_host().map_err(error)?;
        for cell in here {
            let c = cell - chunk.first_cell;
            for q in 0..nq {
                for k in 0..len {
                    out.push(data[((c / tile) * nq * len + q * len + k) * tile + c % tile].widen());
                }
            }
        }
    }
    if out.len() != cells.len() * nq * len {
        return Err(format!(
            "field: {} values for {} cells of {nq} points",
            out.len(),
            cells.len()
        ));
    }
    Ok(out)
}

/// The cells a check compares: all up to [`CHECK_CELLS`], else every s-th for an odd s (so
/// the sample meets every lane position) and the last cell.
pub fn sample(cells: usize) -> Vec<usize> {
    if cells <= CHECK_CELLS {
        return (0..cells).collect();
    }
    let stride = (cells / CHECK_CELLS) | 1;
    let mut out: Vec<usize> = (0..cells).step_by(stride).collect();
    if out.last() != Some(&(cells - 1)) {
        out.push(cells - 1);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::record::DEGREE_FAMILIES;

    #[test]
    fn family_names_are_the_record_names() {
        let names: Vec<&str> = DegreeFamily::ALL.iter().map(|f| f.name()).collect();
        assert_eq!(names, DEGREE_FAMILIES);
        for f in DegreeFamily::ALL {
            assert_eq!(DegreeFamily::named(&f.name().to_lowercase()), Some(f));
        }
        assert_eq!(DegreeFamily::named("GeomFactors"), None);
    }

    #[test]
    fn points_cells_and_sides() {
        assert_eq!(points_per_cell(CellType::Hexahedron, 8), 729);
        assert_eq!(points_per_cell(CellType::Interval, 1), 2);
        assert_eq!(points_per_cell(CellType::Tetrahedron, 1), 8);
        assert_eq!(box_cells(CellType::Prism, 3), 54);
        assert_eq!(box_cells(CellType::Pyramid, 2), 48);
        let side = |cell, n| nearest_side(n, |s| box_cells(cell, s));
        // 10^7 points at hexahedron p = 8: 13 717 cells, nearest 24³ = 13 824.
        assert_eq!(side(CellType::Hexahedron, 10_000_000 / 729), 24);
        assert_eq!(side(CellType::Interval, 5_000_000), 5_000_000);
    }

    #[test]
    fn samples_meet_every_lane_and_the_last_cell() {
        assert_eq!(sample(5), [0, 1, 2, 3, 4]);
        let s = sample(100 * CHECK_CELLS + 7);
        assert_eq!(*s.last().unwrap(), 100 * CHECK_CELLS + 6);
        let lanes: std::collections::BTreeSet<usize> = s.iter().map(|c| c % 16).collect();
        assert_eq!(lanes.len(), 16);
        assert!(s.len() <= CHECK_CELLS + 2);
    }

    fn options(families: &[DegreeFamily], cells: &[CellType], launch_only: bool) -> DegreeOptions {
        DegreeOptions {
            target: "test".to_owned(),
            run_id: "test".to_owned(),
            families: families.to_vec(),
            cells: cells.to_vec(),
            degrees: vec![1, 2],
            node_families: vec![NodeFamily::Equispaced, NodeFamily::Gll],
            scalars: vec![Scalar::F32, Scalar::F64],
            sizes: vec![10_000],
            strategies: vec!["direct".to_owned()],
            launch_only,
            launch_vector_sizes: LaunchVectorSizes::Both,
            runs: 2,
            fallback_reason: None,
            done: Vec::new(),
            stopped_after: None,
        }
    }

    /// Every family on every cell type at p = 1–2 and 10^4 points, launch-only: each record is
    /// `ok`, passed its check, launched at vector size 1 and at the policy's, and round-trips
    /// through JSON; the counts follow the cells, node families and iterations.
    #[test]
    fn launch_only_smoke_at_p_1_and_2() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let o = options(&DegreeFamily::ALL, &CELLS, true);
        let mut seen = 0;
        let records = degree_sweep(&client, &machine, &o, &mut |_| seen += 1).unwrap();
        assert_eq!(seen, records.len());
        let types = if check_precision::<f64>(&client).is_ok() {
            2
        } else {
            1
        };
        for family in DegreeFamily::ALL {
            let policy =
                |bytes| VectorPolicy::for_client(&client).vector_size(family.kind(), bytes);
            let sizes = |bytes| if policy(bytes) == 1 { 1 } else { 2 };
            for cell in CELLS {
                let of: Vec<&DegreeRecord> = records
                    .iter()
                    .filter(|r| r.family == family.name() && r.cell == format!("{cell:?}"))
                    .collect();
                let nodes = if offers(cell, NodeFamily::Gll) { 2 } else { 1 };
                let per_type: usize = [4, 8][..types].iter().map(|&b| sizes(b)).sum();
                let want = if family.applies_to(cell) {
                    2 * nodes * per_type * family.iterations().len()
                } else {
                    0
                };
                assert_eq!(of.len(), want, "{family:?} {cell:?}");
                for r in of {
                    assert_eq!(r.status, "ok", "{r:?}");
                    assert!(r.launch_only && r.times_s.is_empty(), "{r:?}");
                    assert_eq!(
                        (r.strategy.as_str(), r.strategy_ran.as_str()),
                        ("direct", "direct")
                    );
                    assert!(
                        r.vector_size_ran >= 1 && r.vector_size_ran <= r.vector_size,
                        "{r:?}"
                    );
                    assert!(
                        r.check_error.unwrap() <= r.check_tolerance.unwrap(),
                        "{r:?}"
                    );
                    assert_eq!(r.units_per_group, 1);
                    assert_eq!(r.gdim, family.gdim(cell));
                    let points = r.cells * r.points_per_cell;
                    assert!((3_000..=30_000).contains(&points), "{r:?}");
                    assert_eq!(DegreeRecord::from_json(&r.to_json()).unwrap(), *r);
                }
            }
        }
    }

    /// A timed run of J and the pull-back (1 and 9 iterations) on quadrilaterals at p = 2, twice:
    /// two timed runs each, and the second identical run compiles nothing.
    #[test]
    fn timed_smoke_and_second_call_compile_time() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let mut o = options(
            &[DegreeFamily::Jacobian, DegreeFamily::PullBack],
            &[CellType::Quadrilateral],
            false,
        );
        o.degrees = vec![2];
        o.node_families = vec![NodeFamily::Equispaced];
        o.scalars = vec![Scalar::F32];
        let first = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        let second = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        assert_eq!(first.len(), 3);
        assert_eq!(
            first.iter().map(|r| r.iterations).collect::<Vec<_>>(),
            [None, Some(1), Some(9)]
        );
        for (a, b) in first.iter().zip(&second) {
            for r in [a, b] {
                assert_eq!(r.status, "ok", "{r:?}");
                assert_eq!(r.times_s.len(), 2, "{r:?}");
                assert!(r.median_s > 0.0 && r.calls_per_run >= 1, "{r:?}");
                assert_eq!(r.vector_size, r.policy_vector_size);
            }
            // The kernels of the first run are cached: no compile time is left.
            assert!(
                b.compile_s <= 0.05 + 0.5 * a.compile_s.min(0.1),
                "first {} s, second {} s",
                a.compile_s,
                b.compile_s
            );
        }
        // Nine iterations take longer than one.
        assert!(first[2].median_s > first[1].median_s);
    }

    /// A resumed run skips the configurations it has records of; a stopped process's
    /// configurations become `too slow to compile` before their kernel's first record and
    /// `too slow` after it, without running.
    #[test]
    fn resumed_and_stopped_runs() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let mut o = options(&[DegreeFamily::Points], &[CellType::Interval], true);
        o.degrees = vec![1];
        o.node_families = vec![NodeFamily::Equispaced];
        o.scalars = vec![Scalar::F32];
        o.sizes = vec![10_000, 20_000];
        o.launch_vector_sizes = LaunchVectorSizes::Scalar;
        let first = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        assert_eq!(first.len(), 2);
        o.done = first.clone();
        assert!(
            degree_sweep(&client, &machine, &o, &mut |_| ())
                .unwrap()
                .is_empty()
        );

        o.stopped_after = Some(60.0);
        o.done = first[..1].to_vec();
        let rest = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        assert_eq!(rest.len(), 1);
        assert_eq!(rest[0].nominal_points, 20_000);
        assert_eq!(rest[0].status, "too slow (process stopped after 60 s)");
        o.done.clear();
        let none = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        let statuses: Vec<&str> = none.iter().map(|r| r.status.as_str()).collect();
        let want = "too slow to compile (process stopped after 60 s)";
        assert_eq!(statuses, [want, want]);
        assert!(none.iter().all(|r| r.cells >= 1 && r.times_s.is_empty()));
    }

    #[test]
    fn records_name_the_strategy_that_ran() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let mut o = options(&[DegreeFamily::Points], &[CellType::Interval], true);
        o.degrees = vec![2];
        o.node_families = vec![NodeFamily::Equispaced];
        o.scalars = vec![Scalar::F32];
        o.launch_vector_sizes = LaunchVectorSizes::Scalar;
        o.strategies = ["policy", "direct", "loops", "cooperative"]
            .map(str::to_owned)
            .to_vec();
        let records = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap();
        assert_eq!(records.len(), 4);
        let ran: Vec<(&str, &str)> = records[..3]
            .iter()
            .map(|r| (r.strategy.as_str(), r.strategy_ran.as_str()))
            .collect();
        assert_eq!(
            ran,
            [
                ("policy", "direct"),
                ("direct", "direct"),
                ("loops", "loops")
            ]
        );
        assert!(records[..3].iter().all(|r| r.status == "ok"), "{records:?}");
        // `cooperative` runs from T10a (docs/plans/M9h.md): until then a failed record.
        assert!(
            records[3].status.starts_with("failed: ") && records[3].status.contains("cooperative"),
            "{}",
            records[3].status
        );
        o.strategies = vec!["fast".to_owned()];
        let e = degree_sweep(&client, &machine, &o, &mut |_| ()).unwrap_err();
        assert!(e.contains("unknown strategy fast"), "{e}");
        o.strategies = vec!["direct".to_owned()];
        o.sizes = vec![20_000, 10_000];
        assert!(degree_sweep(&client, &machine, &o, &mut |_| ()).is_err());
    }

    #[test]
    fn families_apply_to_their_cells() {
        assert!(DegreeFamily::Normals.applies_to(CellType::Quadrilateral));
        assert!(!DegreeFamily::Normals.applies_to(CellType::Prism));
        assert_eq!(DegreeFamily::Normals.gdim(CellType::Interval), 2);
        assert_eq!(DegreeFamily::FacetNormals.gdim(CellType::Pyramid), 3);
        assert!(offers(CellType::Hexahedron, NodeFamily::Gll));
        assert!(!offers(CellType::Prism, NodeFamily::Gll));
        assert_eq!(max_degree(CellType::Pyramid), 3);
    }
}
