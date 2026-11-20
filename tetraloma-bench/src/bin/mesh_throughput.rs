//! End-to-end throughput of mesh creation and geometry (docs/plans/M3s.md, T13, question 10;
//! from M10 at p = 1, 4 and 8, ADR-0041 and ADR-0058) on the device named by
//! `TETRALOMA_TEST_DEVICE`, with the vector policy and with every kernel scalar, as JSON records
//! (`tetraloma_bench::record::ThroughputRecord`).
//!
//! ```text
//! cargo run --release -p tetraloma-bench --bin mesh_throughput -- [--target NAME] [--cells N]
//!     [--points N] [--degrees 1,4,8] [--types f32,f64] [--measures entities,factors]
//!     [--vector-size policy|scalar|both] [--runs R] [--fallback-reason TEXT] [--out FILE]
//! ```
//!
//! - **Entities per second:** the entities that `create_entities(1)` and `create_entities(2)`
//!   create on a P1 tetrahedron box and on a P1 hexahedron box of about `--cells` cells (default
//!   10^6), over the time of those two calls, each run on a fresh copy of the mesh made outside
//!   the timing.
//! - **Factors per second:** the (cell, quadrature point) pairs for which X, J, det J and K are
//!   computed on a P`p` tetrahedron box and a P`p` hexahedron box of about `--points` points
//!   (default 10^6) with a Gauss rule of degree 2p, for every p of `--degrees` (default 2),
//!   over the time of `physical_points`, `jacobians`, `detj` and `pseudo_inverse_jacobians`.
//!   Each field is dropped after its call; the E-vector and tabulations are cached by the
//!   warm-up. The meshes are named `P<p> <cell> box`.
//!
//! Each measure runs in every type of `--types` the device supports (default f64), with the
//! protocol of `tetraloma_bench::protocol`: a warm-up checked against closed forms (entity counts;
//! the volume and centroid of the unit cube; K J = I), then the median of `--runs` timed runs
//! (default 10; the candidate protocol's five for M10, ADR-0057), each of at least 20 ms
//! (ADR-0018). `--vector-size` chooses `with_vector_size(Policy)`, `with_vector_size(Scalar)`
//! or both (default). `--target`, `--fallback-reason` and `--out` are as in `vector_sweep`; the
//! commit likewise.

use std::any::Any;
use std::path::PathBuf;
use std::process::ExitCode;

use cubecl::prelude::Client;
use tetraloma::{Cell3, D3, DeviceGrid, QuadRule, Real, VolumeMesh};
use tetraloma_bench::families::Machine;
use tetraloma_bench::families::degree::{box_cells, points_per_cell};
use tetraloma_bench::families::mesh::{hexahedron_side, nearest_side, tetrahedron_side};
use tetraloma_bench::protocol::{self, RUNS, Workload, measure};
use tetraloma_bench::record::{LegacyClass, Measure, Scalar, ThroughputRecord, write_records};
use tetraloma_device::testing::with_vector_size;
use tetraloma_device::{KernelKind, VectorSizeOverride, check_precision};

const USAGE: &str = "usage: mesh_throughput [--target NAME] [--cells N] [--points N] \
    [--degrees LIST] [--types f32,f64] [--measures entities,factors] \
    [--vector-size policy|scalar|both] [--runs R] [--fallback-reason TEXT] [--out FILE]";

/// Cells of the entity meshes by default.
const CELLS: usize = 1_000_000;
/// Quadrature points of the factor meshes by default.
const POINTS: usize = 1_000_000;
/// The highest degree of the factor meshes (M9h's range).
const MAX_DEGREE: u32 = 8;

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("mesh_throughput: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if cfg!(debug_assertions) {
        eprintln!("mesh_throughput: debug build; record only release builds (--release)");
    }
    let client = tetraloma_testing::test_client();
    let mut records = Vec::new();
    let result = run(&client, &args, &mut |record| {
        eprintln!(
            "{:?} {} with {}: {} items, median {:.3} ms, {:.3e} per s",
            record.measure,
            record.mesh,
            record.vector_size_override,
            record.items,
            record.median_s * 1e3,
            record.per_s
        );
        records.push(record.to_json());
        match &args.out {
            Some(path) => write_records(path, &records),
            None => Ok(()),
        }
    });
    if let Err(message) = result {
        eprintln!("mesh_throughput: {message}");
        return ExitCode::FAILURE;
    }
    if args.out.is_none() {
        println!(
            "{}",
            serde_json::to_string_pretty(&records).expect("JSON of records")
        );
    }
    ExitCode::SUCCESS
}

/// Command-line options.
#[derive(Clone, Debug)]
struct Args {
    target: String,
    cells: usize,
    points: usize,
    degrees: Vec<u32>,
    scalars: Vec<Scalar>,
    measures: Vec<Measure>,
    overrides: Vec<VectorSizeOverride>,
    runs: usize,
    fallback_reason: Option<String>,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut parsed = Args {
            target: std::env::var("TETRALOMA_HPC_TARGET")
                .ok()
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "local".to_owned()),
            cells: CELLS,
            points: POINTS,
            degrees: vec![2],
            scalars: vec![Scalar::F64],
            measures: vec![Measure::EntitiesPerSecond, Measure::FactorsPerSecond],
            overrides: vec![VectorSizeOverride::Policy, VectorSizeOverride::Scalar],
            runs: RUNS,
            fallback_reason: None,
            out: None,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--target" => parsed.target = value()?,
                "--cells" => parsed.cells = positive(&flag, &value()?)?,
                "--points" => parsed.points = positive(&flag, &value()?)?,
                "--degrees" => {
                    parsed.degrees = value()?
                        .split(',')
                        .map(|d| {
                            d.parse::<u32>()
                                .ok()
                                .filter(|p| (1..=MAX_DEGREE).contains(p))
                                .ok_or_else(|| format!("--degrees: not a degree 1-8: '{d}'"))
                        })
                        .collect::<Result<_, _>>()?;
                }
                "--types" => {
                    parsed.scalars = value()?
                        .split(',')
                        .map(|t| match Scalar::named(t) {
                            Some(s @ (Scalar::F32 | Scalar::F64)) => Ok(s),
                            _ => Err(format!("--types: not f32 or f64: '{t}'")),
                        })
                        .collect::<Result<_, _>>()?;
                }
                "--runs" => parsed.runs = positive(&flag, &value()?)?,
                "--measures" => {
                    parsed.measures = value()?
                        .split(',')
                        .map(|m| match m {
                            "entities" => Ok(Measure::EntitiesPerSecond),
                            "factors" => Ok(Measure::FactorsPerSecond),
                            _ => Err(format!("--measures: unknown measure '{m}'")),
                        })
                        .collect::<Result<_, _>>()?;
                }
                "--vector-size" => {
                    parsed.overrides = match value()?.as_str() {
                        "policy" => vec![VectorSizeOverride::Policy],
                        "scalar" => vec![VectorSizeOverride::Scalar],
                        "both" => vec![VectorSizeOverride::Policy, VectorSizeOverride::Scalar],
                        v => return Err(format!("--vector-size: unknown value '{v}'")),
                    };
                }
                "--fallback-reason" => parsed.fallback_reason = Some(value()?),
                "--out" => parsed.out = Some(PathBuf::from(value()?)),
                _ => return Err(format!("unknown argument {flag}")),
            }
        }
        Ok(parsed)
    }
}

fn positive(flag: &str, v: &str) -> Result<usize, String> {
    v.parse()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{flag}: not a positive integer: {v}"))
}

/// The name of `v` in records: `policy` or `scalar`.
fn override_name(v: VectorSizeOverride) -> &'static str {
    match v {
        VectorSizeOverride::Scalar => "scalar",
        _ => "policy",
    }
}

/// One workload and what its records say about it.
struct Job {
    measure: Measure,
    /// The kind whose plans the warm-up must make.
    kind: KernelKind,
    mesh: String,
    cells: usize,
    scalar: Scalar,
    work: Box<dyn Workload>,
}

/// The jobs of `args` in `F`.
fn jobs<F: Real>(client: &Client, args: &Args, scalar: Scalar) -> Result<Vec<Job>, String> {
    let mut jobs = Vec::new();
    for &m in &args.measures {
        match m {
            Measure::EntitiesPerSecond => {
                for cell in [Cell3::Tetrahedron, Cell3::Hexahedron] {
                    let w = Entities::<F>::new(client, cell, args.cells)?;
                    jobs.push(Job {
                        measure: m,
                        kind: KernelKind::TopologyKeys,
                        mesh: w.mesh(),
                        cells: w.cells(),
                        scalar,
                        work: Box::new(w),
                    });
                }
            }
            Measure::FactorsPerSecond => {
                for &p in &args.degrees {
                    for cell in [Cell3::Tetrahedron, Cell3::Hexahedron] {
                        let w = Factors::<F>::new(client, cell, p, args.points)?;
                        jobs.push(Job {
                            measure: m,
                            kind: KernelKind::GeomFactors,
                            mesh: w.mesh(),
                            cells: w.cells(),
                            scalar,
                            work: Box::new(w),
                        });
                    }
                }
            }
        }
    }
    Ok(jobs)
}

/// Measures everything `args` asks for and passes each record to `on_record`.
fn run(
    client: &Client,
    args: &Args,
    on_record: &mut dyn FnMut(&ThroughputRecord) -> Result<(), String>,
) -> Result<Vec<ThroughputRecord>, String> {
    let machine = Machine::new(client);
    eprintln!(
        "mesh_throughput: {} on {} ({}), about {} cells and {} points, degrees {:?}, {} runs",
        machine.backend,
        machine.device,
        machine.commit,
        args.cells,
        args.points,
        args.degrees,
        args.runs
    );
    let mut records = Vec::new();
    for &scalar in &args.scalars {
        let mut jobs = match scalar {
            Scalar::F64 => {
                if let Err(e) = check_precision::<f64>(client) {
                    eprintln!("mesh_throughput: f64 skipped: {e}");
                    continue;
                }
                jobs::<f64>(client, args, scalar)?
            }
            _ => jobs::<f32>(client, args, scalar)?,
        };
        for Job {
            measure: m,
            kind,
            mesh,
            cells,
            scalar,
            work,
        } in &mut jobs
        {
            for &v in &args.overrides {
                let measured =
                    with_vector_size(v, || measure(client, *kind, args.runs, work.as_mut()))
                        .map_err(|e| {
                            format!(
                                "{m:?} {mesh} {} with {}: {e}",
                                scalar.name(),
                                override_name(v)
                            )
                        })?;
                let median_s = measured.median_s();
                let items = work.items();
                let record = ThroughputRecord {
                    backend: machine.backend.clone(),
                    device: machine.device.clone(),
                    device_class: machine.class,
                    legacy_class: LegacyClass::of_new(machine.class),
                    target: args.target.clone(),
                    measure: *m,
                    mesh: mesh.clone(),
                    cells: *cells,
                    scalar: *scalar,
                    vector_size_override: override_name(v).to_owned(),
                    items,
                    times_s: measured.times_s.clone(),
                    median_s,
                    calls_per_run: measured.calls_per_run,
                    calibration_s: measured.calibration_s,
                    per_s: items as f64 / median_s,
                    commit: machine.commit.clone(),
                    profile: protocol::profile().to_owned(),
                    cores: machine.cores.clone(),
                    load_width: Some(machine.load_width),
                    fallback_reason: args.fallback_reason.clone(),
                    node_load: None,
                };
                on_record(&record)?;
                records.push(record);
            }
        }
    }
    Ok(records)
}

fn device_error(e: tetraloma::GridError) -> String {
    e.to_string()
}

/// Edges and faces of `create_entities(1)` and `(2)` on a P1 box.
struct Entities<F: Real> {
    cell: Cell3,
    side: usize,
    mesh: VolumeMesh<F>,
    fresh: Option<VolumeMesh<F>>,
}

impl<F: Real> Entities<F> {
    fn new(client: &Client, cell: Cell3, cells: usize) -> Result<Entities<F>, String> {
        let side = match cell {
            Cell3::Hexahedron => hexahedron_side(cells),
            _ => tetrahedron_side(cells),
        };
        let mesh = tetraloma_testing::cube_box(cell, side, 1)
            .build(client)
            .map_err(device_error)?;
        Ok(Entities {
            cell,
            side,
            mesh,
            fresh: None,
        })
    }

    /// Edges and faces of the box, from closed forms: for tetrahedra (6 s³ cells, 12 s²
    /// boundary triangles) F = 12 s³ + 6 s² and Euler's V − E + F − C = 1.
    fn expected(&self) -> [usize; 2] {
        let s = self.side;
        match self.cell {
            Cell3::Hexahedron => [3 * s * (s + 1) * (s + 1), 3 * s * s * (s + 1)],
            _ => [
                (s + 1).pow(3) + 6 * s.pow(3) + 6 * s * s - 1,
                12 * s.pow(3) + 6 * s * s,
            ],
        }
    }

    fn mesh(&self) -> String {
        format!("P1 {} box", format!("{:?}", self.cell).to_lowercase())
    }

    fn cells(&self) -> usize {
        match self.cell {
            Cell3::Hexahedron => self.side.pow(3),
            _ => 6 * self.side.pow(3),
        }
    }
}

impl<F: Real> Workload for Entities<F> {
    fn items(&self) -> usize {
        self.expected().iter().sum()
    }

    fn bytes(&self) -> Option<usize> {
        None
    }

    fn prepare(&mut self) -> Result<(), String> {
        self.fresh = Some(self.mesh.try_clone().map_err(device_error)?);
        Ok(())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let mut mesh = self.fresh.take().ok_or("no fresh mesh")?;
        mesh.create_entities(1).map_err(device_error)?;
        mesh.create_entities(2).map_err(device_error)?;
        Ok(Box::new(mesh))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let mesh = out
            .downcast::<VolumeMesh<F>>()
            .map_err(|_| "the output is not a mesh")?;
        for (d, want) in [1, 2].into_iter().zip(self.expected()) {
            let got: usize = mesh
                .entity_blocks(d)
                .map_err(device_error)?
                .iter()
                .map(|b| b.index_map.num_local())
                .sum();
            if got != want {
                return Err(format!(
                    "{}: {got} entities of dimension {d}, expected {want}",
                    self.mesh()
                ));
            }
        }
        Ok(())
    }
}

/// X, J, det J and K on a P`p` box of tetrahedra or hexahedra with a Gauss rule of degree 2p.
struct Factors<F: Real> {
    cell: Cell3,
    p: u32,
    side: usize,
    mesh: VolumeMesh<F>,
    rule: QuadRule<F, D3>,
}

impl<F: Real> Factors<F> {
    /// The box whose cells hold about `points` points.
    fn new(client: &Client, cell: Cell3, p: u32, points: usize) -> Result<Factors<F>, String> {
        let q = points_per_cell(cell.into(), p);
        let side = nearest_side((points / q).max(1), |s| box_cells(cell.into(), s));
        let mesh: VolumeMesh<F> = tetraloma_testing::cube_box(cell, side, p)
            .build(client)
            .map_err(device_error)?;
        let rule = QuadRule::for_grid(&mesh, 2 * p).map_err(device_error)?;
        Ok(Factors {
            cell,
            p,
            side,
            mesh,
            rule,
        })
    }

    fn points(&self) -> usize {
        self.rule.get(self.cell).map_or(0, |pts| pts.len())
    }

    fn weights(&self) -> Result<Vec<f64>, String> {
        Ok(self
            .rule
            .get(self.cell)
            .map_err(device_error)?
            .weights()
            .iter()
            .map(|w| w.widen())
            .collect())
    }

    fn mesh(&self) -> String {
        format!(
            "P{} {} box",
            self.p,
            format!("{:?}", self.cell).to_lowercase()
        )
    }

    fn cells(&self) -> usize {
        box_cells(self.cell.into(), self.side)
    }

    /// The tolerance of K J = I: 1e-12 in f64 and 1e-5 in f32 (CLAUDE.md), times the box side
    /// for the conditioning of J (entries 1/side). At 10^6 points the errors stay below 10^-2
    /// of it at p = 1, 4 and 8 on both cells (measured in M10 T2a on the CPU runtime).
    fn tolerance(&self) -> f64 {
        let base = if size_of::<F>() == 4 { 1e-5 } else { 1e-12 };
        base * self.side as f64
    }
}

impl<F: Real> Workload for Factors<F> {
    fn items(&self) -> usize {
        self.cells() * self.points()
    }

    fn bytes(&self) -> Option<usize> {
        None
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        drop(
            self.mesh
                .physical_points(&self.rule)
                .map_err(device_error)?,
        );
        drop(self.mesh.jacobians(&self.rule).map_err(device_error)?);
        drop(self.mesh.detj(&self.rule).map_err(device_error)?);
        drop(
            self.mesh
                .pseudo_inverse_jacobians(&self.rule)
                .map_err(device_error)?,
        );
        Ok(Box::new(()))
    }

    /// Repeats the calls one field at a time: Σ w det J = 1 and Σ w det J X = (1/2, 1/2, 1/2)
    /// on the unit cube, within n ε of recursive summation over n positive terms in f64 (1e-4
    /// in f32), and K J = I within [`Factors::tolerance`].
    #[allow(clippy::needless_range_loop)] // K J indexes two column-major matrices
    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        let w = self.weights()?;
        let q = w.len();
        let host = |e: tetraloma::GridError| e.to_string();
        let detj = self.mesh.detj(&self.rule).map_err(host)?.blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let x = self.mesh.physical_points(&self.rule).map_err(host)?.blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let (mut volume, mut moment) = (0.0, [0.0; 3]);
        for (i, (d, x)) in detj.iter().zip(&x).enumerate() {
            let wd = w[i % q] * d.widen();
            volume += wd;
            for g in 0..3 {
                moment[g] += wd * x[g].widen();
            }
        }
        let bound = if size_of::<F>() == 4 {
            1e-4
        } else {
            detj.len() as f64 * f64::EPSILON
        };
        if (volume - 1.0).abs() > bound || moment.iter().any(|m| (m - 0.5).abs() > bound) {
            return Err(format!(
                "{}: volume {volume} and moments {moment:?} of the unit cube, expected 1 and 1/2",
                self.mesh()
            ));
        }
        drop(x);
        let j = self.mesh.jacobians(&self.rule).map_err(host)?.blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let k = self
            .mesh
            .pseudo_inverse_jacobians(&self.rule)
            .map_err(host)?
            .blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let tol = self.tolerance();
        for (i, (k, j)) in k.iter().zip(&j).enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    let kj: f64 = (0..3).map(|g| k[g][a].widen() * j[b][g].widen()).sum();
                    let id = if a == b { 1.0 } else { 0.0 };
                    let err = (kj - id).abs();
                    // A NaN error fails too.
                    if err.is_nan() || err > tol {
                        return Err(format!(
                            "{}: K J = {kj} at ({a}, {b}) of point {i}, expected {id}",
                            self.mesh()
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entity_counts_of_one_cube() {
        let client = tetraloma_testing::test_client();
        if check_precision::<f64>(&client).is_err() {
            eprintln!("entity_counts_of_one_cube skipped: the device has no f64");
            return;
        }
        // The Kuhn cube: 19 edges, 18 faces; the hexahedron: 12 edges, 6 faces.
        let tet = Entities::<f64>::new(&client, Cell3::Tetrahedron, 6).unwrap();
        assert_eq!((tet.side, tet.expected()), (1, [19, 18]));
        let hex = Entities::<f64>::new(&client, Cell3::Hexahedron, 1).unwrap();
        assert_eq!((hex.side, hex.expected()), (1, [12, 6]));
    }

    /// Both measures at about 10^3 cells and 2000 points, p = 1 and 2, f32 and f64, two runs,
    /// with the policy and scalar: one record per type, mesh and override, counting the boxes'
    /// cells and passing the warm-up checks.
    #[test]
    fn smoke_run_at_a_thousand_cells() {
        let client = tetraloma_testing::test_client();
        let args = Args::parse(
            [
                "--cells",
                "1000",
                "--points",
                "2000",
                "--degrees",
                "1,2",
                "--types",
                "f32,f64",
                "--runs",
                "2",
                "--target",
                "test",
            ]
            .into_iter()
            .map(str::to_owned),
        )
        .unwrap();
        let mut seen = 0;
        let records = run(&client, &args, &mut |_| {
            seen += 1;
            Ok(())
        })
        .unwrap();
        assert_eq!(seen, records.len());
        let got: Vec<(Scalar, Measure, &str, &str)> = records
            .iter()
            .map(|r| {
                (
                    r.scalar,
                    r.measure,
                    r.mesh.as_str(),
                    r.vector_size_override.as_str(),
                )
            })
            .collect();
        use Measure::{EntitiesPerSecond as E, FactorsPerSecond as F};
        let meshes = [
            (E, "P1 tetrahedron box"),
            (E, "P1 hexahedron box"),
            (F, "P1 tetrahedron box"),
            (F, "P1 hexahedron box"),
            (F, "P2 tetrahedron box"),
            (F, "P2 hexahedron box"),
        ];
        let has_f64 = check_precision::<f64>(&client).is_ok();
        let scalars: &[Scalar] = if has_f64 {
            &[Scalar::F32, Scalar::F64]
        } else {
            &[Scalar::F32]
        };
        let want: Vec<(Scalar, Measure, &str, &str)> = scalars
            .iter()
            .flat_map(|&s| {
                meshes
                    .iter()
                    .flat_map(move |&(m, mesh)| ["policy", "scalar"].map(move |v| (s, m, mesh, v)))
            })
            .collect();
        assert_eq!(got, want);
        for r in &records {
            assert_eq!((r.times_s.len(), r.target.as_str()), (2, "test"));
            assert!(r.items > r.cells && r.median_s > 0.0, "{r:?}");
            assert_eq!(r.per_s, r.items as f64 / r.median_s);
            assert!(r.load_width.is_some());
            if r.measure == F {
                assert!((1000..=4000).contains(&r.items), "{r:?}");
            }
            assert_eq!(
                ThroughputRecord::from_json(&r.to_json()).unwrap(),
                *r,
                "JSON round trip"
            );
        }
    }

    /// The factor checks pass at p = 8 on both cells and in both types (the tolerances hold at
    /// the highest degree).
    #[test]
    fn factor_checks_hold_at_degree_eight() {
        let client = tetraloma_testing::test_client();
        let run_checks = |scalar: Scalar| {
            for cell in [Cell3::Tetrahedron, Cell3::Hexahedron] {
                let mut w: Box<dyn Workload> = match scalar {
                    Scalar::F64 => Box::new(Factors::<f64>::new(&client, cell, 8, 4000).unwrap()),
                    _ => Box::new(Factors::<f32>::new(&client, cell, 8, 4000).unwrap()),
                };
                let out = w.run().unwrap();
                w.check(out)
                    .unwrap_or_else(|e| panic!("{} {e}", scalar.name()));
            }
        };
        run_checks(Scalar::F32);
        if check_precision::<f64>(&client).is_ok() {
            run_checks(Scalar::F64);
        }
    }

    #[test]
    fn bad_options_are_rejected() {
        for list in [
            &["--degrees", "0"][..],
            &["--degrees", "9"],
            &["--types", "u32"],
            &["--points", "0"],
        ] {
            assert!(
                Args::parse(list.iter().map(|s| s.to_string())).is_err(),
                "{list:?}"
            );
        }
    }
}
