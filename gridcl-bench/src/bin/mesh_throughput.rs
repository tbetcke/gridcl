//! End-to-end throughput of mesh creation and geometry (docs/plans/M3s.md, T13, question 10)
//! on the device named by `GRIDCL_TEST_DEVICE`, with the vector policy and with every kernel
//! scalar, as JSON records (`gridcl_bench::record::ThroughputRecord`).
//!
//! ```text
//! cargo run --release -p gridcl-bench --bin mesh_throughput -- [--target NAME] [--cells N]
//!     [--measures entities,factors] [--vector-size policy|scalar|both] [--runs R]
//!     [--fallback-reason TEXT] [--out FILE]
//! ```
//!
//! - **Entities per second:** the entities that `create_entities(1)` and `create_entities(2)`
//!   create on a P1 tetrahedron box and on a P1 hexahedron box of about `--cells` cells (default
//!   10^6), over the time of those two calls, each run on a fresh copy of the mesh made outside
//!   the timing.
//! - **Factors per second:** the (cell, quadrature point) pairs for which X, J, det J and K are
//!   computed on a P2 tetrahedron box of about `--cells` cells with a degree-4 rule, over the
//!   time of `physical_points`, `jacobians`, `detj` and `pseudo_inverse_jacobians`. Each field
//!   is dropped after its call; the E-vector and tabulations are cached by the warm-up.
//!
//! Everything runs in f64, with the protocol of `gridcl_bench::protocol`: a warm-up checked
//! against closed forms (entity counts; the volume and centroid of the unit cube; K J = I),
//! then the median of `--runs` timed runs (default 10). `--vector-size` chooses
//! `with_vector_size(Policy)`, `with_vector_size(Scalar)` or both (default). `--target`,
//! `--fallback-reason` and `--out` are as in `vector_sweep`; the commit likewise.

use std::any::Any;
use std::path::PathBuf;
use std::process::ExitCode;

use cubecl::prelude::Client;
use gridcl::{Cell3, D3, DeviceGrid, QuadRule, VolumeMesh};
use gridcl_bench::families::Machine;
use gridcl_bench::families::mesh::{hexahedron_side, tetrahedron_side};
use gridcl_bench::protocol::{self, RUNS, Workload, measure};
use gridcl_bench::record::{Measure, Scalar, ThroughputRecord, write_records};
use gridcl_device::testing::with_vector_size;
use gridcl_device::{KernelKind, VectorSizeOverride, check_precision};

const USAGE: &str = "usage: mesh_throughput [--target NAME] [--cells N] \
    [--measures entities,factors] [--vector-size policy|scalar|both] [--runs R] \
    [--fallback-reason TEXT] [--out FILE]";

/// Cells of the meshes by default.
const CELLS: usize = 1_000_000;
/// Degree of exactness of the factor rule.
const RULE_DEGREE: u32 = 4;

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
    let client = gridcl_testing::test_client();
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
    measures: Vec<Measure>,
    overrides: Vec<VectorSizeOverride>,
    runs: usize,
    fallback_reason: Option<String>,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut parsed = Args {
            target: std::env::var("GRIDCL_HPC_TARGET")
                .ok()
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "local".to_owned()),
            cells: CELLS,
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
    work: Box<dyn Workload>,
}

/// Measures everything `args` asks for and passes each record to `on_record`.
fn run(
    client: &Client,
    args: &Args,
    on_record: &mut dyn FnMut(&ThroughputRecord) -> Result<(), String>,
) -> Result<Vec<ThroughputRecord>, String> {
    check_precision::<f64>(client).map_err(|e| format!("the measures run in f64: {e}"))?;
    let machine = Machine::new(client);
    eprintln!(
        "mesh_throughput: {} on {} ({}), about {} cells, {} runs",
        machine.backend, machine.device, machine.commit, args.cells, args.runs
    );
    let mut jobs = Vec::new();
    for &m in &args.measures {
        match m {
            Measure::EntitiesPerSecond => {
                for cell in [Cell3::Tetrahedron, Cell3::Hexahedron] {
                    let w = Entities::new(client, cell, args.cells)?;
                    jobs.push(Job {
                        measure: m,
                        kind: KernelKind::TopologyKeys,
                        mesh: w.mesh(),
                        cells: w.cells(),
                        work: Box::new(w),
                    });
                }
            }
            Measure::FactorsPerSecond => {
                let w = Factors::new(client, args.cells)?;
                jobs.push(Job {
                    measure: m,
                    kind: KernelKind::GeomFactors,
                    mesh: w.mesh(),
                    cells: w.cells(),
                    work: Box::new(w),
                });
            }
        }
    }
    let mut records = Vec::new();
    for Job {
        measure: m,
        kind,
        mesh,
        cells,
        work,
    } in &mut jobs
    {
        for &v in &args.overrides {
            let measured = with_vector_size(v, || measure(client, *kind, args.runs, work.as_mut()))
                .map_err(|e| format!("{m:?} with {}: {e}", override_name(v)))?;
            let median_s = measured.median_s();
            let items = work.items();
            let record = ThroughputRecord {
                backend: machine.backend.clone(),
                device: machine.device.clone(),
                device_class: machine.class,
                target: args.target.clone(),
                measure: *m,
                mesh: mesh.clone(),
                cells: *cells,
                scalar: Scalar::F64,
                vector_size_override: override_name(v).to_owned(),
                items,
                times_s: measured.times_s.clone(),
                median_s,
                per_s: items as f64 / median_s,
                commit: machine.commit.clone(),
                profile: protocol::profile().to_owned(),
                cores: machine.cores.clone(),
                fallback_reason: args.fallback_reason.clone(),
                node_load: None,
            };
            on_record(&record)?;
            records.push(record);
        }
    }
    Ok(records)
}

fn device_error(e: gridcl::GridError) -> String {
    e.to_string()
}

/// Edges and faces of `create_entities(1)` and `(2)` on a P1 box.
struct Entities {
    cell: Cell3,
    side: usize,
    mesh: VolumeMesh<f64>,
    fresh: Option<VolumeMesh<f64>>,
}

impl Entities {
    fn new(client: &Client, cell: Cell3, cells: usize) -> Result<Entities, String> {
        let side = match cell {
            Cell3::Hexahedron => hexahedron_side(cells),
            _ => tetrahedron_side(cells),
        };
        let mesh = gridcl_testing::cube_box(cell, side, 1)
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

impl Workload for Entities {
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
            .downcast::<VolumeMesh<f64>>()
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

/// X, J, det J and K on a P2 tetrahedron box with a degree-4 rule.
struct Factors {
    side: usize,
    mesh: VolumeMesh<f64>,
    rule: QuadRule<f64, D3>,
}

impl Factors {
    fn new(client: &Client, cells: usize) -> Result<Factors, String> {
        let side = tetrahedron_side(cells);
        let mesh: VolumeMesh<f64> = gridcl_testing::cube_box(Cell3::Tetrahedron, side, 2)
            .build(client)
            .map_err(device_error)?;
        let rule = QuadRule::for_grid(&mesh, RULE_DEGREE).map_err(device_error)?;
        Ok(Factors { side, mesh, rule })
    }

    fn points(&self) -> usize {
        self.rule.get(Cell3::Tetrahedron).map_or(0, |pts| pts.len())
    }

    fn weights(&self) -> Result<Vec<f64>, String> {
        Ok(self
            .rule
            .get(Cell3::Tetrahedron)
            .map_err(device_error)?
            .weights()
            .to_vec())
    }

    fn mesh(&self) -> String {
        "P2 tetrahedron box".to_owned()
    }

    fn cells(&self) -> usize {
        6 * self.side.pow(3)
    }
}

impl Workload for Factors {
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
    /// on the unit cube, within n ε of recursive summation over n positive terms, and K J = I
    /// within 1e-12 times the box side (the condition of J grows with it).
    #[allow(clippy::needless_range_loop)] // K J indexes two column-major matrices
    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        let w = self.weights()?;
        let q = w.len();
        let host = |e: gridcl::GridError| e.to_string();
        let detj = self.mesh.detj(&self.rule).map_err(host)?.blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let x = self.mesh.physical_points(&self.rule).map_err(host)?.blocks[0]
            .to_host_cell_major()
            .map_err(host)?;
        let (mut volume, mut moment) = (0.0, [0.0; 3]);
        for (i, (d, x)) in detj.iter().zip(&x).enumerate() {
            let wd = w[i % q] * d;
            volume += wd;
            for g in 0..3 {
                moment[g] += wd * x[g];
            }
        }
        let bound = detj.len() as f64 * f64::EPSILON;
        if (volume - 1.0).abs() > bound || moment.iter().any(|m| (m - 0.5).abs() > bound) {
            return Err(format!(
                "volume {volume} and moments {moment:?} of the unit cube, expected 1 and 1/2"
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
        let tol = 1e-12 * self.side as f64;
        for (i, (k, j)) in k.iter().zip(&j).enumerate() {
            for a in 0..3 {
                for b in 0..3 {
                    let kj: f64 = (0..3).map(|g| k[g][a] * j[b][g]).sum();
                    let id = if a == b { 1.0 } else { 0.0 };
                    if (kj - id).abs() > tol {
                        return Err(format!(
                            "K J = {kj} at ({a}, {b}) of point {i}, expected {id}"
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
        let client = gridcl_testing::test_client();
        if check_precision::<f64>(&client).is_err() {
            eprintln!("entity_counts_of_one_cube skipped: the device has no f64");
            return;
        }
        // The Kuhn cube: 19 edges, 18 faces; the hexahedron: 12 edges, 6 faces.
        let tet = Entities::new(&client, Cell3::Tetrahedron, 6).unwrap();
        assert_eq!((tet.side, tet.expected()), (1, [19, 18]));
        let hex = Entities::new(&client, Cell3::Hexahedron, 1).unwrap();
        assert_eq!((hex.side, hex.expected()), (1, [12, 6]));
    }

    /// Both measures at about 10^3 cells, two runs, with the policy and scalar: one record per
    /// mesh and override, counting the boxes' cells and passing the warm-up checks.
    #[test]
    fn smoke_run_at_a_thousand_cells() {
        let client = gridcl_testing::test_client();
        if check_precision::<f64>(&client).is_err() {
            eprintln!("smoke_run_at_a_thousand_cells skipped: the device has no f64");
            return;
        }
        let args = Args::parse(
            ["--cells", "1000", "--runs", "2", "--target", "test"]
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
        let got: Vec<(Measure, &str, usize, &str)> = records
            .iter()
            .map(|r| {
                (
                    r.measure,
                    r.mesh.as_str(),
                    r.cells,
                    r.vector_size_override.as_str(),
                )
            })
            .collect();
        use Measure::{EntitiesPerSecond as E, FactorsPerSecond as F};
        assert_eq!(
            got,
            [
                (E, "P1 tetrahedron box", 750, "policy"),
                (E, "P1 tetrahedron box", 750, "scalar"),
                (E, "P1 hexahedron box", 1000, "policy"),
                (E, "P1 hexahedron box", 1000, "scalar"),
                (F, "P2 tetrahedron box", 750, "policy"),
                (F, "P2 tetrahedron box", 750, "scalar"),
            ]
        );
        for r in &records {
            assert_eq!(
                (r.scalar, r.times_s.len(), r.target.as_str()),
                (Scalar::F64, 2, "test")
            );
            assert!(r.items > r.cells && r.median_s > 0.0, "{r:?}");
            assert_eq!(r.per_s, r.items as f64 / r.median_s);
            assert_eq!(
                ThroughputRecord::from_json(&r.to_json()).unwrap(),
                *r,
                "JSON round trip"
            );
        }
    }
}
