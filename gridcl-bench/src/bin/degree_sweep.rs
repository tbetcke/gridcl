//! The M9h degree sweep (docs/plans/M9h.md, T1; ADR-0042, items 1, 6, 7 and 12) on the device
//! named by `GRIDCL_TEST_DEVICE`, as JSON records (`gridcl_bench::record::DegreeRecord`).
//!
//! ```text
//! cargo run --release -p gridcl-bench --bin degree_sweep -- [--target NAME] [--run-id ID]
//!     [--families evector,points,...] [--cells hexahedron,...] [--degrees 1,2,...]
//!     [--node-families equispaced,gll] [--types f32,f64] [--sizes 100000,...]
//!     [--strategies direct,policy,...] [--launch-only] [--vector-sizes both|scalar|policy]
//!     [--runs R] [--fallback-reason TEXT] [--out FILE]
//! degree_sweep annotate --node-load RESULTS_DIR FILE...
//! ```
//!
//! - `--target`: the machine, `macmini` (the local Mac) or a target of
//!   scripts/hpc/remote-test.sh; default `GRIDCL_HPC_TARGET` (set in runner jobs), else `local`.
//! - `--run-id`: the job the records belong to (rule 11's paired runs share it); default
//!   `GRIDCL_BENCH_RUN_ID`, else the runner's `JOB_ID`, else `local-<unix time>`.
//! - `--families`, `--cells`, `--degrees`, `--node-families`, `--types`, `--sizes`: filters,
//!   so a run can be split by family; by default every family, cell type, degree 1–8
//!   (pyramids to 3), node family where offered, f32 and f64, and 10^5, 10^6 and 10^7 points.
//!   Names are the record names, in any case.
//! - `--strategies`: default `direct`; `policy` runs the degree policy. Until T4 only these two
//!   run.
//! - `--launch-only` (rule 7): the warm-up with its check and one more call, at vector size 1
//!   and at the policy's, no timed runs; by default equispaced nodes and 10^4 points.
//! - `--vector-sizes` (launch-only): `both` (default), `scalar` (1 only) or `policy` (the
//!   policy's only), so that a process compiles one kernel per configuration
//!   (`scripts/bench/cuda-resources.sh`).
//! - `--out`: rewritten after every record, so an interrupted sweep keeps what it measured;
//!   default stdout at the end. Committed files are `m9h-baseline-*`, `m9h-degree-*` and
//!   `m9h-launch-<backend>-<target>.json`.
//!
//! `annotate` adds the runner's node-load snapshots (`node-load-before.txt`,
//! `node-load-after.txt` in `results/<job id>/`) to every record of each file.
//!
//! For first-use compile times on CUDA, run with `CUDA_CACHE_DISABLE=1` (rule 6); the records
//! then say `kernel_cache: disabled`.

use std::path::PathBuf;
use std::process::ExitCode;

use gridcl::NodeFamily;
use gridcl_bench::families::Machine;
use gridcl_bench::families::degree::{
    CELLS, DEGREE_SIZES, DegreeFamily, DegreeOptions, LAUNCH_SIZE, LaunchVectorSizes, degree_sweep,
    node_family_named,
};
use gridcl_bench::protocol::RUNS;
use gridcl_bench::record::{NodeLoad, STRATEGIES, Scalar, annotate_node_load, write_records};
use gridcl_dims::CellType;
use serde_json::Value;

const USAGE: &str = "usage: degree_sweep [--target NAME] [--run-id ID] [--families LIST] \
    [--cells LIST] [--degrees LIST] [--node-families LIST] [--types LIST] [--sizes LIST] \
    [--strategies LIST] [--launch-only] [--vector-sizes both|scalar|policy] [--runs R] \
    [--fallback-reason TEXT] [--out FILE]\n       \
    degree_sweep annotate --node-load RESULTS_DIR FILE...";

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).peekable();
    let result = if args.peek().map(String::as_str) == Some("annotate") {
        args.next();
        annotate(args)
    } else {
        Args::parse(args).and_then(run)
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("degree_sweep: {message}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Run(message)) => {
            eprintln!("degree_sweep: {message}");
            ExitCode::FAILURE
        }
    }
}

enum Failure {
    Usage(String),
    Run(String),
}

impl From<String> for Failure {
    fn from(message: String) -> Failure {
        Failure::Usage(message)
    }
}

impl From<&str> for Failure {
    fn from(message: &str) -> Failure {
        Failure::Usage(message.to_owned())
    }
}

/// Command-line options of the sweep.
#[derive(Debug)]
struct Args {
    target: String,
    run_id: String,
    families: Vec<DegreeFamily>,
    cells: Vec<CellType>,
    degrees: Vec<u32>,
    node_families: Option<Vec<NodeFamily>>,
    scalars: Vec<Scalar>,
    sizes: Option<Vec<usize>>,
    strategies: Vec<String>,
    launch_only: bool,
    launch_vector_sizes: Option<LaunchVectorSizes>,
    runs: usize,
    fallback_reason: Option<String>,
    out: Option<PathBuf>,
}

/// The first non-empty value of the environment variables `vars`.
fn env(vars: &[&str]) -> Option<String> {
    vars.iter()
        .find_map(|v| std::env::var(v).ok().filter(|s| !s.is_empty()))
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, Failure> {
        let mut parsed = Args {
            target: env(&["GRIDCL_HPC_TARGET"]).unwrap_or_else(|| "local".to_owned()),
            run_id: env(&["GRIDCL_BENCH_RUN_ID", "JOB_ID"]).unwrap_or_else(|| {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map_or(0, |d| d.as_secs());
                format!("local-{now}")
            }),
            families: DegreeFamily::ALL.to_vec(),
            cells: CELLS.to_vec(),
            degrees: (1..=8).collect(),
            node_families: None,
            scalars: vec![Scalar::F32, Scalar::F64],
            sizes: None,
            strategies: vec!["direct".to_owned()],
            launch_only: false,
            launch_vector_sizes: None,
            runs: RUNS,
            fallback_reason: None,
            out: None,
        };
        while let Some(flag) = args.next() {
            if flag == "--launch-only" {
                parsed.launch_only = true;
                continue;
            }
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--target" => parsed.target = value()?,
                "--run-id" => parsed.run_id = value()?,
                "--families" => parsed.families = list(&flag, &value()?, DegreeFamily::named)?,
                "--cells" => {
                    parsed.cells = list(&flag, &value()?, |name| {
                        CELLS
                            .into_iter()
                            .find(|c| format!("{c:?}").eq_ignore_ascii_case(name))
                    })?
                }
                "--degrees" => {
                    parsed.degrees = list(&flag, &value()?, |v| {
                        v.parse::<u32>().ok().filter(|p| (1..=8).contains(p))
                    })?
                }
                "--node-families" => {
                    parsed.node_families = Some(list(&flag, &value()?, |v| {
                        node_family_named(&v.to_lowercase())
                    })?)
                }
                "--types" => {
                    parsed.scalars = list(&flag, &value()?, |v| {
                        Scalar::named(v).filter(|&s| s != Scalar::U32)
                    })?
                }
                "--sizes" => {
                    let mut sizes = list(&flag, &value()?, positive)?;
                    sizes.sort_unstable();
                    sizes.dedup();
                    parsed.sizes = Some(sizes);
                }
                "--strategies" => {
                    parsed.strategies = list(&flag, &value()?, |v| {
                        let v = v.to_lowercase();
                        (STRATEGIES.contains(&v.as_str()) || v == "policy").then_some(v)
                    })?
                }
                "--vector-sizes" => {
                    parsed.launch_vector_sizes = Some(match value()?.to_lowercase().as_str() {
                        "both" => LaunchVectorSizes::Both,
                        "scalar" => LaunchVectorSizes::Scalar,
                        "policy" => LaunchVectorSizes::Policy,
                        v => return Err(format!("--vector-sizes: cannot read '{v}'").into()),
                    })
                }
                "--runs" => {
                    parsed.runs = positive(&value()?).ok_or("--runs needs a positive integer")?
                }
                "--fallback-reason" => parsed.fallback_reason = Some(value()?),
                "--out" => parsed.out = Some(PathBuf::from(value()?)),
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        if parsed.launch_vector_sizes.is_some() && !parsed.launch_only {
            return Err("--vector-sizes needs --launch-only".into());
        }
        Ok(parsed)
    }

    /// The node families: as given, else both (equispaced only in a launch-only run).
    fn node_families(&self) -> Vec<NodeFamily> {
        self.node_families.clone().unwrap_or(if self.launch_only {
            vec![NodeFamily::Equispaced]
        } else {
            vec![NodeFamily::Equispaced, NodeFamily::Gll]
        })
    }

    /// The sizes: as given, else the protocol's (10^4 points in a launch-only run).
    fn sizes(&self) -> Vec<usize> {
        self.sizes.clone().unwrap_or(if self.launch_only {
            vec![LAUNCH_SIZE]
        } else {
            DEGREE_SIZES.to_vec()
        })
    }
}

/// The comma-separated items of `value`, each parsed by `item`.
fn list<T>(flag: &str, value: &str, item: impl Fn(&str) -> Option<T>) -> Result<Vec<T>, String> {
    let items: Option<Vec<T>> = value.split(',').map(|s| item(s.trim())).collect();
    items
        .filter(|items| !items.is_empty())
        .ok_or_else(|| format!("{flag}: cannot read '{value}'"))
}

fn positive(value: &str) -> Option<usize> {
    value.parse::<usize>().ok().filter(|&v| v > 0)
}

fn run(args: Args) -> Result<(), Failure> {
    if cfg!(debug_assertions) {
        eprintln!("degree_sweep: debug build; record only release builds (--release)");
    }
    let client = gridcl_testing::test_client();
    let machine = Machine::new(&client);
    eprintln!(
        "degree_sweep: {} on {} ({:?}, target {}, run {}, {}), cores {:?}, kernel cache {}",
        machine.backend,
        machine.device,
        machine.class,
        args.target,
        args.run_id,
        machine.commit,
        machine.cores,
        machine.kernel_cache
    );
    let options = DegreeOptions {
        node_families: args.node_families(),
        sizes: args.sizes(),
        target: args.target,
        run_id: args.run_id,
        families: args.families,
        cells: args.cells,
        degrees: args.degrees,
        scalars: args.scalars,
        strategies: args.strategies,
        launch_only: args.launch_only,
        launch_vector_sizes: args.launch_vector_sizes.unwrap_or_default(),
        runs: args.runs,
        fallback_reason: args.fallback_reason,
    };

    let mut written: Vec<Value> = Vec::new();
    let mut write_error = None;
    let out = args.out.as_deref();
    let records = degree_sweep(&client, &machine, &options, &mut |r| {
        eprintln!(
            "{} {} p={} {} {} {} n={} ({} cells × {}){} v={} (ran {}): {}, median {:.3} ms \
             ({} calls a run), compile {:.3} s{}",
            r.family,
            r.cell,
            r.degree,
            r.node_family,
            r.scalar.name(),
            r.strategy,
            r.nominal_points,
            r.cells,
            r.points_per_cell,
            r.iterations
                .map(|i| format!(" iterations {i}"))
                .unwrap_or_default(),
            r.vector_size,
            r.vector_size_ran,
            r.status,
            r.median_s * 1e3,
            r.calls_per_run,
            r.compile_s,
            match (r.check_error, r.check_tolerance) {
                (Some(e), Some(t)) => format!(", check {e:.1e} / {t:.1e}"),
                _ => String::new(),
            }
        );
        written.push(r.to_json());
        if let Some(path) = out
            && write_error.is_none()
        {
            write_error = write_records(path, &written).err();
        }
    })
    .map_err(Failure::Usage)?;
    if let Some(error) = write_error {
        return Err(Failure::Run(error));
    }
    if out.is_none() {
        let values: Vec<Value> = records.iter().map(|r| r.to_json()).collect();
        let text = serde_json::to_string_pretty(&values).expect("JSON of records");
        println!("{text}");
    }
    Ok(())
}

/// `annotate --node-load DIR FILE...`: sets `node_load` in every record of every file.
fn annotate(mut args: impl Iterator<Item = String>) -> Result<(), Failure> {
    let mut dir = None;
    let mut files = Vec::new();
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--node-load" => {
                dir = Some(PathBuf::from(
                    args.next().ok_or("--node-load needs a value")?,
                ))
            }
            _ if arg.starts_with("--") => return Err(format!("unknown option {arg}").into()),
            _ => files.push(PathBuf::from(arg)),
        }
    }
    let dir = dir.ok_or("annotate needs --node-load RESULTS_DIR")?;
    if files.is_empty() {
        return Err("annotate needs at least one FILE".into());
    }
    let load = NodeLoad::from_results_dir(&dir);
    eprintln!(
        "degree_sweep: node load before: {}; after: {}",
        load.before, load.after
    );
    for file in files {
        annotate_node_load(&file, &load).map_err(Failure::Run)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, Failure> {
        Args::parse(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_are_the_protocol() {
        let parsed = args(&[]).ok().unwrap();
        assert_eq!(parsed.sizes(), DEGREE_SIZES);
        assert_eq!(parsed.degrees, (1..=8).collect::<Vec<_>>());
        assert_eq!(parsed.families, DegreeFamily::ALL);
        assert_eq!(parsed.cells, CELLS);
        assert_eq!(parsed.node_families().len(), 2);
        assert_eq!(
            (parsed.runs, parsed.strategies.as_slice()),
            (RUNS, &["direct".to_owned()][..])
        );
        let launch = args(&["--launch-only"]).ok().unwrap();
        assert_eq!(launch.sizes(), [LAUNCH_SIZE]);
        assert_eq!(launch.node_families(), [NodeFamily::Equispaced]);
    }

    #[test]
    fn filters_are_read() {
        let parsed = args(&[
            "--target",
            "locust",
            "--run-id",
            "42",
            "--families",
            "jacobian,PULLBACK",
            "--cells",
            "hexahedron,Quadrilateral",
            "--degrees",
            "8,4",
            "--node-families",
            "gll",
            "--types",
            "f32",
            "--sizes",
            "10000000,1000000",
            "--strategies",
            "policy,direct",
            "--launch-only",
            "--vector-sizes",
            "Scalar",
        ])
        .ok()
        .unwrap();
        assert_eq!(
            (parsed.target.as_str(), parsed.run_id.as_str()),
            ("locust", "42")
        );
        assert_eq!(
            parsed.families,
            [DegreeFamily::Jacobian, DegreeFamily::PullBack]
        );
        assert_eq!(
            parsed.cells,
            [CellType::Hexahedron, CellType::Quadrilateral]
        );
        assert_eq!(parsed.degrees, [8, 4]);
        assert_eq!(parsed.node_families(), [NodeFamily::Gll]);
        assert_eq!(parsed.scalars, [Scalar::F32]);
        assert_eq!(parsed.sizes(), [1_000_000, 10_000_000]);
        assert_eq!(parsed.strategies, ["policy", "direct"]);
        assert!(parsed.launch_only);
        assert_eq!(parsed.launch_vector_sizes, Some(LaunchVectorSizes::Scalar));
    }

    #[test]
    fn bad_options_are_rejected() {
        for list in [
            &["--sizes"][..],
            &["--sizes", "0"],
            &["--families", "geomfactors"],
            &["--cells", "point"],
            &["--degrees", "0"],
            &["--degrees", "9"],
            &["--node-families", "gauss"],
            &["--types", "u32"],
            &["--strategies", "fast"],
            &["--launch-only", "--vector-sizes", "2"],
            &["--vector-sizes", "policy"],
            &["--fast"],
        ] {
            assert!(matches!(args(list), Err(Failure::Usage(_))), "{list:?}");
        }
    }
}
