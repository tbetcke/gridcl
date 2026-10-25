//! The vector-size sweep of "Benchmarks" (docs/plans/M3s.md, T8; batched runs, ADR-0018) on the
//! device named by `GRIDCL_TEST_DEVICE`, as JSON records (`gridcl_bench::record::SweepRecord`).
//!
//! ```text
//! cargo run --release -p gridcl-bench --bin vector_sweep -- [--target NAME]
//!     [--families fill,copy,...] [--types f32,f64,u32] [--sizes 100000,...]
//!     [--vector-sizes 1,2,...] [--blocks 4,16,64] [--cube-dim auto|plan|perf|both]
//!     [--runs R] [--fallback-reason TEXT] [--out FILE]
//! vector_sweep annotate --node-load RESULTS_DIR FILE...
//! ```
//!
//! - `--target`: the machine, `m3max` or a target of scripts/hpc/remote-test.sh; default
//!   `GRIDCL_HPC_TARGET` (set in runner jobs), else `local`.
//! - `--families`, `--types`, `--sizes`, `--vector-sizes`: filters, so a busy part can be
//!   repeated; by default every family, type, size (10^5, 10^6, 10^7) and vector size the device
//!   offers. Family names are the policy keys, in any case.
//! - `--blocks`: CPU block targets in KiB (default 4, 16, 64); ignored on GPUs.
//! - `--cube-dim`: `plan` (one unit per hardware thread), `perf` (the performance-core count
//!   from `sysctl`, through `with_cpu_units`), `both`, or `auto` (default: `both` where the
//!   performance-core count is known on a CPU, else `plan`).
//! - `--out`: rewritten after every record, so an interrupted sweep keeps what it measured;
//!   default stdout at the end.
//!
//! `annotate` adds the runner's node-load snapshots (`node-load-before.txt`,
//! `node-load-after.txt` in `results/<job id>/`) to every record of each file, because the
//! "after" snapshot is taken once the command has finished (ADR-0002, ADR-0005).
//!
//! The commit is `GRIDCL_BENCH_COMMIT`, else `GITHUB_SHA`, else `git rev-parse HEAD` (with
//! `-dirty` when the tree has changes), else `unknown`.

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use gridcl_bench::families::{self, BLOCKS, Machine, SIZES, SweepOptions};
use gridcl_bench::protocol::RUNS;
use gridcl_bench::record::{CubeDimMode, NodeLoad, Scalar, kind_named, write_records};
use gridcl_device::KernelKind;
use serde_json::Value;

const USAGE: &str = "usage: vector_sweep [--target NAME] [--families LIST] [--types LIST] \
    [--sizes LIST] [--vector-sizes LIST] [--blocks KIB,...] [--cube-dim auto|plan|perf|both] \
    [--runs R] [--fallback-reason TEXT] [--out FILE]\n       \
    vector_sweep annotate --node-load RESULTS_DIR FILE...";

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
            eprintln!("vector_sweep: {message}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Run(message)) => {
            eprintln!("vector_sweep: {message}");
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

/// `--cube-dim`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CubeDims {
    Auto,
    Plan,
    Perf,
    Both,
}

/// Command-line options of the sweep.
#[derive(Debug)]
struct Args {
    target: String,
    families: Vec<KernelKind>,
    scalars: Option<Vec<Scalar>>,
    sizes: Vec<usize>,
    vector_sizes: Option<Vec<usize>>,
    blocks: Vec<usize>,
    cube_dims: CubeDims,
    runs: usize,
    fallback_reason: Option<String>,
    out: Option<PathBuf>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, Failure> {
        let mut parsed = Args {
            target: std::env::var("GRIDCL_HPC_TARGET")
                .ok()
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "local".to_owned()),
            families: families::families().into_iter().map(|(k, _)| k).collect(),
            scalars: None,
            sizes: SIZES.to_vec(),
            vector_sizes: None,
            blocks: BLOCKS.to_vec(),
            cube_dims: CubeDims::Auto,
            runs: RUNS,
            fallback_reason: None,
            out: None,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--target" => parsed.target = value()?,
                "--families" => {
                    let measured: Vec<KernelKind> =
                        families::families().into_iter().map(|(k, _)| k).collect();
                    parsed.families = list(&flag, &value()?, |name| {
                        kind_named(name).filter(|k| measured.contains(k))
                    })?;
                }
                "--types" => parsed.scalars = Some(list(&flag, &value()?, Scalar::named)?),
                "--sizes" => parsed.sizes = list(&flag, &value()?, positive)?,
                "--vector-sizes" => {
                    parsed.vector_sizes = Some(list(&flag, &value()?, |v| {
                        positive(v).filter(|n| n.is_power_of_two())
                    })?)
                }
                "--blocks" => {
                    parsed.blocks = list(&flag, &value()?, |v| {
                        positive(v)
                            .filter(|k| k.is_power_of_two() && (4..=64).contains(k))
                            .map(|k| k << 10)
                    })?
                }
                "--cube-dim" => {
                    parsed.cube_dims = match value()?.as_str() {
                        "auto" => CubeDims::Auto,
                        "plan" => CubeDims::Plan,
                        "perf" => CubeDims::Perf,
                        "both" => CubeDims::Both,
                        other => return Err(format!("--cube-dim {other}").into()),
                    }
                }
                "--runs" => {
                    parsed.runs = positive(&value()?).ok_or("--runs needs a positive integer")?
                }
                "--fallback-reason" => parsed.fallback_reason = Some(value()?),
                "--out" => parsed.out = Some(PathBuf::from(value()?)),
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        Ok(parsed)
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
        eprintln!("vector_sweep: debug build; record only release builds (--release)");
    }
    let client = gridcl_testing::test_client();
    let machine = Machine::new(&client);
    let cube_dims = match args.cube_dims {
        CubeDims::Plan => vec![CubeDimMode::Plan],
        CubeDims::Perf => vec![CubeDimMode::PerformanceCores],
        CubeDims::Both => vec![CubeDimMode::Plan, CubeDimMode::PerformanceCores],
        CubeDims::Auto if machine.cores.performance.is_some() => {
            vec![CubeDimMode::Plan, CubeDimMode::PerformanceCores]
        }
        CubeDims::Auto => vec![CubeDimMode::Plan],
    };
    eprintln!(
        "vector_sweep: {} on {} ({:?}, target {}, {}), cores {:?}, kernel cache {}",
        machine.backend,
        machine.device,
        machine.class,
        args.target,
        machine.commit,
        machine.cores,
        machine.kernel_cache
    );
    let options = SweepOptions {
        target: args.target,
        families: args.families,
        scalars: args.scalars,
        sizes: args.sizes,
        vector_sizes: args.vector_sizes,
        blocks: args.blocks,
        cube_dims,
        runs: args.runs,
        fallback_reason: args.fallback_reason,
    };

    let mut written: Vec<Value> = Vec::new();
    let mut write_error = None;
    let out = args.out.as_deref();
    let records = families::sweep(&client, &machine, &options, &mut |r| {
        eprintln!(
            "{:?} {} n = {} v = {} (ran {}) block = {:?} {:?}: median {:.3} ms ({} calls a run), \
             compile {:.3} s",
            r.family,
            r.scalar.name(),
            r.n,
            r.vector_size,
            r.vector_size_ran,
            r.block_bytes.map(|b| b >> 10),
            r.cube_dim,
            r.median_s * 1e3,
            r.calls_per_run,
            r.compile_s
        );
        written.push(r.to_json());
        if let Some(path) = out
            && write_error.is_none()
        {
            write_error = write_records(path, &written).err();
        }
    })
    .map_err(Failure::Run)?;
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
        "vector_sweep: node load before: {}; after: {}",
        load.before, load.after
    );
    for file in files {
        annotate_file(&file, &load).map_err(Failure::Run)?;
    }
    Ok(())
}

fn annotate_file(path: &Path, load: &NodeLoad) -> Result<(), String> {
    let what = path.display();
    let text = std::fs::read_to_string(path).map_err(|e| format!("{what}: {e}"))?;
    let mut value: Value = serde_json::from_str(&text).map_err(|e| format!("{what}: {e}"))?;
    let records = value
        .as_array_mut()
        .ok_or_else(|| format!("{what}: not a JSON array"))?;
    for record in records.iter_mut() {
        let object = record
            .as_object_mut()
            .ok_or_else(|| format!("{what}: a record is not a JSON object"))?;
        object.insert(
            "node_load".to_owned(),
            serde_json::json!({ "before": load.before, "after": load.after }),
        );
    }
    write_records(path, records)
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
        assert_eq!(parsed.sizes, SIZES);
        assert_eq!(parsed.blocks, BLOCKS);
        assert_eq!((parsed.runs, parsed.cube_dims), (RUNS, CubeDims::Auto));
        assert!(parsed.scalars.is_none() && parsed.vector_sizes.is_none());
    }

    #[test]
    fn filters_are_read() {
        let parsed = args(&[
            "--target",
            "locust",
            "--families",
            "fill,REDUCE",
            "--types",
            "u32",
            "--sizes",
            "100000",
            "--vector-sizes",
            "1,4",
            "--blocks",
            "4,64",
            "--cube-dim",
            "plan",
        ])
        .ok()
        .unwrap();
        assert_eq!(parsed.target, "locust");
        assert_eq!(parsed.families, [KernelKind::Fill, KernelKind::Reduce]);
        assert_eq!(parsed.scalars, Some(vec![Scalar::U32]));
        assert_eq!(parsed.sizes, [100_000]);
        assert_eq!(parsed.vector_sizes, Some(vec![1, 4]));
        assert_eq!(parsed.blocks, [4 << 10, 64 << 10]);
        assert_eq!(parsed.cube_dims, CubeDims::Plan);
    }

    #[test]
    fn bad_options_are_rejected() {
        for list in [
            &["--sizes"][..],
            &["--sizes", "0"],
            &["--families", "pack"],
            &["--families", "nope"],
            &["--types", "f16"],
            &["--vector-sizes", "3"],
            &["--blocks", "8,128"],
            &["--blocks", "2"],
            &["--cube-dim", "all"],
            &["--fast"],
        ] {
            assert!(matches!(args(list), Err(Failure::Usage(_))), "{list:?}");
        }
    }
}
