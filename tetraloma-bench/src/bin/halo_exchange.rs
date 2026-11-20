//! The host-staged halo exchange, part by part (M10 T6; ADR-0059, ADR-0061), on the device
//! named by `GRIDCL_TEST_DEVICE`, as JSON records (`tetraloma_bench::record::HaloRecord`).
//!
//! ```text
//! cargo run --release -p tetraloma-bench --bin halo_exchange [--features mpi] -- [--target NAME]
//!     [--types f32,f64] [--halos 1000,10000,100000] [--owned N] [--block B] [--runs R]
//!     [--warmups W] [--fallback-reason TEXT] [--out FILE [--append]]
//! halo_exchange annotate --node-load RESULTS_DIR FILE...
//! halo_exchange report [--check] [--data DIR] [--doc FILE]
//! ```
//!
//! Each rank owns `--owned` nodes (default 10^6) of `--block` values (default 3) and ghosts
//! `--halos` nodes of the next rank (rank r + 1 mod p), spread evenly over its owned nodes;
//! so on two ranks each rank sends and receives one halo. Parts:
//!
//! - `pack`, `d2h`, `mpi`, `h2d`, `unpack`: the steps of `Scatterer::forward` one by one, with
//!   the map's own send and receive lists: the pack kernel, the device-to-host copy of the
//!   packed blocks (`to_host`), the host neighbourhood all-to-all-v, the host-to-device copy of
//!   the received blocks (`write_host`) and the unpack kernel into the ghost slots;
//! - `forward` and `reverse` (`ReverseOp::Add`): the whole `Scatterer` calls.
//!
//! On one rank there is nothing to exchange: `pack`, `d2h`, `h2d` and `unpack` move the same
//! number of blocks between the owned nodes and a buffer, and the other parts are not recorded.
//!
//! Every part runs `--warmups` untimed (default 5) and `--runs` timed times (default 50), each
//! after a barrier and followed by a device sync; a run's time is the largest over the ranks.
//! A forward is checked once against the owners' values before the timing. Types default to
//! f64, or f32 on a device without f64. Rank 0 writes the records. Without the feature `mpi`
//! the binary runs on one rank (`Serial`). `--target`, `--fallback-reason` and `--out` are as
//! in `vector_sweep`; the commit likewise. `--append` keeps the records already in `--out`.
//!
//! `annotate` adds the runner's node-load snapshots to every record of each file, as
//! `vector_sweep annotate` does. `report` writes the tables of docs/perf/halo-exchange.md
//! between its M10 markers from the `m10-halo-*.json` records of `--data` (default
//! docs/perf/data); `--check` fails if the file differs instead.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use cubecl::prelude::Client;
use tetraloma::{
    CommDyn, CommExt, DType, DeviceBuffer, IndexMap, Neighbourhood, ReverseOp, ScatterNumeric,
};
use tetraloma_bench::families::Machine;
use tetraloma_bench::protocol::{self, median};
use tetraloma_bench::record::{
    HALO_PARTS, HaloRecord, NodeLoad, Scalar, annotate_node_load, load_m10_dir, read_records,
    write_records,
};
use tetraloma_bench::report::{M10_BEGIN_MARKER as BEGIN_MARKER, M10_END_MARKER as END_MARKER};
use tetraloma_device::{check_precision, pack, unpack};

const USAGE: &str = "usage: halo_exchange [--target NAME] [--types f32,f64] [--halos LIST] \
    [--owned N] [--block B] [--runs R] [--warmups W] [--fallback-reason TEXT] [--out FILE [--append]]\n       \
    halo_exchange annotate --node-load RESULTS_DIR FILE...\n       \
    halo_exchange report [--check] [--data DIR] [--doc FILE]";

/// Halo sizes of bullet 1 (ADR-0059), in nodes.
const HALOS: [usize; 3] = [1_000, 10_000, 100_000];
/// Owned nodes per rank.
const OWNED: usize = 1_000_000;
/// Values per node.
const BLOCK: usize = 3;
/// Timed runs per part.
const RUNS: usize = 50;
/// Untimed runs before them.
const WARMUPS: usize = 5;

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).peekable();
    let result = match args.peek().map(String::as_str) {
        Some("annotate") => {
            args.next();
            annotate(args)
        }
        Some("report") => {
            args.next();
            report(args)
        }
        _ => Args::parse(args)
            .and_then(|args| with_world(|comm| run(comm, &args).map_err(Failure::Run))),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(Failure::Usage(message)) => {
            eprintln!("halo_exchange: {message}\n{USAGE}");
            ExitCode::from(2)
        }
        Err(Failure::Run(message)) => {
            eprintln!("halo_exchange: {message}");
            ExitCode::FAILURE
        }
    }
}

enum Failure {
    Usage(String),
    Run(String),
}

impl From<&str> for Failure {
    fn from(message: &str) -> Failure {
        Failure::Usage(message.to_owned())
    }
}

impl From<String> for Failure {
    fn from(message: String) -> Failure {
        Failure::Usage(message)
    }
}

/// Runs `f` on MPI's world communicator, with a panic on any rank aborting the job.
#[cfg(feature = "mpi")]
fn with_world(f: impl FnOnce(&dyn CommDyn) -> Result<(), Failure>) -> Result<(), Failure> {
    use mpi::traits::Communicator;

    let universe = mpi::initialize().ok_or("MPI is already initialised")?;
    let default = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        default(info);
        mpi::topology::SimpleCommunicator::world().abort(1);
    }));
    let comm = tetraloma::Mpi::world(&universe);
    let result = f(&comm);
    if let Err(Failure::Run(message)) = &result {
        // A failure on one rank would leave the others waiting in the next collective.
        eprintln!("halo_exchange: rank {}: {message}", comm.rank());
        universe.world().abort(1);
    }
    result
}

/// Runs `f` on one rank.
#[cfg(not(feature = "mpi"))]
fn with_world(f: impl FnOnce(&dyn CommDyn) -> Result<(), Failure>) -> Result<(), Failure> {
    f(&tetraloma::Serial)
}

struct Args {
    target: String,
    scalars: Option<Vec<Scalar>>,
    halos: Vec<usize>,
    owned: usize,
    block: usize,
    runs: usize,
    warmups: usize,
    fallback_reason: Option<String>,
    out: Option<PathBuf>,
    /// Keep the records already in `out`.
    append: bool,
}

fn positive(flag: &str, value: &str) -> Result<usize, String> {
    value
        .parse::<usize>()
        .ok()
        .filter(|&n| n > 0)
        .ok_or_else(|| format!("{flag}: '{value}' is not a positive integer"))
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, Failure> {
        let mut parsed = Args {
            target: std::env::var("GRIDCL_HPC_TARGET")
                .ok()
                .filter(|t| !t.is_empty())
                .unwrap_or_else(|| "local".to_owned()),
            scalars: None,
            halos: HALOS.to_vec(),
            owned: OWNED,
            block: BLOCK,
            runs: RUNS,
            warmups: WARMUPS,
            fallback_reason: None,
            out: None,
            append: false,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--target" => parsed.target = value()?,
                "--types" => {
                    parsed.scalars = Some(
                        value()?
                            .split(',')
                            .map(|t| match Scalar::named(t) {
                                Some(s @ (Scalar::F32 | Scalar::F64)) => Ok(s),
                                _ => Err(format!("--types: '{t}' is not f32 or f64")),
                            })
                            .collect::<Result<_, _>>()?,
                    )
                }
                "--halos" => {
                    parsed.halos = value()?
                        .split(',')
                        .map(|h| positive(&flag, h))
                        .collect::<Result<_, _>>()?
                }
                "--owned" => parsed.owned = positive(&flag, &value()?)?,
                "--block" => parsed.block = positive(&flag, &value()?)?,
                "--runs" => parsed.runs = positive(&flag, &value()?)?,
                "--warmups" => {
                    let v = value()?;
                    parsed.warmups = v
                        .parse()
                        .map_err(|_| format!("--warmups: '{v}' is not an integer"))?
                }
                "--fallback-reason" => parsed.fallback_reason = Some(value()?),
                "--out" => parsed.out = Some(PathBuf::from(value()?)),
                "--append" => parsed.append = true,
                _ => return Err(format!("unknown option {flag}").into()),
            }
        }
        if parsed.append && parsed.out.is_none() {
            return Err("--append needs --out".into());
        }
        if let Some(&h) = parsed.halos.iter().find(|&&h| h > parsed.owned) {
            return Err(format!("a halo of {h} nodes exceeds --owned {}", parsed.owned).into());
        }
        Ok(parsed)
    }
}

/// Measures every scalar type and halo, and writes the records on rank 0.
fn run(comm: &dyn CommDyn, args: &Args) -> Result<(), String> {
    let client = tetraloma_testing::test_client();
    let machine = Machine::new(&client);
    let rank = comm.rank();
    let scalars = args.scalars.clone().unwrap_or_else(|| {
        if check_precision::<f64>(&client).is_ok() {
            vec![Scalar::F64]
        } else {
            vec![Scalar::F32]
        }
    });
    if rank == 0 {
        eprintln!(
            "halo_exchange: {} on {} ({}), {} ranks, {} owned nodes of {} values, halos {:?}, \
             {} runs after {} warm-ups",
            machine.backend,
            machine.device,
            machine.commit,
            comm.size(),
            args.owned,
            args.block,
            args.halos,
            args.runs,
            args.warmups
        );
    }
    let mut records = match &args.out {
        Some(out) if args.append && rank == 0 && out.exists() => {
            read_records(out, |v| Ok::<_, String>(v.clone()))?
        }
        _ => Vec::new(),
    };
    for scalar in scalars {
        for &halo in &args.halos {
            let parts = match scalar {
                Scalar::F64 => {
                    check_precision::<f64>(&client).map_err(|e| format!("f64: {e}"))?;
                    measure_halo::<f64>(comm, &client, args, halo, f64::from)?
                }
                _ => measure_halo::<f32>(comm, &client, args, halo, |v| v as f32)?,
            };
            for (part, times_s) in parts {
                let record = HaloRecord {
                    backend: machine.backend.clone(),
                    device: machine.device.clone(),
                    device_class: machine.class,
                    target: args.target.clone(),
                    scalar,
                    ranks: comm.size(),
                    owned_nodes: args.owned,
                    halo_nodes: halo,
                    block: args.block,
                    part: part.to_owned(),
                    warmups: args.warmups,
                    median_s: median(&times_s),
                    times_s,
                    commit: machine.commit.clone(),
                    profile: protocol::profile().to_owned(),
                    cores: machine.cores.clone(),
                    load_width: Some(machine.load_width),
                    fallback_reason: args.fallback_reason.clone(),
                    node_load: None,
                };
                if rank == 0 {
                    eprintln!(
                        "halo_exchange: {} halo {halo:>6} {:<8} {:>10.1} µs",
                        scalar.name(),
                        record.part,
                        record.median_s * 1e6
                    );
                }
                records.push(record.to_json());
                if rank == 0
                    && let Some(out) = &args.out
                {
                    write_records(out, &records)?;
                }
            }
        }
    }
    if rank == 0 && args.out.is_none() {
        println!(
            "{}",
            serde_json::to_string_pretty(&records).map_err(|e| e.to_string())?
        );
    }
    Ok(())
}

/// The value of component `c` of global node `g`: exact in f32.
fn value(g: usize, c: usize, block: usize) -> u32 {
    ((g * block + c) % (1 << 20)) as u32
}

/// The times of each part for one halo size, in the order of [`HALO_PARTS`] (on one rank
/// only `pack`, `d2h`, `h2d` and `unpack`).
fn measure_halo<T: ScatterNumeric + PartialEq + std::fmt::Debug>(
    comm: &dyn CommDyn,
    client: &Client,
    args: &Args,
    halo: usize,
    to_t: impl Fn(u32) -> T,
) -> Result<Vec<(&'static str, Vec<f64>)>, String> {
    let (rank, size) = (comm.rank(), comm.size());
    let (n, block) = (args.owned, args.block);
    let dev = |e: tetraloma::GridError| e.to_string();
    // Ghost k is node k · stride of the next rank, which owns [owner · n, (owner + 1) · n).
    let stride = n / halo;
    let map = if size > 1 {
        let owner = (rank + 1) % size;
        let ghosts = (0..halo).map(|k| (owner * n + k * stride) as i64).collect();
        IndexMap::with_ghosts(comm, n, ghosts, vec![owner as i32; halo]).map_err(dev)?
    } else {
        IndexMap::new(comm, n).map_err(dev)?
    };
    let begin = map.owned_range().0 as usize;
    let global = |i: usize| -> usize {
        if i < n {
            begin + i
        } else {
            map.ghosts()[i - n] as usize
        }
    };
    let local = map.num_local();
    let host: Vec<T> = (0..local * block)
        .map(|i| {
            let node = i / block;
            if node < n {
                to_t(value(global(node), i % block, block))
            } else {
                to_t(0)
            }
        })
        .collect();
    let mut x = DeviceBuffer::from_host(client, &host).map_err(dev)?;

    // Forward's lists, or on one rank the same number of owned blocks both ways.
    let (send_ranks, send_counts, send_idx, recv_ranks, recv_counts, recv_idx) = if size > 1 {
        let (dst, sc, si) = map.send_lists();
        let (src, rc, ri) = map.recv_lists();
        (
            dst.to_vec(),
            sc.to_vec(),
            si.to_vec(),
            src.to_vec(),
            rc.to_vec(),
            ri.to_vec(),
        )
    } else {
        let spread: Vec<u32> = (0..halo).map(|k| (k * stride) as u32).collect();
        (vec![], vec![], spread.clone(), vec![], vec![], spread)
    };
    let send_idx_d = DeviceBuffer::from_host(client, &send_idx).map_err(dev)?;
    let recv_idx_d = DeviceBuffer::from_host(client, &recv_idx).map_err(dev)?;
    let mut send = DeviceBuffer::<T>::uninit(client, send_idx.len() * block).map_err(dev)?;
    let mut recv = DeviceBuffer::<T>::uninit(client, recv_idx.len() * block).map_err(dev)?;
    let mut host_recv = vec![to_t(0); recv_idx.len() * block];
    let nb = Neighbourhood::new(comm, recv_ranks, send_ranks).map_err(dev)?;
    let words = std::mem::size_of::<T>() / 4 * block;
    let send_words: Vec<usize> = send_counts.iter().map(|c| c * words).collect();
    let recv_words: Vec<usize> = recv_counts.iter().map(|c| c * words).collect();
    let scatterer = map.scatterer(client, block).map_err(dev)?;

    // The forward gives each ghost its owner's values (bitwise results: the scatter tests).
    if size > 1 {
        scatterer.forward(&mut x).map_err(dev)?;
        let got = x.to_host().map_err(dev)?;
        for (i, got) in got.iter().enumerate().skip(n * block) {
            let want = to_t(value(global(i / block), i % block, block));
            if *got != want {
                return Err(format!(
                    "rank {rank}: forward gave {got:?} at {i}, expected {want:?}"
                ));
            }
        }
    }

    let sync = || protocol::sync(client);
    // One part: warm-ups, then timed runs, each after a barrier and followed by a device sync;
    // a run's time is the largest over the ranks.
    let time = |part: &mut dyn FnMut() -> Result<(), String>| -> Result<Vec<f64>, String> {
        let mut times = Vec::with_capacity(args.runs);
        for i in 0..args.warmups + args.runs {
            comm.barrier();
            let start = Instant::now();
            part()?;
            sync()?;
            let t = comm.max(start.elapsed().as_secs_f64());
            if i >= args.warmups {
                times.push(t);
            }
        }
        Ok(times)
    };

    let mut out = Vec::new();
    out.push((
        "pack",
        time(&mut || pack(&x, &send_idx_d, block, &mut send).map_err(dev))?,
    ));
    let packed = send.to_host().map_err(dev)?;
    out.push(("d2h", time(&mut || send.to_host().map(drop).map_err(dev))?));
    if size > 1 {
        out.push((
            "mpi",
            time(&mut || {
                comm.neighbour_all_to_all_v_bytes(
                    &nb,
                    cast(&packed),
                    &send_words,
                    cast_mut(&mut host_recv),
                    &recv_words,
                    DType::U32,
                )
                .map_err(dev)
            })?,
        ));
    } else {
        host_recv.copy_from_slice(&packed);
    }
    out.push((
        "h2d",
        time(&mut || recv.write_host(&host_recv).map_err(dev))?,
    ));
    out.push((
        "unpack",
        time(&mut || unpack(&recv, &recv_idx_d, block, &mut x).map_err(dev))?,
    ));
    if size > 1 {
        out.push((
            "forward",
            time(&mut || scatterer.forward(&mut x).map_err(dev))?,
        ));
        out.push((
            "reverse",
            time(&mut || scatterer.reverse(&mut x, ReverseOp::Add).map_err(dev))?,
        ));
    }
    debug_assert!(out.iter().all(|(p, _)| HALO_PARTS.contains(p)));
    Ok(out)
}

/// The bytes of `values` (f32 or f64).
fn cast<T: ScatterNumeric>(values: &[T]) -> &[u8] {
    bytemuck::cast_slice(values)
}

fn cast_mut<T: ScatterNumeric>(values: &mut [T]) -> &mut [u8] {
    bytemuck::cast_slice_mut(values)
}

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
        "halo_exchange: node load before: {}; after: {}",
        load.before, load.after
    );
    for file in files {
        annotate_node_load(&file, &load).map_err(Failure::Run)?;
    }
    Ok(())
}

fn report(mut args: impl Iterator<Item = String>) -> Result<(), Failure> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut data = root.join("docs/perf/data");
    let mut doc = root.join("docs/perf/halo-exchange.md");
    let mut check = false;
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--check" => check = true,
            "--data" => data = PathBuf::from(args.next().ok_or("--data needs a value")?),
            "--doc" => doc = PathBuf::from(args.next().ok_or("--doc needs a value")?),
            _ => return Err(format!("unknown option {arg}").into()),
        }
    }
    let records = load_m10_dir(&data).map_err(Failure::Run)?.halo;
    let tables = tetraloma_bench::report::render_m10_halo(&records);
    let text = std::fs::read_to_string(&doc)
        .map_err(|e| Failure::Run(format!("{}: {e}", doc.display())))?;
    let (Some(begin), Some(end)) = (text.find(BEGIN_MARKER), text.find(END_MARKER)) else {
        return Err(Failure::Run(format!(
            "{}: no {BEGIN_MARKER} … {END_MARKER}",
            doc.display()
        )));
    };
    let new = format!(
        "{}{BEGIN_MARKER}\n\n{tables}\n{}",
        &text[..begin],
        &text[end..]
    );
    if check {
        if new != text {
            return Err(Failure::Run(format!(
                "{} is out of date: run halo_exchange report",
                doc.display()
            )));
        }
        eprintln!("halo_exchange: {} is up to date", doc.display());
    } else {
        std::fs::write(&doc, new).map_err(|e| Failure::Run(format!("{}: {e}", doc.display())))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, Failure> {
        Args::parse(list.iter().map(|s| s.to_string()))
    }

    /// The defaults are bullet 1's protocol: halos of 10^3, 10^4 and 10^5 nodes, block 3,
    /// 10^6 owned nodes, 50 runs after 5 warm-ups.
    #[test]
    fn defaults_are_the_protocol() {
        let a = args(&[]).ok().unwrap();
        assert_eq!(a.halos, HALOS);
        assert_eq!((a.owned, a.block, a.runs, a.warmups), (OWNED, 3, 50, 5));
        assert!(a.scalars.is_none());
    }

    #[test]
    fn bad_options_are_usage_errors() {
        for bad in [
            &["--types", "u32"][..],
            &["--halos", "0"],
            &["--owned", "10", "--halos", "100"],
            &["--frobnicate"],
        ] {
            assert!(matches!(args(bad), Err(Failure::Usage(_))), "{bad:?}");
        }
    }

    /// One rank: the four device parts, forward's checks skipped; tiny sizes on the test
    /// device.
    #[test]
    fn one_rank_measures_the_device_parts() {
        let a = args(&[
            "--owned",
            "1000",
            "--halos",
            "10",
            "--runs",
            "3",
            "--warmups",
            "1",
        ])
        .ok()
        .unwrap();
        let client = tetraloma_testing::test_client();
        let parts = measure_halo::<f32>(&tetraloma::Serial, &client, &a, 10, |v| v as f32).unwrap();
        let names: Vec<_> = parts.iter().map(|(p, _)| *p).collect();
        assert_eq!(names, ["pack", "d2h", "h2d", "unpack"]);
        assert!(
            parts
                .iter()
                .all(|(_, t)| t.len() == 3 && t.iter().all(|&t| t > 0.0))
        );
    }
}
