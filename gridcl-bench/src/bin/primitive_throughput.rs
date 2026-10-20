//! Throughput of sorting 10^7 two-word keys and of scanning 10^7 u32 values on the device named
//! by `GRIDCL_TEST_DEVICE`, as JSON (M2 T11, docs/perf/m2-primitives.md).
//!
//! Each operation runs once as a warm-up (JIT compilation), and its result is checked against a
//! host reference before any time is reported. Then ten timed runs follow, each ending with
//! `client.sync()`; the median is reported.
//!
//! ```text
//! cargo run --release -p gridcl-bench --bin primitive_throughput -- [--n N] [--runs R] [--out FILE]
//! ```
//!
//! The JSON is an array with one record per operation: backend, device, operation, n, words,
//! median in seconds, items per second, commit, and the build profile, the vector-size override
//! and every timed run. The commit is `GRIDCL_BENCH_COMMIT`, else `GITHUB_SHA`, else
//! `git rev-parse HEAD` (with `-dirty` when the tree has changes), else `unknown`.

use std::process::{Command, ExitCode};
use std::time::Instant;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, SortKeys, exclusive_scan, inclusive_scan, radix_sort};
use serde_json::{Value, json};

/// Seed of the random inputs.
const SEED: u64 = 0x6772_6964_636c_0002;
/// Words per sort key.
const SORT_WORDS: usize = 2;
/// Scan inputs are below this bound, so the sum of 10^7 values does not wrap.
const SCAN_BOUND: u64 = 256;

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(message) => {
            eprintln!("primitive_throughput: {message}");
            eprintln!("usage: primitive_throughput [--n N] [--runs R] [--out FILE]");
            return ExitCode::from(2);
        }
    };
    if cfg!(debug_assertions) {
        eprintln!("primitive_throughput: debug build; record only release builds (--release)");
    }
    let client = gridcl_testing::test_client();
    let context = Context::new(&client);
    eprintln!(
        "primitive_throughput: {} on {} ({}), n = {}, {} runs",
        context.backend, context.device, context.commit, args.n, args.runs
    );

    let mut records = Vec::new();
    let benches: [(&str, Bench); 3] = [
        ("radix_sort", bench_sort),
        ("inclusive_scan", |c, a| bench_scan(c, a, Scan::Inclusive)),
        ("exclusive_scan", |c, a| bench_scan(c, a, Scan::Exclusive)),
    ];
    for (operation, bench) in benches {
        match bench(&client, &args) {
            Ok(measured) => {
                let record = context.record(operation, &args, &measured);
                eprintln!(
                    "{operation}: median {:.3} ms, {:.3e} items/s",
                    measured.median() * 1e3,
                    args.n as f64 / measured.median()
                );
                records.push(record);
            }
            Err(message) => {
                eprintln!("primitive_throughput: {operation}: {message}");
                return ExitCode::FAILURE;
            }
        }
    }

    let text = serde_json::to_string_pretty(&Value::Array(records)).expect("JSON of records");
    match &args.out {
        Some(path) => {
            if let Err(error) = std::fs::write(path, format!("{text}\n")) {
                eprintln!("primitive_throughput: cannot write {path}: {error}");
                return ExitCode::FAILURE;
            }
        }
        None => println!("{text}"),
    }
    ExitCode::SUCCESS
}

/// One operation's benchmark: warm-up, check, timed runs.
type Bench = fn(&Client, &Args) -> Result<Measured, String>;

/// Command-line options.
struct Args {
    n: usize,
    runs: usize,
    out: Option<String>,
}

impl Args {
    fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
        let mut parsed = Args {
            n: 10_000_000,
            runs: 10,
            out: None,
        };
        while let Some(flag) = args.next() {
            let mut value = || args.next().ok_or_else(|| format!("{flag} needs a value"));
            match flag.as_str() {
                "--n" => parsed.n = positive(&flag, &value()?)?,
                "--runs" => parsed.runs = positive(&flag, &value()?)?,
                "--out" => parsed.out = Some(value()?),
                _ => return Err(format!("unknown option {flag}")),
            }
        }
        Ok(parsed)
    }
}

fn positive(flag: &str, value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(v) if v > 0 => Ok(v),
        _ => Err(format!("{flag} needs a positive integer, not '{value}'")),
    }
}

/// What every record shares.
struct Context {
    backend: &'static str,
    device: String,
    commit: String,
}

impl Context {
    fn new(client: &Client) -> Context {
        Context {
            backend: client.name(),
            device: client.properties().identity.name.clone(),
            commit: commit(),
        }
    }

    fn record(&self, operation: &str, args: &Args, measured: &Measured) -> Value {
        let median = measured.median();
        json!({
            "backend": self.backend,
            "device": self.device,
            "operation": operation,
            "n": args.n,
            "words": measured.words,
            "median_s": median,
            "items_per_s": args.n as f64 / median,
            "commit": self.commit,
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "vector_size_override": std::env::var("GRIDCL_VECTOR_SIZE").ok(),
            "times_s": measured.times,
        })
    }
}

/// The commit the binary was run from (see the crate docs).
fn commit() -> String {
    for var in ["GRIDCL_BENCH_COMMIT", "GITHUB_SHA"] {
        if let Ok(value) = std::env::var(var)
            && !value.is_empty()
        {
            return value;
        }
    }
    let git = |args: &[&str]| {
        Command::new("git")
            .args(args)
            .output()
            .ok()
            .filter(|out| out.status.success())
            .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
    };
    match git(&["rev-parse", "HEAD"]) {
        Some(head) => match git(&["status", "--porcelain", "--untracked-files=no"]) {
            Some(status) if status.is_empty() => head,
            _ => format!("{head}-dirty"),
        },
        None => "unknown".to_owned(),
    }
}

/// The timed runs of one operation.
struct Measured {
    words: usize,
    times: Vec<f64>,
}

impl Measured {
    fn median(&self) -> f64 {
        let mut sorted = self.times.clone();
        sorted.sort_by(f64::total_cmp);
        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 1 {
            sorted[mid]
        } else {
            0.5 * (sorted[mid - 1] + sorted[mid])
        }
    }
}

/// `runs` timings of `op`, each ending with `client.sync()`.
fn time_runs<R>(
    client: &Client,
    runs: usize,
    mut op: impl FnMut() -> gridcl_device::Result<R>,
) -> Result<Vec<f64>, String> {
    let mut times = Vec::with_capacity(runs);
    for _ in 0..runs {
        let start = Instant::now();
        let result = op().map_err(|e| e.to_string())?;
        sync(client)?;
        times.push(start.elapsed().as_secs_f64());
        drop(result);
    }
    Ok(times)
}

fn sync(client: &Client) -> Result<(), String> {
    cubecl::future::block_on(client.sync()).map_err(|e| format!("sync: {e:?}"))
}

/// SplitMix64, a small deterministic generator.
struct SplitMix(u64);

impl SplitMix {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

/// Random two-word keys with an iota payload: warm-up, check, timed runs.
fn bench_sort(client: &Client, args: &Args) -> Result<Measured, String> {
    let n = args.n;
    let mut rng = SplitMix(SEED);
    let keys_host: Vec<u32> = (0..n * SORT_WORDS).map(|_| rng.next_u32()).collect();
    let payload_host: Vec<u32> = (0..n)
        .map(|i| u32::try_from(i).map_err(|_| format!("n = {n} exceeds u32 payloads")))
        .collect::<Result<_, _>>()?;
    let keys = SortKeys::from_host(client, &keys_host, SORT_WORDS).map_err(|e| e.to_string())?;
    let payload = DeviceBuffer::from_host(client, &payload_host).map_err(|e| e.to_string())?;

    let (sorted, perm) = radix_sort(&keys, &payload).map_err(|e| e.to_string())?;
    let sorted = sorted.to_host().map_err(|e| e.to_string())?;
    let perm = perm.to_host().map_err(|e| e.to_string())?;
    check_sort(&keys_host, &sorted, &perm)?;

    let times = time_runs(client, args.runs, || radix_sort(&keys, &payload))?;
    Ok(Measured {
        words: SORT_WORDS,
        times,
    })
}

/// Whether `sorted` is `keys` in ascending order and `perm` the stable permutation that sorts them.
fn check_sort(keys: &[u32], sorted: &[u32], perm: &[u32]) -> Result<(), String> {
    let w = SORT_WORDS;
    let n = perm.len();
    if sorted.len() != keys.len() || keys.len() != n * w {
        return Err(format!(
            "sort: {} sorted words and {} payload entries for {} keys",
            sorted.len(),
            n,
            keys.len() / w
        ));
    }
    let key = |data: &[u32], i: usize| -> Vec<u32> { data[i * w..(i + 1) * w].to_vec() };
    let mut seen = vec![false; n];
    for (i, &p) in perm.iter().enumerate() {
        let p = p as usize;
        if p >= n || std::mem::replace(&mut seen[p], true) {
            return Err(format!("sort: payload is not a permutation at {i}"));
        }
        if key(keys, p) != key(sorted, i) {
            return Err(format!("sort: key {i} is not the key of payload {p}"));
        }
        if i > 0 {
            let (prev, this) = (key(sorted, i - 1), key(sorted, i));
            if prev > this || (prev == this && perm[i - 1] as usize > p) {
                return Err(format!("sort: keys {} and {i} are out of order", i - 1));
            }
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Scan {
    Inclusive,
    Exclusive,
}

/// Random u32 values below `SCAN_BOUND`: warm-up, check, timed runs.
fn bench_scan(client: &Client, args: &Args, scan: Scan) -> Result<Measured, String> {
    let n = args.n;
    let mut rng = SplitMix(SEED ^ 1);
    let x_host: Vec<u32> = (0..n)
        .map(|_| (rng.next_u64() % SCAN_BOUND) as u32)
        .collect();
    let x = DeviceBuffer::from_host(client, &x_host).map_err(|e| e.to_string())?;
    let run = || match scan {
        Scan::Inclusive => inclusive_scan(&x),
        Scan::Exclusive => exclusive_scan(&x),
    };

    let got = run().and_then(|y| y.to_host()).map_err(|e| e.to_string())?;
    // The exclusive scan has n + 1 entries, the last being the total.
    let mut want = Vec::with_capacity(n + 1);
    if let Scan::Exclusive = scan {
        want.push(0);
    }
    let mut sum = 0u32;
    for &v in &x_host {
        sum = sum.wrapping_add(v);
        want.push(sum);
    }
    if got != want {
        let first = got.iter().zip(&want).position(|(g, w)| g != w);
        return Err(format!(
            "scan: {} entries, {} expected; first difference at {first:?}",
            got.len(),
            want.len()
        ));
    }

    let times = time_runs(client, args.runs, run)?;
    Ok(Measured { words: 1, times })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(list: &[&str]) -> Result<Args, String> {
        Args::parse(list.iter().map(|s| s.to_string()))
    }

    #[test]
    fn defaults_are_ten_million_items_and_ten_runs() {
        let parsed = args(&[]).unwrap();
        assert_eq!((parsed.n, parsed.runs, parsed.out), (10_000_000, 10, None));
        let parsed = args(&["--n", "17", "--runs", "3", "--out", "x.json"]).unwrap();
        assert_eq!((parsed.n, parsed.runs), (17, 3));
        assert_eq!(parsed.out.as_deref(), Some("x.json"));
    }

    #[test]
    fn bad_options_are_rejected() {
        for list in [&["--n"][..], &["--n", "0"], &["--runs", "x"], &["--fast"]] {
            assert!(args(list).is_err(), "{list:?}");
        }
    }

    #[test]
    fn median_of_odd_and_even_counts() {
        let measured = |times: &[f64]| Measured {
            words: 1,
            times: times.to_vec(),
        };
        assert_eq!(measured(&[3.0, 1.0, 2.0]).median(), 2.0);
        assert_eq!(measured(&[4.0, 1.0, 3.0, 2.0]).median(), 2.5);
    }

    #[test]
    fn check_sort_accepts_only_the_stable_sort() {
        // Keys (1, 5), (0, 9), (1, 2), (0, 9).
        let keys = [1, 5, 0, 9, 1, 2, 0, 9];
        let sorted = [0, 9, 0, 9, 1, 2, 1, 5];
        assert!(check_sort(&keys, &sorted, &[1, 3, 2, 0]).is_ok());
        // Not stable: 3 before 1.
        assert!(check_sort(&keys, &sorted, &[3, 1, 2, 0]).is_err());
        // Not a permutation.
        assert!(check_sort(&keys, &sorted, &[1, 1, 2, 0]).is_err());
        // Keys out of order.
        assert!(check_sort(&keys, &[0, 9, 0, 9, 1, 5, 1, 2], &[1, 3, 0, 2]).is_err());
        // Key does not belong to its payload.
        assert!(check_sort(&keys, &[0, 9, 0, 9, 1, 2, 1, 6], &[1, 3, 2, 0]).is_err());
    }
}
