//! M8 acceptance, memory (docs/plans/M8.md, T0b, T5a; ADR-0035, rule 10): reading a binary box
//! keeps the peak resident memory of every rank below twice its final mesh size plus 256 MiB.
//!
//! The file is `GRIDCL_M8_BOX` (the 10^7-cell box written by `gen-gmsh-box 119` before the
//! measured run), else a small box that rank 0 writes first (the smoke test). Peak memory is
//! `getrusage(RUSAGE_SELF).ru_maxrss`: KiB on Linux, bytes on macOS.
//!
//! Every rank prints the device pools gridcl logs after each stage of the read (target
//! `gridcl::memory`) with the peak resident memory so far: the memory profile of T5a.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::OnceLock;

use gridcl::CommDyn;
use gridcl_testing::io_parallel::{self as io, remove_shared_dir, shared_dir};

/// The peak resident memory of this process in bytes.
fn peak_rss() -> u64 {
    use nix::sys::resource::{UsageWho, getrusage};

    let max = getrusage(UsageWho::RUSAGE_SELF)
        .expect("getrusage")
        .max_rss()
        .max(0) as u64;
    if cfg!(target_os = "macos") {
        max
    } else {
        max * 1024
    }
}

/// Prints gridcl's memory log and warnings with this rank's peak resident memory.
struct Profile {
    rank: OnceLock<usize>,
}

impl log::Log for Profile {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Warn
            || (metadata.target() == "gridcl::memory" && metadata.level() <= log::Level::Debug)
    }

    fn log(&self, record: &log::Record<'_>) {
        if self.enabled(record.metadata()) {
            let rank = self.rank.get().copied().unwrap_or(usize::MAX);
            println!(
                "read_memory: rank {rank}: {}, peak RSS {} MiB",
                record.args(),
                peak_rss() >> 20
            );
        }
    }

    fn flush(&self) {}
}

static PROFILE: Profile = Profile {
    rank: OnceLock::new(),
};

fn main() -> ExitCode {
    gridcl_mpi_tests::run("read_memory", |comm| {
        let _ = PROFILE.rank.set(comm.rank());
        log::set_logger(&PROFILE).expect("the only logger");
        log::set_max_level(log::LevelFilter::Debug);
        let client = gridcl_testing::test_client();
        match std::env::var_os("GRIDCL_M8_BOX").filter(|v| !v.is_empty()) {
            Some(path) => io::read_memory(comm, &client, &PathBuf::from(path), &peak_rss),
            None => {
                let dir = shared_dir(comm, "read-memory");
                let path = dir.join("box.msh");
                if comm.rank() == 0 {
                    io::write_box_msh(12, &path).expect("write the small box");
                }
                comm.barrier();
                io::read_memory(comm, &client, &path, &peak_rss);
                remove_shared_dir(comm, &dir);
            }
        }
    })
}
