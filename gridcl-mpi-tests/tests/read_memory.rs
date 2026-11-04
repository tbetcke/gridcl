//! M8 acceptance, memory (docs/plans/M8.md, T0b, T5a; ADR-0035, rule 10): reading a binary box
//! keeps the peak resident memory of every rank below twice its final mesh size plus 256 MiB.
//!
//! The file is `GRIDCL_M8_BOX` (the 10^7-cell box written by `gen-gmsh-box 119` before the
//! measured run), else a small box that rank 0 writes first (the smoke test). Peak memory is
//! `getrusage(RUSAGE_SELF).ru_maxrss`: KiB on Linux, bytes on macOS.
//!
//! On Linux the bound assumes glibc's mmap threshold fixed at 1 MiB: the measured `mpirun` sets
//! `GLIBC_TUNABLES=glibc.malloc.mmap_threshold=1048576:glibc.malloc.trim_threshold=1048576` and
//! forwards it with `-x GLIBC_TUNABLES` (ADR-0036). With glibc's dynamic threshold the peak also
//! counts memory the read has freed and glibc keeps resident.
//!
//! Every rank prints the device pools gridcl logs after each stage of the read (target
//! `gridcl::memory`) with the peak resident memory so far, the live heap and the heap's peak
//! since the previous line: the memory profile of T5a and T5b. The heap is counted by this
//! binary's global allocator, so it does not depend on what the system allocator caches.

use std::alloc::{GlobalAlloc, Layout, System};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize, Ordering};

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

/// The current resident memory by kind (`RssAnon`, `RssFile`, `RssShmem` of `/proc/self/status`;
/// empty where there is no such file, as on macOS).
fn current_rss() -> String {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kinds: Vec<String> = status
        .lines()
        .filter(|l| l.starts_with("Rss"))
        .map(|l| l.split_whitespace().collect::<Vec<_>>().join(" "))
        .collect();
    if kinds.is_empty() {
        String::new()
    } else {
        format!(", now {}", kinds.join(", "))
    }
}

/// The live heap bytes, and their peak since the last [`take_heap_peak`].
static HEAP: AtomicUsize = AtomicUsize::new(0);
static HEAP_PEAK: AtomicUsize = AtomicUsize::new(0);

/// The system allocator, counting the live bytes.
struct Counting;

impl Counting {
    fn grow(by: usize) {
        let now = HEAP.fetch_add(by, Ordering::Relaxed) + by;
        HEAP_PEAK.fetch_max(now, Ordering::Relaxed);
    }
}

// SAFETY: every method forwards to `System` under the caller's contract and only counts.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract of `GlobalAlloc::alloc`.
        let p = unsafe { System.alloc(layout) };
        if !p.is_null() {
            Counting::grow(layout.size());
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract of `GlobalAlloc::alloc_zeroed`.
        let p = unsafe { System.alloc_zeroed(layout) };
        if !p.is_null() {
            Counting::grow(layout.size());
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract of `GlobalAlloc::dealloc`.
        unsafe { System.dealloc(ptr, layout) };
        HEAP.fetch_sub(layout.size(), Ordering::Relaxed);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract of `GlobalAlloc::realloc`.
        let p = unsafe { System.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            // Old and new block coexist while the contents move.
            Counting::grow(new_size);
            HEAP.fetch_sub(layout.size(), Ordering::Relaxed);
        }
        p
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// The heap's peak since the last call; the next peak starts from the live heap.
fn take_heap_peak() -> usize {
    HEAP_PEAK.swap(HEAP.load(Ordering::Relaxed), Ordering::Relaxed)
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
                "read_memory: rank {rank}: {}, peak RSS {} MiB, heap {} MiB (peak {} MiB){}",
                record.args(),
                peak_rss() >> 20,
                HEAP.load(Ordering::Relaxed) >> 20,
                take_heap_peak() >> 20,
                current_rss()
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
            Some(path) => {
                // The measurement of bullet 2: 8 ranks, a 10^7-cell file.
                assert_eq!(
                    comm.size(),
                    8,
                    "GRIDCL_M8_BOX: the measurement runs on 8 ranks"
                );
                let cells = io::read_memory(comm, &client, &PathBuf::from(path), &peak_rss);
                assert!(
                    cells >= 10_000_000,
                    "GRIDCL_M8_BOX: {cells} cells, the measurement needs at least 10^7"
                );
            }
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
