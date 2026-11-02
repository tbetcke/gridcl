//! The performance-core count of the host CPU (ADR-0017, ADR-0034): on macOS the physical cores
//! of every `hw.perflevel` except the last, which is the efficiency level; read once per process.
//! `None` elsewhere, with fewer than two levels, and on any error.

use std::sync::OnceLock;

/// The detected performance-core count, if the OS reports one.
pub(crate) fn performance_cores() -> Option<u32> {
    static CORES: OnceLock<Option<u32>> = OnceLock::new();
    *CORES.get_or_init(read)
}

/// The performance-core count from the physical cores of each performance level, fastest level
/// first: the sum over every level but the last (ADR-0034). `None` with fewer than two levels or
/// a zero sum.
fn count(levels: &[u32]) -> Option<u32> {
    let (_, faster) = levels.split_last()?;
    if faster.is_empty() {
        return None;
    }
    faster
        .iter()
        .try_fold(0u32, |sum, &n| sum.checked_add(n))
        .filter(|&n| n > 0)
}

/// More levels than this are taken as a misread (Apple silicon has two or three).
#[cfg(target_os = "macos")]
const MAX_LEVELS: u32 = 16;

#[cfg(target_os = "macos")]
fn read() -> Option<u32> {
    let n = sysctl_u32(c"hw.nperflevels")?;
    if n > MAX_LEVELS {
        return None;
    }
    let levels = (0..n)
        .map(|i| {
            let name = std::ffi::CString::new(format!("hw.perflevel{i}.physicalcpu")).ok()?;
            sysctl_u32(&name)
        })
        .collect::<Option<Vec<u32>>>()?;
    count(&levels)
}

/// An integer sysctl, read with `sysctlbyname`.
#[cfg(target_os = "macos")]
fn sysctl_u32(name: &std::ffi::CStr) -> Option<u32> {
    let mut value: libc::c_int = 0;
    let mut len = size_of::<libc::c_int>();
    // SAFETY: `name` is a NUL-terminated C string; `value` is a live, writable c_int and `len`
    // holds its size, so sysctlbyname writes at most `len` bytes into it and stores the size
    // written in `len`. No new value is set (null pointer, size 0).
    let rc = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            (&raw mut value).cast(),
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc != 0 || len != size_of::<libc::c_int>() {
        return None;
    }
    u32::try_from(value).ok()
}

#[cfg(not(target_os = "macos"))]
fn read() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A count, where the OS reports one, is positive and at most the hardware threads; Apple
    /// silicon always reports one (Intel Macs have no `hw.perflevel0`); other systems none.
    #[test]
    fn performance_cores_match_the_os() {
        let cores = performance_cores();
        if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
            assert!(
                cores.is_some(),
                "Apple silicon reports hw.nperflevels and hw.perflevel<i>.physicalcpu"
            );
        }
        if cfg!(target_os = "macos") {
            if let Some(p) = cores {
                let threads = std::thread::available_parallelism().unwrap().get();
                assert!(p >= 1 && p as usize <= threads, "{p} of {threads}");
            }
        } else {
            assert_eq!(cores, None);
        }
    }

    /// Every level but the last: the M3 Max (12 + 4) keeps 12, the M6 (2 + 4 + 6) gets 6.
    #[test]
    fn count_sums_every_level_but_the_last() {
        assert_eq!(count(&[12, 4]), Some(12));
        assert_eq!(count(&[2, 4, 6]), Some(6));
        assert_eq!(count(&[1, 2, 3, 4]), Some(6));
    }

    /// One kind of core, no levels, an empty faster level set or an overflow give no count.
    #[test]
    fn count_needs_two_levels_and_a_positive_sum() {
        assert_eq!(count(&[]), None);
        assert_eq!(count(&[8]), None);
        assert_eq!(count(&[0, 4]), None);
        assert_eq!(count(&[u32::MAX, 1, 4]), None);
    }

    /// Cross-check with the level names where the `sysctl` tool answers (not inside Claude
    /// Code's sandbox): the last level is the efficiency level, and the count is the sum of the
    /// physical cores of the others.
    #[cfg(target_os = "macos")]
    #[test]
    fn count_matches_the_named_levels() {
        let sysctl = |name: String| {
            std::process::Command::new("sysctl")
                .args(["-n", &name])
                .output()
                .ok()
                .filter(|out| out.status.success())
                .map(|out| String::from_utf8_lossy(&out.stdout).trim().to_owned())
        };
        let Some(n) = sysctl("hw.nperflevels".into()).and_then(|v| v.parse::<u32>().ok()) else {
            eprintln!("sysctl unavailable here; name cross-check skipped");
            return;
        };
        if n < 2 {
            return;
        }
        let last = sysctl(format!("hw.perflevel{}.name", n - 1)).expect("last level name");
        assert_eq!(
            last,
            "Efficiency",
            "hw.perflevel{} is the efficiency level",
            n - 1
        );
        let faster: u32 = (0..n - 1)
            .map(|i| {
                sysctl(format!("hw.perflevel{i}.physicalcpu"))
                    .and_then(|v| v.parse::<u32>().ok())
                    .expect("physicalcpu of a level")
            })
            .sum();
        assert_eq!(performance_cores(), Some(faster));
    }
}
