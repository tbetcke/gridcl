//! The performance-core count of the host CPU (ADR-0017): `hw.perflevel0.physicalcpu` on
//! macOS, read once per process; `None` elsewhere and on any error.

use std::sync::OnceLock;

/// The detected performance-core count, if the OS reports one.
pub(crate) fn performance_cores() -> Option<u32> {
    static CORES: OnceLock<Option<u32>> = OnceLock::new();
    *CORES.get_or_init(read)
}

#[cfg(target_os = "macos")]
fn read() -> Option<u32> {
    let name = c"hw.perflevel0.physicalcpu";
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
    u32::try_from(value).ok().filter(|&n| n > 0)
}

#[cfg(not(target_os = "macos"))]
fn read() -> Option<u32> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On macOS the count is positive and at most the hardware threads; elsewhere there is none.
    #[test]
    fn performance_cores_match_the_os() {
        let cores = performance_cores();
        if cfg!(target_os = "macos") {
            let threads = std::thread::available_parallelism().unwrap().get();
            let p = cores.expect("macOS reports hw.perflevel0.physicalcpu");
            assert!(p >= 1 && p as usize <= threads, "{p} of {threads}");
        } else {
            assert_eq!(cores, None);
        }
    }
}
