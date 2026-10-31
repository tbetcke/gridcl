//! FFI bindings to PT-Scotch, with the width of `SCOTCH_Num` from the binding.
//!
//! The bindings exist only with the feature `mpi` (off by default); without it the crate is
//! empty, so the workspace build needs neither MPI nor PT-Scotch (ADR-0029, items 4 and 5).
//!
//! With `mpi`, the build script runs bindgen on `ptscotch.h`, found through `SCOTCH_DIR`
//! (`include/`, `lib/`) or the system paths, and links `ptscotch`, `scotch` and `ptscotcherr`.
//! The bindings cover the `SCOTCH_dgraph*`, `SCOTCH_strat*`, `SCOTCH_random*` and
//! `SCOTCH_context*` functions (the last for deterministic, seeded calls; ADR-0031),
//! `SCOTCH_version`, the types they use and the `SCOTCH_*` constants. The strategy flags
//! `SCOTCH_STRAT*` have type `SCOTCH_Num`; the other integer constants are C `int`.
//!
//! - `MPI_Comm` and `MPI_Datatype` in the signatures are mpi-sys's types (`mpi::ffi`), so a
//!   communicator's `as_raw()` passes straight to `SCOTCH_dgraphInit`.
//! - `FILE` is opaque: only the functions that load or save files take it.
//! - gridcl supports PT-Scotch builds without `SCOTCH_PTHREAD_MPI`, which work with MPI
//!   initialised as `Threading::Single` (ADR-0029, item 13; the example `threading_probe`).

#[cfg(feature = "mpi")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
mod bindings {
    use mpi::ffi::{MPI_Comm, MPI_Datatype};

    /// C's `FILE`, opaque.
    #[repr(C)]
    pub struct FILE {
        _opaque: [u8; 0],
    }

    include!(concat!(env!("OUT_DIR"), "/ptscotch.rs"));
}

#[cfg(feature = "mpi")]
pub use bindings::*;

/// The width of [`SCOTCH_Num`] in bits (32 or 64), as the library was built.
#[cfg(feature = "mpi")]
pub const SCOTCH_NUM_BITS: u32 = SCOTCH_Num::BITS;

#[cfg(feature = "mpi")]
const _: () = assert!(
    SCOTCH_NUM_BITS == 32 || SCOTCH_NUM_BITS == 64,
    "SCOTCH_Num is neither 32 nor 64 bits wide"
);

#[cfg(all(test, feature = "mpi"))]
mod tests {
    use super::*;
    use std::ffi::c_int;
    use std::mem::size_of;

    #[test]
    fn library_links_and_matches_the_header() {
        let (mut version, mut release, mut patch): (c_int, c_int, c_int) = (0, 0, 0);
        // SAFETY: SCOTCH_version writes one int through each pointer; all three are valid.
        unsafe { SCOTCH_version(&mut version, &mut release, &mut patch) };
        assert!(version >= 6, "PT-Scotch {version}.{release}.{patch}");
        assert_eq!(
            (version, release),
            (SCOTCH_VERSION, SCOTCH_RELEASE),
            "the library is not the version of the header"
        );
    }

    #[test]
    #[allow(clippy::manual_bits, reason = "the definition the plan checks against")]
    fn num_bits_is_the_width_of_scotch_num() {
        assert_eq!(SCOTCH_NUM_BITS as usize, 8 * size_of::<SCOTCH_Num>());
    }

    #[test]
    fn opaque_structs_hold_the_library_objects() {
        // Scotch 7.0.4 and older (Ubuntu 24.04; Debian 12's 7.0.3) has no SCOTCH_contextSizeof
        // (7.0.7 has it).
        // SAFETY: the Sizeof functions take no arguments and only return a constant.
        let (dgraph, strat) = unsafe { (SCOTCH_dgraphSizeof(), SCOTCH_stratSizeof()) };
        assert!(usize::try_from(dgraph).unwrap() <= size_of::<SCOTCH_Dgraph>());
        assert!(usize::try_from(strat).unwrap() <= size_of::<SCOTCH_Strat>());
    }
}
