//! FFI bindings to ParMETIS, with the widths of `idx_t` and `real_t` from the binding.
//!
//! The bindings exist only with the feature `mpi` (off by default); without it the crate is
//! empty, so the workspace build needs neither MPI nor ParMETIS (ADR-0029, items 4 and 5).
//!
//! With `mpi`, the build script runs bindgen on `parmetis.h` (which includes `metis.h`), found
//! through `PARMETIS_DIR` and `METIS_DIR` (`include/`, `lib/`) or the system paths, and links
//! `parmetis`. The bindings cover the `ParMETIS_*` functions, `idx_t` and `real_t`, METIS's
//! status codes (`METIS_OK`, `METIS_ERROR*`), the version constants
//! `PARMETIS_{MAJOR,MINOR,SUBMINOR}_VERSION` and `IDXTYPEWIDTH`, `REALTYPEWIDTH`.
//!
//! - `MPI_Comm` in the signatures is mpi-sys's type (`mpi::ffi`), so a pointer to a
//!   communicator's `as_raw()` passes straight to `ParMETIS_V3_PartKway`.
//! - `libparmetis` must resolve METIS itself, as a shared library does that contains METIS
//!   (scripts/hpc/build-partition-libs.sh) or depends on `libmetis` (Debian, Ubuntu).
//!
//! ParMETIS is not free software: its licence allows non-commercial use and redistribution
//! with conditions. Check the licence terms before redistributing binaries that link it.

#[cfg(feature = "mpi")]
#[allow(
    non_upper_case_globals,
    non_camel_case_types,
    non_snake_case,
    dead_code,
    clippy::all
)]
mod bindings {
    use mpi::ffi::MPI_Comm;

    include!(concat!(env!("OUT_DIR"), "/parmetis.rs"));
}

#[cfg(feature = "mpi")]
pub use bindings::*;

/// The width of [`idx_t`] in bits (32 or 64), as the library was built.
#[cfg(feature = "mpi")]
pub const IDX_BITS: u32 = idx_t::BITS;

/// The width of [`real_t`] in bits (32 or 64), as the library was built.
#[cfg(feature = "mpi")]
pub const REAL_BITS: u32 = 8 * std::mem::size_of::<real_t>() as u32;

#[cfg(feature = "mpi")]
const _: () = {
    assert!(
        IDX_BITS == 32 || IDX_BITS == 64,
        "idx_t is neither 32 nor 64 bits wide"
    );
    assert!(
        REAL_BITS == 32 || REAL_BITS == 64,
        "real_t is neither 32 nor 64 bits wide"
    );
};

#[cfg(all(test, feature = "mpi"))]
mod tests {
    use super::*;
    use mpi::ffi::MPI_Comm;
    use std::mem::size_of;

    #[test]
    fn library_links_and_is_version_4() {
        // Taking the address makes the linker resolve the symbol in libparmetis; a call needs
        // MPI (the gridcl-partition and gridcl-mpi-tests tests make one).
        let part_kway: unsafe extern "C" fn(
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut real_t,
            *mut real_t,
            *mut idx_t,
            *mut idx_t,
            *mut idx_t,
            *mut MPI_Comm,
        ) -> std::ffi::c_int = ParMETIS_V3_PartKway;
        assert_ne!(std::hint::black_box(part_kway as usize), 0);
        assert_eq!(PARMETIS_MAJOR_VERSION, 4);
    }

    #[test]
    #[allow(clippy::manual_bits, reason = "the definition the plan checks against")]
    fn bits_are_the_widths_of_the_bound_types_and_the_header() {
        assert_eq!(IDX_BITS as usize, 8 * size_of::<idx_t>());
        assert_eq!(REAL_BITS as usize, 8 * size_of::<real_t>());
        assert_eq!(i64::from(IDX_BITS), i64::from(IDXTYPEWIDTH));
        assert_eq!(i64::from(REAL_BITS), i64::from(REALTYPEWIDTH));
    }

    #[test]
    fn status_codes_are_metis_5s() {
        assert_eq!(
            (METIS_OK, METIS_ERROR_INPUT, METIS_ERROR_MEMORY, METIS_ERROR),
            (1, -2, -3, -4)
        );
    }
}
