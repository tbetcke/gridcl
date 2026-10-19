//! gridcl-conformance: compares gridcl-refcell with ndelement at the pinned commit.

extern crate blas_src;
extern crate lapack_src;
#[cfg(target_os = "linux")]
extern crate openblas_src;
