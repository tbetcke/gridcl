//! MPI integration tests of gridcl.
//!
//! Each test is a `[[test]]` target with `harness = false` and the `mpi` feature, run on 1–4
//! ranks by `scripts/mpi-test.sh`. A test exits with a non-zero status on any rank that fails.
