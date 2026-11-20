# ADR-0068: Real multi-rank runs under MPICH in hosted CI

- **Status:** Proposed
- **Date:** 2026-11-20
- **Raised in:** D1 T16 preparation (final runs of D1; the hosted CI run of commit b21ce5e)
- **Design sections affected:** "CI matrix" (rows `mpi-mpich, mpi-openmpi`); "Testing and validation"
  (parallel invariance "under both MPICH and Open MPI"); "Implementation roadmap" (dependency table, a new
  entry E2, and D1's acceptance, which depends on it); Listing H1 (the MPI-tests bullet)

## Context

The design runs every MPI target on 1–4 ranks "under both MPICH and Open MPI" ("Testing and
validation", parallel invariance; M0, M6 and D1 acceptance), and the CI matrix row `mpi-mpich,
mpi-openmpi` names hosted Linux as the MPICH platform: "tetraloma-mpi-tests on 1–4 ranks … the MPI programs
of tetraloma-examples on 1–4 ranks". Hosted CI is the only MPICH platform: the local Mac, Locust,
Kathleen and Myriad run Open MPI (or the site's MPI).

D1 added the program `parallel_io` (parallel Gmsh reading, PVTU and VTKHDF output), which the
`mpi-mpich-hdf5` job runs on 1–4 ranks. It failed on 3 ranks with `Parse { format: "Gmsh", offset: 0,
message: "unexpected end of file" }`, and the output showed why: every process printed
"parallel I/O on 1 rank(s)". Each of the N processes of `mpirun -n N` was a singleton (its own
`MPI_COMM_WORLD` of size 1), and the singletons raced on one output file.

## Problem

1. **MPICH runs are singletons.** Ubuntu 24.04's `mpich` 4.2.0 (`4.2.0-5build3`, configured
   `--with-device=ch4:ucx`) pulls in `libpmix2t64`: its `libmpich.so.12` takes the process manager's
   PMIx interface, but its launcher `mpiexec.hydra` (Ubuntu's `mpirun` for MPICH) offers PMI-1/2 only.
   Without a PMIx server each process falls back to singleton initialisation.
2. **This predates D1.** On `main` (run 44470173506, 2026-11-19), the `mpi-mpich` job's `halo_exchange`
   example on 2 ranks printed `rank 0 of 1: 642 owned and 0 ghost nodes` twice. The example passed,
   because it adapts to the world size it sees.
3. **The MPI tests cannot tell.** `tetraloma_mpi_tests::run` runs its checks on whatever world it gets
   and prints "N ranks passed" into a log that CI does not show. `scripts/mpi-test.sh` and
   `scripts/run-examples.sh` never compare the world size with `-n`. So the MPICH half of the
   design's "under both MPICH and Open MPI" has most likely run on one rank per process since the jobs
   moved to `ubuntu-24.04`, and nothing failed. This also affects the acceptance of earlier MPI
   milestones (M6, M7 on MPICH, M8, M9), which were accepted on these CI results together with Open MPI
   runs.
4. D1's acceptance asks for "the MPI programs on 1–4 ranks under MPICH and Open MPI". That cannot be
   met until MPICH runs are real.

## Options

1. **Keep the design as written.** The MPICH rows stay green while testing nothing beyond one rank;
   `parallel_io` stays red. Rejected: the design's MPICH coverage would be fictitious.
2. **Drop MPICH from hosted CI and the design.** Open MPI stays the only flavour tested. Loses the reason
   MPICH is in the matrix ("`MPI_Comm` is a pointer in Open MPI and an integer in MPICH", Risks table;
   MAJ-10), which only a real MPICH run exercises.
3. **A rank-count guard, and an MPICH whose launcher matches its library.**
    - Guard: `scripts/mpi-test.sh` and `scripts/run-examples.sh` export `TETRALOMA_EXPECT_RANKS=N` to
      every rank (MPICH's hydra passes the environment; Open MPI gets `-x TETRALOMA_EXPECT_RANKS`), and
      `tetraloma_mpi_tests::run` and the MPI examples' `main` fail at once if `comm.size()` differs. A
      singleton launch then fails loudly on every MPI flavour and target, now and later.
    - Launch: the MPICH jobs get an MPICH whose `mpiexec` serves the library's process-manager
      interface. In order of preference, the implementing milestone takes the first that works:
      (a) MPICH built from source in CI (a pinned 4.2.x or 4.3.x release, `--with-device=ch4:ofi`,
      default PMI with hydra), installed under a cached prefix keyed by version (saved from `main`, as
      the other caches), with rsmpi's build and the parallel HDF5 pointed at it; (b) the MPICH jobs in a
      container image whose distribution MPICH is built without PMIx (for example `ubuntu:22.04`'s
      MPICH 4.0); (c) a PMIx-capable launcher for Ubuntu's MPICH (PRRTE's `prterun`), if Ubuntu 24.04
      packages one that works with `libmpich` 4.2.0.
    - Then the MPICH jobs re-run `scripts/mpi-test.sh 1 2 3 4` (and the hdf5 part) for real; failures
      that real multi-rank MPICH runs expose are fixed in that milestone (they would be library bugs that
      Open MPI hides, so they get the runs ADR-0057 asks for).
4. **Guard only.** Add the guard, and let the MPICH jobs fail until someone fixes the launch. Leaves
   every push red for an unknown time.

## Proposed decision

Option 3, as a small milestone **E2 "Real MPICH runs in hosted CI"** after D1's other work, on its own
branch: the guard first (its test: `mpirun -n 2` of a program that sees size 1 must fail, checked once by
hand under the broken Ubuntu MPICH), then the MPICH install, then the full MPICH MPI suite on 1–4 ranks
and the examples. D1 is set to implemented with this one acceptance item ("the MPI programs on 1–4 ranks
under MPICH") open; D1 merges after E2, or E2 merges first and D1's branch takes it in. Open MPI coverage
(local Mac, CI, Kathleen) is unaffected.

## Consequences

- Text to change in docs/design.md:
    - "CI matrix", row `mpi-mpich, mpi-openmpi`: MPICH is the pinned source build (or the image E2
      chooses), and every MPI run checks that the world size equals the rank count requested;
    - "Testing and validation": the same rank-count check for every MPI run, on every target;
    - "Implementation roadmap": table row "E2 Real MPICH runs in hosted CI | E1", D1's row becomes
      "D1 User documentation | E1, E2" for its MPICH acceptance item only, and a new entry E2 (scope as
      above; acceptance: the guard fails a singleton launch; `scripts/mpi-test.sh 1 2 3 4` and the hdf5
      targets pass under MPICH with the guard; the MPI programs pass on 1–4 ranks under MPICH; the
      Open MPI jobs pass unchanged);
    - Listing H1: the MPI-tests bullet adds "the scripts check the world size".
- Public API changes: none. `tetraloma-mpi-tests` and `tetraloma-examples` (unpublished) gain the check;
  scripts and `.github/workflows/ci.yml` change.
- Acceptance tests affected: none of the library's. The MPICH results of M6–M9 were most likely
  one-rank runs; E2's full MPICH run re-establishes them, and any failure it finds is fixed there.
- Later milestones affected: none beyond D1 (its MPICH acceptance waits for E2). Every later MPI run
  carries the guard.
- Development environment and test targets: docs/dev-environment.md, "Hosted CI", records the MPICH
  install and the cache.

## Decision record

<Filled by the maintainer: accepted or rejected, date, and any conditions.>
