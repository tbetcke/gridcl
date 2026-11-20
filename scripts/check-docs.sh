#!/usr/bin/env bash
# Rustdoc of every workspace crate builds without warnings, and every public item is documented
# (docs/design.md, Roadmap M10 bullet 7; ADR-0060 item 7).
#
# Usage: scripts/check-docs.sh [default|libraries]
#   default    cargo doc at default features (CI: cpu job)
#   libraries  with tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis (CI: the
#              ptscotch job; needs MPI, a parallel HDF5, Scotch and PT-Scotch), and
#              tetraloma-testing/mpi: with tetraloma/mpi, tetraloma-testing's ThreadComm builds only with
#              it (CommDyn::raw)
#   (none)     both
#
# Exit status: 0 if every run passes, cargo's status otherwise.
set -euo pipefail

cd "$(dirname "$0")/.."

export RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }-Dwarnings -Dmissing_docs"

libraries="tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis,tetraloma-testing/mpi"

run() {
    echo "check-docs: cargo doc --workspace --no-deps $*" >&2
    cargo doc --workspace --no-deps "$@"
}

case "${1:-both}" in
default) run ;;
libraries) run --features "$libraries" ;;
both)
    run
    run --features "$libraries"
    ;;
*)
    echo "usage: scripts/check-docs.sh [default|libraries]" >&2
    exit 2
    ;;
esac
echo "check-docs: ok" >&2
