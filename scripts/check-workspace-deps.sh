#!/usr/bin/env bash
# Fails if the default build of the workspace depends on ndelement, BLAS/LAPACK or MPI
# (docs/design.md, "Dependency rules" 5 and 6; M0 acceptance).
#
# Usage: scripts/check-workspace-deps.sh [CARGO-TREE-ARGS...]
#   Extra arguments go to `cargo tree`; for example `--features gridcl-mpi-tests/mpi` must fail.
set -euo pipefail

cd "$(dirname "$0")/.."

forbidden='^(ndelement|.*(blas|lapack|mkl|netlib).*|accelerate-src|mpi(-.*)?)$'

crates=$(cargo tree --workspace --target all -e normal,build,dev --prefix none --format '{p}' "$@" |
    awk '{ print $1 }' | sort -u)
found=$(grep -E "$forbidden" <<<"$crates" || true)

if [[ -n "$found" ]]; then
    echo "check-workspace-deps: the workspace build depends on:" >&2
    while IFS= read -r crate; do
        echo "  $crate" >&2
    done <<<"$found"
    exit 1
fi
echo "check-workspace-deps: ok ($(wc -l <<<"$crates" | tr -d ' ') crates, none forbidden)"
