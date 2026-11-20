#!/usr/bin/env bash
# Checks the rank-count guard (docs/design.md, roadmap E2; ADR-0068): a run whose world size
# differs from TETRALOMA_EXPECT_RANKS must fail at once with the guard's message.
#
# Usage: scripts/check-rank-guard.sh [--ranks N]
#
# Builds the MPI test allreduce_smoke (tetraloma-mpi-tests, feature mpi) and the example
# halo_exchange (tetraloma-examples, feature mpi) in the debug build and runs each under
# `mpirun -n 1`:
#   - with TETRALOMA_EXPECT_RANKS=2, a singleton world as a broken launcher gives it, the run must
#     exit non-zero and print "rank-count guard: world size 1";
#   - with TETRALOMA_EXPECT_RANKS=1 (the control) it must pass.
# Then allreduce_smoke under `mpirun -n N` (default 2) with TETRALOMA_EXPECT_RANKS=N+1: the run
# must exit non-zero and every rank 0..N-1 must print the guard's message, which shows that the
# launcher forwards the variable to every rank (with -x under Open MPI; on every node of a
# multi-node job, for example with TETRALOMA_MPIRUN_FLAGS="--map-by node"). A rank without it
# would wait in the allreduce until the time limit.
# Works under any MPI flavour; applies mpi-test.sh's launcher flags and its hwloc workaround
# on Apple chips with three or more performance levels (ADR-0034). Runs on the CPU runtime
# (TETRALOMA_TEST_DEVICE is unset).
#
# Options:
#   --ranks N            the world size of the multi-rank check (N >= 2, default 2)
#
# Environment:
#   TETRALOMA_MPIRUN_FLAGS  extra mpirun flags, split on whitespace (as in mpi-test.sh)
#   CARGO_TARGET_DIR     as for cargo
#
# Prints one line per check and exits non-zero if any failed. Runs under bash 3.2.
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
    echo "check-rank-guard.sh: $*" >&2
    exit 2
}

ranks=2
while [[ $# -gt 0 ]]; do
    case $1 in
        --ranks)
            [[ $# -ge 2 ]] || die "--ranks needs a value"
            ranks=$2
            shift 2
            ;;
        --ranks=*)
            ranks=${1#--ranks=}
            shift
            ;;
        -h | --help)
            sed -n '2,29p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
done
[[ "$ranks" =~ ^[0-9]+$ ]] && ((ranks >= 2)) || die "--ranks '$ranks' is not an integer >= 2"

guard_message="rank-count guard: world size 1"
time_limit=300
unset TETRALOMA_TEST_DEVICE

command -v mpirun >/dev/null || die "mpirun not found"
timeout_cmd=()
if command -v timeout >/dev/null; then
    timeout_cmd=(timeout --kill-after=10 "$time_limit")
elif command -v gtimeout >/dev/null; then
    timeout_cmd=(gtimeout --kill-after=10 "$time_limit")
fi

# As in mpi-test.sh: hwloc 2.15's darwin backend never finishes on Apple chips with three or
# more performance levels (hwloc issue 852); without it Open MPI needs --map-by slot.
hwloc_workaround=
if [[ $(uname -s) == Darwin ]]; then
    levels=$(sysctl -n hw.nperflevels 2>/dev/null || echo 3)
    if [[ ! "$levels" =~ ^[0-9]+$ ]] || ((levels >= 3)); then
        export HWLOC_COMPONENTS=-darwin
        hwloc_workaround=1
    fi
fi

# Open MPI forwards only the variables named by -x; MPICH's hydra forwards the environment.
version=$(mpirun --version 2>&1 || true)
mpirun_flags=()
case $version in
    *"Open MPI"* | *"Open RTE"* | *OpenRTE*)
        flavour="Open MPI"
        mpirun_flags+=(--oversubscribe -x TETRALOMA_EXPECT_RANKS)
        if [[ -n $hwloc_workaround ]]; then
            mpirun_flags+=(-x HWLOC_COMPONENTS)
            [[ ${TETRALOMA_MPIRUN_FLAGS:-} == *--map-by* ]] || mpirun_flags+=(--map-by slot)
        fi
        ;;
    *HYDRA* | *MPICH*) flavour="MPICH" ;;
    *)
        flavour="unknown"
        echo "check-rank-guard.sh: unknown MPI flavour, no launcher flags added" >&2
        ;;
esac
if [[ -n "${TETRALOMA_MPIRUN_FLAGS:-}" ]]; then
    read -r -a extra <<<"$TETRALOMA_MPIRUN_FLAGS"
    mpirun_flags+=("${extra[@]}")
fi

# The executable of target $2 of kind $3 from cargo's JSON messages in file $1 (as in
# mpi-test.sh: one compact object per line, no nested object in a target).
executable() {
    grep -aF '"reason":"compiler-artifact"' "$1" |
        grep -aF "\"kind\":[\"$3\"]" |
        sed -n -E "s/.*\"target\":\\{[^}]*\"name\":\"$2\".*\"executable\":\"([^\"]*)\".*/\\1/p" |
        tail -n 1
}

logs=${CARGO_TARGET_DIR:-target}/check-rank-guard
mkdir -p "$logs"
cargo test -p tetraloma-mpi-tests --features mpi --test allreduce_smoke --no-run \
    --message-format=json >"$logs/build-tests.json" || die "build of allreduce_smoke failed"
cargo build -p tetraloma-examples --features mpi --example halo_exchange \
    --message-format=json >"$logs/build-examples.json" || die "build of halo_exchange failed"
smoke=$(executable "$logs/build-tests.json" allreduce_smoke test)
halo=$(executable "$logs/build-examples.json" halo_exchange example)
[[ -x $smoke ]] || die "allreduce_smoke was not built"
[[ -x $halo ]] || die "halo_exchange was not built"

echo "check-rank-guard.sh: $flavour, mpirun -n 1 of allreduce_smoke and halo_exchange, -n $ranks of allreduce_smoke"
[[ -z $hwloc_workaround ]] || echo "check-rank-guard.sh: HWLOC_COMPONENTS=-darwin (hwloc issue 852)"
failed=0

# check NAME EXECUTABLE EXPECT: runs EXECUTABLE under mpirun -n 1 with TETRALOMA_EXPECT_RANKS=EXPECT;
# EXPECT=1 must pass, any other value must fail with the guard's message.
check() {
    local name=$1 exe=$2 expect=$3 status=0
    local log=$logs/$name-expect-$expect.log
    TETRALOMA_EXPECT_RANKS=$expect ${timeout_cmd[@]+"${timeout_cmd[@]}"} \
        mpirun ${mpirun_flags[@]+"${mpirun_flags[@]}"} -n 1 "$exe" >"$log" 2>&1 </dev/null || status=$?
    local verdict=
    if [[ $expect -eq 1 ]]; then
        [[ $status -eq 0 ]] || verdict="exited $status with TETRALOMA_EXPECT_RANKS=1"
    elif [[ $status -eq 0 ]]; then
        verdict="exited 0 in a singleton world"
    elif [[ $status -eq 124 || $status -eq 137 ]]; then
        verdict="time limit of $time_limit s exceeded"
    elif ! LC_ALL=C tr -d '\000' <"$log" | grep -qF "$guard_message"; then
        verdict="exited $status without the guard's message"
    fi
    if [[ -z $verdict ]]; then
        printf 'PASS  %s, TETRALOMA_EXPECT_RANKS=%s (exit status %d)\n' "$name" "$expect" "$status"
    else
        failed=$((failed + 1))
        printf 'FAIL  %s, TETRALOMA_EXPECT_RANKS=%s: %s (log %s)\n' "$name" "$expect" "$verdict" "$log"
        LC_ALL=C tr -d '\000' <"$log" | tail -n 20 | LC_ALL=C sed 's/^/      /' || true
    fi
}

# check_ranks NAME EXECUTABLE: runs EXECUTABLE under mpirun -n $ranks with
# TETRALOMA_EXPECT_RANKS=$ranks+1; it must fail, and every rank must print
# "NAME: rank r: rank-count guard: world size $ranks," (tetraloma_mpi_tests::run's message).
check_ranks() {
    local name=$1 exe=$2 expect=$((ranks + 1)) status=0 r missing=
    local log=$logs/$name-ranks-$ranks-expect-$expect.log
    TETRALOMA_EXPECT_RANKS=$expect ${timeout_cmd[@]+"${timeout_cmd[@]}"} \
        mpirun ${mpirun_flags[@]+"${mpirun_flags[@]}"} -n "$ranks" "$exe" >"$log" 2>&1 </dev/null ||
        status=$?
    local verdict=
    if [[ $status -eq 0 ]]; then
        verdict="exited 0 in a world of $ranks"
    elif [[ $status -eq 124 || $status -eq 137 ]]; then
        verdict="time limit of $time_limit s exceeded (a rank without TETRALOMA_EXPECT_RANKS waits in the allreduce?)"
    else
        for ((r = 0; r < ranks; r++)); do
            LC_ALL=C tr -d '\000' <"$log" |
                grep -qF "$name: rank $r: rank-count guard: world size $ranks," ||
                missing="$missing $r"
        done
        [[ -z $missing ]] || verdict="no guard message from rank(s)$missing"
    fi
    if [[ -z $verdict ]]; then
        printf 'PASS  %s, -n %d, TETRALOMA_EXPECT_RANKS=%d: every rank failed the guard (exit status %d)\n' \
            "$name" "$ranks" "$expect" "$status"
    else
        failed=$((failed + 1))
        printf 'FAIL  %s, -n %d, TETRALOMA_EXPECT_RANKS=%d: %s (log %s)\n' "$name" "$ranks" "$expect" "$verdict" "$log"
        LC_ALL=C tr -d '\000' <"$log" | tail -n 20 | LC_ALL=C sed 's/^/      /' || true
    fi
}

for entry in "allreduce_smoke $smoke" "halo_exchange $halo"; do
    name=${entry%% *}
    exe=${entry#* }
    check "$name" "$exe" 2
    check "$name" "$exe" 1
done
check_ranks allreduce_smoke "$smoke"

if [[ $failed -gt 0 ]]; then
    echo "check-rank-guard.sh: $failed check(s) failed" >&2
    exit 1
fi
echo "check-rank-guard.sh: all checks passed"
