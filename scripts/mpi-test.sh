#!/usr/bin/env bash
# Runs every gridcl-mpi-tests target at each rank count (docs/design.md, "Testing and
# validation"; CLAUDE.md, MPI tests).
#
# Usage: scripts/mpi-test.sh [RANKS...]
#   Rank counts from the arguments, else from GRIDCL_RANKS (set by scripts/hpc for --ranks N),
#   else 1 2 3 4.
#
# Environment:
#   GRIDCL_MPIRUN_FLAGS      extra mpirun flags, split on whitespace (as in env-check)
#   GRIDCL_MPI_TEST_TIMEOUT  time limit of one run in seconds (default 600)
#   GRIDCL_TEST_DEVICE       forwarded to every rank (Open MPI: -x; MPICH forwards by default);
#                            a backend other than cpu also builds with gridcl-testing/<backend>
#                            (scripts/hpc jobs set cuda)
#
# Prints one line per (ranks, target) and exits non-zero if any run failed. The output of each
# run is kept in target/mpi-test/<target>-<ranks>.log.
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
    echo "mpi-test.sh: $*" >&2
    exit 2
}

if [[ $# -gt 0 ]]; then
    ranks=("$@")
elif [[ -n "${GRIDCL_RANKS:-}" ]]; then
    read -r -a ranks <<<"$GRIDCL_RANKS"
else
    ranks=(1 2 3 4)
fi
[[ ${#ranks[@]} -gt 0 ]] || die "no rank counts given"
for n in "${ranks[@]}"; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "rank count '$n' is not a positive integer"
done

time_limit=${GRIDCL_MPI_TEST_TIMEOUT:-600}
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] || die "GRIDCL_MPI_TEST_TIMEOUT '$time_limit' is not a positive integer"

command -v mpirun >/dev/null || die "mpirun not found"
timeout_cmd=()
if command -v timeout >/dev/null; then
    timeout_cmd=(timeout --kill-after=10 "$time_limit")
elif command -v gtimeout >/dev/null; then
    timeout_cmd=(gtimeout --kill-after=10 "$time_limit")
else
    echo "mpi-test.sh: no timeout command; runs have no time limit" >&2
fi

# The MPI flavour decides the launcher flags.
version=$(mpirun --version 2>&1 || true)
mpirun_flags=()
case $version in
    *"Open MPI"* | *"Open RTE"* | *OpenRTE*)
        flavour="Open MPI"
        mpirun_flags+=(--oversubscribe)
        if [[ -n "${GRIDCL_TEST_DEVICE+set}" ]]; then
            mpirun_flags+=(-x GRIDCL_TEST_DEVICE)
        fi
        ;;
    *HYDRA* | *MPICH*)
        flavour="MPICH"
        ;;
    *)
        flavour="unknown"
        echo "mpi-test.sh: unknown MPI flavour, no launcher flags added:" >&2
        sed -n 1p <<<"$version" >&2
        ;;
esac
if [[ -n "${GRIDCL_MPIRUN_FLAGS:-}" ]]; then
    read -r -a extra <<<"$GRIDCL_MPIRUN_FLAGS"
    mpirun_flags+=("${extra[@]}")
fi

# The device every rank opens must be linked into the test executables. The backends are
# gridcl-testing's runtime features (gridcl-testing/Cargo.toml, TestBackend::ALL).
features=mpi
case ${GRIDCL_TEST_DEVICE:-} in
    "" | cpu) ;;
    cuda | hip | vulkan | metal | metal-native | wgpu)
        features+=",gridcl-testing/$GRIDCL_TEST_DEVICE"
        ;;
    *) die "GRIDCL_TEST_DEVICE '$GRIDCL_TEST_DEVICE' is not a gridcl-testing backend" ;;
esac

# One build; the [[test]] executables come from cargo's JSON messages. Each message is one
# compact object per line, and a target object holds no nested object, so sed suffices (jq is
# not installed on the Myriad compute nodes).
logs=target/mpi-test
mkdir -p "$logs"
build_json=$logs/build.json
if ! cargo test -p gridcl-mpi-tests --features "$features" --no-run --message-format=json >"$build_json"; then
    die "build failed"
fi
tests=()
while IFS= read -r line; do
    tests+=("$line")
done < <(grep -F '"reason":"compiler-artifact"' "$build_json" |
    grep -F '"kind":["test"]' |
    sed -n -E 's/.*"target":\{[^}]*"name":"([^"]*)".*"executable":"([^"]*)".*/\1 \2/p' |
    sort -u)
[[ ${#tests[@]} -gt 0 ]] || die "no gridcl-mpi-tests targets were built"

echo "mpi-test.sh: $flavour, ranks ${ranks[*]}, ${#tests[@]} target(s), features $features, time limit ${time_limit} s"
failed=0
for n in "${ranks[@]}"; do
    for entry in "${tests[@]}"; do
        name=${entry%% *}
        exe=${entry#* }
        log=$logs/$name-$n.log
        start=$SECONDS
        status=0
        ${timeout_cmd[@]+"${timeout_cmd[@]}"} mpirun ${mpirun_flags[@]+"${mpirun_flags[@]}"} -n "$n" "$exe" >"$log" 2>&1 </dev/null ||
            status=$?
        elapsed=$((SECONDS - start))
        if [[ $status -eq 0 ]]; then
            printf 'PASS  ranks=%-2s %s (%d s)\n' "$n" "$name" "$elapsed"
        else
            failed=$((failed + 1))
            reason="exit status $status"
            [[ $status -eq 124 || $status -eq 137 ]] && reason="time limit of $time_limit s exceeded"
            printf 'FAIL  ranks=%-2s %s (%d s, %s; log %s)\n' "$n" "$name" "$elapsed" "$reason" "$log"
            # The panic message comes first and a backtrace last; show both ends.
            if [[ $(wc -l <"$log") -le 80 ]]; then
                sed 's/^/      /' "$log"
            else
                head -n 50 "$log" | sed 's/^/      /'
                echo "      [...]"
                tail -n 30 "$log" | sed 's/^/      /'
            fi
        fi
    done
done

if [[ $failed -gt 0 ]]; then
    echo "mpi-test.sh: $failed run(s) failed" >&2
    exit 1
fi
echo "mpi-test.sh: all runs passed"
