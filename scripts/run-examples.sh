#!/usr/bin/env bash
# Runs the tetraloma-examples binaries (ADR-0060 item 6; ADR-0067, rule 5): the serial examples
# directly, and the MPI examples (feature mpi) under mpirun. The book includes these programs.
#
# Usage: scripts/run-examples.sh [--serial | --mpi] [--example NAME] [--ranks N]
#                                [--features FEATURES]
#   --serial             only the serial examples (hosted CI's cpu job, which has no MPI)
#   --mpi                only the MPI examples under mpirun
#   --example NAME       only the example NAME (serial or MPI)
#   --ranks N            ranks of the MPI examples: default 2, and 1 with a cuda feature (one GPU)
#   --features FEATURES  cargo features of tetraloma-examples, for example metal, cuda or hdf5;
#                        each example opens the first device of Listing B8 the build links
#
# Environment:
#   TETRALOMA_MPIRUN_FLAGS        extra mpirun flags, split on whitespace (as in mpi-test.sh)
#   TETRALOMA_EXAMPLES_TIMEOUT    time limit of one example in seconds (default 600)
#   TETRALOMA_EXAMPLES_PROFILE    build profile: release (default) or debug (hosted CI, which has
#                              the debug build of the tests already)
#   CARGO_TARGET_DIR           as for cargo (default target)
#
# Prints one line per example and exits non-zero if any failed. Runs under bash 3.2.
set -euo pipefail

cd "$(dirname "$0")/.."

die() {
    echo "run-examples.sh: $*" >&2
    exit 2
}

# The programs (scripts/doc_audit.py reads these two lists: the book may include only them).
serial_examples="first_program devices build_mesh read_gmsh iterate_cells geometric_factors pull_back surface_submesh vtk_output custom_kernel"
mpi_examples="halo_exchange distributed_mesh parallel_io"
only=
run_serial=1
run_mpi=1
ranks=
features=
while [[ $# -gt 0 ]]; do
    case $1 in
        --serial) run_mpi= ;;
        --mpi) run_serial= ;;
        --example)
            [[ $# -ge 2 ]] || die "--example needs a name"
            only=$2
            shift
            ;;
        --example=*) only=${1#--example=} ;;
        --ranks)
            [[ $# -ge 2 ]] || die "--ranks needs a value"
            ranks=$2
            shift
            ;;
        --ranks=*) ranks=${1#--ranks=} ;;
        --features)
            [[ $# -ge 2 ]] || die "--features needs a value"
            features=$2
            shift
            ;;
        --features=*) features=${1#--features=} ;;
        -h | --help)
            sed -n '2,21p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
    shift
done
[[ -n $run_serial || -n $run_mpi ]] || die "--serial and --mpi exclude each other"
if [[ -n $only ]]; then
    case " $serial_examples " in
        *" $only "*)
            serial_examples=$only
            run_mpi=
            ;;
        *)
            case " $mpi_examples " in
                *" $only "*)
                    mpi_examples=$only
                    run_serial=
                    ;;
                *) die "unknown example '$only' (serial: $serial_examples; MPI: $mpi_examples)" ;;
            esac
            ;;
    esac
    [[ -n $run_serial || -n $run_mpi ]] || die "--example $only is not in the selected set"
fi
[[ -z $features || "$features" =~ ^[A-Za-z0-9_/,-]+$ ]] ||
    die "--features '$features' is not a comma-separated feature list"
if [[ -z $ranks ]]; then
    case ",$features," in
        *,cuda,* | *,tetraloma-examples/cuda,*) ranks=1 ;;
        *) ranks=2 ;;
    esac
fi
[[ "$ranks" =~ ^[1-9][0-9]*$ ]] || die "rank count '$ranks' is not a positive integer"

time_limit=${TETRALOMA_EXAMPLES_TIMEOUT:-600}
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] ||
    die "TETRALOMA_EXAMPLES_TIMEOUT '$time_limit' is not a positive integer"
timeout_cmd=()
if command -v timeout >/dev/null; then
    timeout_cmd=(timeout --kill-after=10 "$time_limit")
elif command -v gtimeout >/dev/null; then
    timeout_cmd=(gtimeout --kill-after=10 "$time_limit")
else
    echo "run-examples.sh: no timeout command; examples have no time limit" >&2
fi

profile=${TETRALOMA_EXAMPLES_PROFILE:-release}
case $profile in
    release) profile_flags=(--release) ;;
    debug) profile_flags=() ;;
    *) die "TETRALOMA_EXAMPLES_PROFILE '$profile' is not release or debug" ;;
esac

failed=0

# Runs "$@" under the time limit and prints PASS or FAIL with the elapsed time for $1's example.
run_one() {
    local name=$1
    shift
    local start=$SECONDS status=0
    ${timeout_cmd[@]+"${timeout_cmd[@]}"} "$@" </dev/null || status=$?
    local elapsed=$((SECONDS - start))
    if [[ $status -eq 0 ]]; then
        printf 'PASS  %s (%d s)\n' "$name" "$elapsed"
    else
        failed=$((failed + 1))
        local reason="exit status $status"
        [[ $status -eq 124 || $status -eq 137 ]] && reason="time limit of $time_limit s exceeded"
        printf 'FAIL  %s (%d s, %s)\n' "$name" "$elapsed" "$reason"
    fi
}

feature_flags=()
[[ -z $features ]] || feature_flags=(--features "$features")

if [[ -n $run_serial ]]; then
    echo "run-examples.sh: serial examples, features ${features:-default}, profile $profile"
    # One build of every serial example, so that each run below times the example alone.
    example_flags=()
    for name in $serial_examples; do
        example_flags+=(--example "$name")
    done
    cargo build ${profile_flags[@]+"${profile_flags[@]}"} -p tetraloma-examples \
        ${feature_flags[@]+"${feature_flags[@]}"} "${example_flags[@]}"
    for name in $serial_examples; do
        echo "--- $name"
        run_one "$name" cargo run --quiet ${profile_flags[@]+"${profile_flags[@]}"} \
            -p tetraloma-examples ${feature_flags[@]+"${feature_flags[@]}"} --example "$name"
    done
fi

if [[ -n $run_mpi ]]; then
    command -v mpirun >/dev/null || die "mpirun not found"

    # As in mpi-test.sh: hwloc 2.15's darwin backend never finishes on Apple chips with three
    # or more performance levels (hwloc issue 852); without it Open MPI needs --map-by slot.
    # Set for the mpirun commands of this script only (ADR-0034).
    mpi_env=()
    hwloc_workaround=
    if [[ $(uname -s) == Darwin ]]; then
        levels=$(sysctl -n hw.nperflevels 2>/dev/null || echo 3)
        if [[ ! "$levels" =~ ^[0-9]+$ ]] || ((levels >= 3)); then
            mpi_env=(env HWLOC_COMPONENTS=-darwin)
            hwloc_workaround=1
        fi
    fi
    version=$(mpirun --version 2>&1 || true)
    mpirun_flags=()
    case $version in
        *"Open MPI"* | *"Open RTE"* | *OpenRTE*)
            flavour="Open MPI"
            mpirun_flags+=(--oversubscribe)
            if [[ -n $hwloc_workaround ]]; then
                mpirun_flags+=(-x HWLOC_COMPONENTS)
                [[ ${TETRALOMA_MPIRUN_FLAGS:-} == *--map-by* ]] || mpirun_flags+=(--map-by slot)
            fi
            ;;
        *HYDRA* | *MPICH*) flavour="MPICH" ;;
        *)
            flavour="unknown"
            echo "run-examples.sh: unknown MPI flavour, no launcher flags added" >&2
            ;;
    esac
    if [[ -n "${TETRALOMA_MPIRUN_FLAGS:-}" ]]; then
        read -r -a extra <<<"$TETRALOMA_MPIRUN_FLAGS"
        mpirun_flags+=("${extra[@]}")
    fi

    mpi_features=mpi
    [[ -z $features ]] || mpi_features="mpi,$features"
    echo "run-examples.sh: MPI examples on $ranks rank(s), $flavour, features $mpi_features, profile $profile"
    [[ -z $hwloc_workaround ]] || echo "run-examples.sh: HWLOC_COMPONENTS=-darwin (hwloc issue 852)"
    example_flags=()
    for name in $mpi_examples; do
        example_flags+=(--example "$name")
    done
    cargo build ${profile_flags[@]+"${profile_flags[@]}"} -p tetraloma-examples \
        --features "$mpi_features" "${example_flags[@]}"
    for name in $mpi_examples; do
        exe=${CARGO_TARGET_DIR:-target}/$profile/examples/$name
        [[ -x $exe ]] || die "$exe was not built"
        echo "--- $name"
        run_one "$name" ${mpi_env[@]+"${mpi_env[@]}"} \
            mpirun ${mpirun_flags[@]+"${mpirun_flags[@]}"} -n "$ranks" "$exe"
    done
fi

if [[ $failed -gt 0 ]]; then
    echo "run-examples.sh: $failed example(s) failed" >&2
    exit 1
fi
echo "run-examples.sh: all examples passed"
