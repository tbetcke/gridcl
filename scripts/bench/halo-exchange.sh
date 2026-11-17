#!/usr/bin/env bash
# The M10 halo-exchange records (docs/plans/M10.md, T6; ADR-0059, ADR-0061): `halo_exchange`
# under mpirun at each rank count, all appending to one JSON file.
#
# Usage: scripts/bench/halo-exchange.sh --out FILE [--ranks 1,2] [-- HALO_EXCHANGE_OPTIONS...]
#   --out: the JSON file of the records; an existing file is replaced.
#   --ranks: the rank counts, separated by commas or spaces (default 2).
#   After `--`: further halo_exchange options (--target, --types, --fallback-reason, ...).
#
# The backend is GRIDCL_TEST_DEVICE (cpu by default), built with the matching gridcl-testing
# feature and `mpi`. In a scripts/hpc job on a clean tree the records carry the job's commit
# (GRIDCL_BENCH_COMMIT from GRIDCL_HPC_COMMIT), since the remote tree has no .git. On Apple chips
# with three or more performance levels it applies mpi-test.sh's hwloc workaround (ADR-0034).
# Runs under bash 3.2 (macOS).
set -euo pipefail

cd "$(dirname "$0")/../.."

die() {
    echo "halo-exchange.sh: $*" >&2
    exit 2
}

out=""
ranks="2"
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --out | --ranks)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case $1 in
                --out) out=$2 ;;
                --ranks) ranks=$2 ;;
            esac
            shift 2
            ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -n "$out" ]] || die "--out FILE is required"
ranks=${ranks//,/ }
for n in $ranks; do
    [[ "$n" =~ ^[1-9][0-9]*$ ]] || die "--ranks: '$n' is not a positive integer"
done

device=${GRIDCL_TEST_DEVICE:-cpu}
case $device in
    cpu | cuda | metal | vulkan) features="mpi,gridcl-testing/$device" ;;
    *) die "GRIDCL_TEST_DEVICE '$device' is not cpu, cuda, metal or vulkan" ;;
esac
if [[ -n "${GRIDCL_HPC_COMMIT:-}" && -z "${GRIDCL_BENCH_COMMIT:-}" ]]; then
    [[ "${GRIDCL_HPC_DIRTY:-}" == no ]] || die "the job's tree is dirty; record a clean commit"
    export GRIDCL_BENCH_COMMIT=$GRIDCL_HPC_COMMIT
fi

command -v mpirun >/dev/null || die "mpirun not found"
flags=()
if [[ $(uname -s) == Darwin ]]; then
    levels=$(sysctl -n hw.nperflevels 2>/dev/null || echo 3)
    if [[ ! "$levels" =~ ^[0-9]+$ ]] || ((levels >= 3)); then
        export HWLOC_COMPONENTS=-darwin
    fi
fi
case $(mpirun --version 2>&1 || true) in
    *"Open MPI"* | *"Open RTE"* | *OpenRTE*)
        flags+=(--oversubscribe -x GRIDCL_TEST_DEVICE)
        [[ -z "${GRIDCL_BENCH_COMMIT:-}" ]] || flags+=(-x GRIDCL_BENCH_COMMIT)
        [[ -z "${GRIDCL_HPC_TARGET:-}" ]] || flags+=(-x GRIDCL_HPC_TARGET)
        if [[ -n "${HWLOC_COMPONENTS:-}" ]]; then
            flags+=(-x HWLOC_COMPONENTS --map-by slot)
        fi
        ;;
esac
export GRIDCL_TEST_DEVICE=$device

cargo build --release -p gridcl-bench --bin halo_exchange --features "$features"
bin=${CARGO_TARGET_DIR:-target}/release/halo_exchange
[[ -x "$bin" ]] || die "$bin not built"
mkdir -p "$(dirname "$out")"
rm -f "$out"
for n in $ranks; do
    echo "halo-exchange.sh: $n rank(s), $device ($SECONDS s)"
    mpirun ${flags[@]+"${flags[@]}"} -n "$n" "$bin" --out "$out" --append "$@"
done
echo "halo-exchange.sh: done ($SECONDS s)"
