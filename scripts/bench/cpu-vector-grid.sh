#!/usr/bin/env bash
# The M10c grid of the `[Cpu]` vector entries (docs/plans/M10c.md, T3 and T5; ADR-0065): two
# `vector_sweep --grid` processes on the CPU runtime, seeds 1 and 2, one after the other, each
# writing `m10c-vector-cpu-<target>-s<seed>.json` into the output directory.
#
# Usage: scripts/bench/cpu-vector-grid.sh --target NAME --out-dir DIR [--bind COMMAND]
#            [--max-load L] [--seeds 1,2] [-- VECTOR_SWEEP_OPTIONS...]
#        scripts/bench/cpu-vector-grid.sh annotate RESULTS_DIR DIR TARGET
#   --target: macmini, locust or kathleen (myriad-cpu only as Kathleen's fallback, with
#       --fallback-reason after `--`).
#   --bind: a command prefix for each process, e.g. 'numactl --physcpubind=0-19 --membind=0'
#       on one Kathleen socket (ADR-0020).
#   --max-load: before each process, wait until the 1-minute load average is below L (at most
#       15 minutes, else fail); for the Mac mini (2), where no scheduler keeps the node free.
#   --seeds: the process seeds (default 1,2).
#   After `--`: further vector_sweep options (--sizes, --rounds, --families, --types,
#       --fallback-reason), passed to every process.
#   annotate: after a scripts/hpc job has been fetched, adds the runner's node-load snapshots
#       of RESULTS_DIR (results/<job id>/) to every record of DIR's files of TARGET, since the
#       "after" snapshot is taken once the job's command has finished (ADR-0002, ADR-0005).
#
# In a scripts/hpc job on a clean tree the records carry the job's commit (GRIDCL_BENCH_COMMIT
# from GRIDCL_HPC_COMMIT), since the remote tree has no .git. Runs under bash 3.2 (macOS).
set -euo pipefail

cd "$(dirname "$0")/../.."

die() {
    echo "cpu-vector-grid.sh: $*" >&2
    exit 2
}

bin=${CARGO_TARGET_DIR:-target}/release/vector_sweep

if [[ ${1:-} == annotate ]]; then
    [[ $# -eq 4 ]] || die "usage: annotate RESULTS_DIR DIR TARGET"
    shopt -s nullglob
    files=("$3"/m10c-vector-cpu-"$4"-s*.json)
    [[ ${#files[@]} -gt 0 ]] || die "no m10c-vector-cpu-$4-s*.json in $3"
    cargo build --release -p gridcl-bench --bin vector_sweep
    exec "$bin" annotate --node-load "$2" "${files[@]}"
fi

target=""
out_dir=""
bind=""
max_load=""
seeds=1,2
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --target | --out-dir | --bind | --max-load | --seeds)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case $1 in
                --target) target=$2 ;;
                --out-dir) out_dir=$2 ;;
                --bind) bind=$2 ;;
                --max-load) max_load=$2 ;;
                --seeds) seeds=$2 ;;
            esac
            shift 2
            ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -n "$target" ]] || die "--target NAME is required"
[[ -n "$out_dir" ]] || die "--out-dir DIR is required"
[[ -z "$max_load" || "$max_load" =~ ^[0-9]+(\.[0-9]+)?$ ]] || die "--max-load '$max_load'"
[[ "${GRIDCL_TEST_DEVICE:-cpu}" == cpu ]] || die "the grid runs on the CPU runtime only"
export GRIDCL_TEST_DEVICE=cpu
if [[ -n "${GRIDCL_HPC_COMMIT:-}" && -z "${GRIDCL_BENCH_COMMIT:-}" ]]; then
    [[ "${GRIDCL_HPC_DIRTY:-}" == no ]] || die "the job's tree is dirty; record a clean commit"
    export GRIDCL_BENCH_COMMIT=$GRIDCL_HPC_COMMIT
fi

# The 1-minute load average.
load1() {
    if [[ -r /proc/loadavg ]]; then
        cut -d' ' -f1 /proc/loadavg
    else
        sysctl -n vm.loadavg | awk '{print $2}'
    fi
}

# Waits until the 1-minute load average is below --max-load, at most 15 minutes.
wait_for_load() {
    [[ -n "$max_load" ]] || return 0
    local waited=0 load
    while true; do
        load=$(load1)
        if awk -v l="$load" -v m="$max_load" 'BEGIN { exit !(l < m) }'; then
            echo "load average $load (below $max_load) after ${waited} s"
            return 0
        fi
        [[ $waited -lt 900 ]] || die "load average $load still at least $max_load after 900 s"
        sleep 15
        waited=$((waited + 15))
    done
}

cargo build --release -p gridcl-bench --bin vector_sweep
[[ -x "$bin" ]] || die "$bin not built"
mkdir -p "$out_dir"
echo "== $(uname -a)"
command -v numactl >/dev/null && numactl --hardware | head -5 || true

IFS=, read -r -a seed_list <<<"$seeds"
for seed in "${seed_list[@]}"; do
    [[ "$seed" =~ ^[0-9]+$ ]] || die "seed '$seed' is not a non-negative integer"
    out=$out_dir/m10c-vector-cpu-$target-s$seed.json
    wait_for_load
    echo "seed $seed: start $(date -u +%FT%TZ), load: $(uptime)"
    start=$SECONDS
    # shellcheck disable=SC2086 # $bind is a command prefix
    $bind "$bin" --grid --seed "$seed" --target "$target" --out "$out" "$@"
    echo "seed $seed: $((SECONDS - start)) s, load: $(uptime); records in $out"
done
