#!/usr/bin/env bash
# The M9h degree sweep (docs/plans/M9h.md, T3 and T11a; ADR-0042, item 12) under a time limit
# per process: one `degree_sweep` process per (cell, degree) with every family and type, so the
# meshes are built once per process, all writing to one JSON file with `--resume`. A process
# stopped by its limit is followed by one process per (family, type) without a record, each
# under the limit of one configuration, so a compile that never finishes costs its group's
# limit and one more, not the run.
#
# Usage: scripts/bench/degree-sweep.sh --out FILE [--timeout SECONDS] [--bind COMMAND]
#            [--families LIST] [--cells LIST] [--degrees LIST] [--types LIST]
#            [-- DEGREE_SWEEP_OPTIONS...]
#   --out: the JSON file of the records; an existing file is resumed (its configurations are
#       not run again), so a run can be repeated after an interruption.
#   --timeout: time limit of one (family, cell, degree, type) configuration, in seconds
#       (default 1800); a (cell, degree) process gets it once per configuration it runs. A
#       process stopped by its limit leaves the records it wrote; after the per-configuration
#       processes, `degree_sweep --stopped-after` records the configurations still without a
#       record as `too slow to compile` (no record of the kernel yet) or `too slow`.
#   --bind: a command prefix for the sweep, e.g. 'numactl --physcpubind=0-19 --membind=0' on
#       one Kathleen socket (ADR-0020).
#   --families, --cells, --degrees, --types: comma-separated, as degree_sweep names them
#       (default every family and cell type, degrees 1-8 with pyramids to 3, f32 and f64; f64
#       is skipped by degree_sweep on a device without it).
#   After `--`: further degree_sweep options (--sizes, --node-families, --strategies, --skip,
#       --fallback-reason, ...), passed to every process.
#
# The backend is GRIDCL_TEST_DEVICE (cpu by default), built with the matching gridcl-testing
# feature. In a scripts/hpc job on a clean tree the records carry the job's commit
# (GRIDCL_BENCH_COMMIT from GRIDCL_HPC_COMMIT), since the remote tree has no .git. Prints
# `[k/N]` before each (cell, degree) process (shown by scripts/hpc/status.sh while the job
# runs) and the sweep's lines; logs of stopped or failed processes stay next to FILE. Runs
# under bash 3.2 (macOS).
set -euo pipefail

cd "$(dirname "$0")/../.."

die() {
    echo "degree-sweep.sh: $*" >&2
    exit 2
}

families=EVector,Points,Jacobian,DetJ,Pinv,Normals,FacetNormals,PullBack,BasisEval
cells=Interval,Triangle,Quadrilateral,Tetrahedron,Hexahedron,Prism,Pyramid
degrees=1,2,3,4,5,6,7,8
types=f32,f64
time_limit=1800
bind=""
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --)
            shift
            break
            ;;
        --families | --cells | --degrees | --types | --timeout | --bind | --out)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case $1 in
                --families) families=$2 ;;
                --cells) cells=$2 ;;
                --degrees) degrees=$2 ;;
                --types) types=$2 ;;
                --timeout) time_limit=$2 ;;
                --bind) bind=$2 ;;
                --out) out=$2 ;;
            esac
            shift 2
            ;;
        *) die "unknown option $1" ;;
    esac
done
[[ -n "$out" ]] || die "--out FILE is required"
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] || die "--timeout '$time_limit' is not a positive integer"
command -v timeout >/dev/null || die "timeout not found"

device=${GRIDCL_TEST_DEVICE:-cpu}
case $device in
    cpu) features=() ;;
    cuda | metal | vulkan) features=(--features "gridcl-testing/$device") ;;
    *) die "GRIDCL_TEST_DEVICE '$device' is not cpu, cuda, metal or vulkan" ;;
esac
if [[ -n "${GRIDCL_HPC_COMMIT:-}" && -z "${GRIDCL_BENCH_COMMIT:-}" ]]; then
    [[ "${GRIDCL_HPC_DIRTY:-}" == no ]] || die "the job's tree is dirty; record a clean commit"
    export GRIDCL_BENCH_COMMIT=$GRIDCL_HPC_COMMIT
fi

cargo build --release -p gridcl-bench --bin degree_sweep ${features[@]+"${features[@]}"}
bin=${CARGO_TARGET_DIR:-target}/release/degree_sweep
[[ -x "$bin" ]] || die "$bin not built"
mkdir -p "$(dirname "$out")"
logs=${out%.json}.logs
mkdir -p "$logs"

IFS=, read -r -a family_list <<<"$families"
IFS=, read -r -a cell_list <<<"$cells"
IFS=, read -r -a degree_list <<<"$degrees"
IFS=, read -r -a type_list <<<"$types"
groups=()
for cell in "${cell_list[@]}"; do
    for degree in "${degree_list[@]}"; do
        [[ $cell != Pyramid || $degree -le 3 ]] || continue
        configs=()
        for family in "${family_list[@]}"; do
            if [[ $family == Normals ]] && [[ $cell =~ ^(Tetrahedron|Hexahedron|Prism|Pyramid)$ ]]; then
                continue
            fi
            for type in "${type_list[@]}"; do
                configs+=("$family:$type")
            done
        done
        [[ ${#configs[@]} -eq 0 ]] || groups+=("$cell $degree ${configs[*]}")
    done
done

# run <limit> <log> <filters...>: one degree_sweep process under <limit> seconds; sets rc.
run() {
    local limit=$1 log=$2
    shift 2
    rc=0
    # shellcheck disable=SC2086 # $bind is a command prefix
    timeout -k 30 "$limit" $bind "$bin" "$@" "${options[@]+"${options[@]}"}" --out "$out" \
        --resume 2> >(tee "$log" >&2) || rc=$?
}

options=("$@")
processes=0
stopped=0
failed=0
for i in "${!groups[@]}"; do
    read -r cell degree configs <<<"${groups[$i]}"
    read -r -a configs <<<"$configs"
    group_families=$(printf '%s\n' "${configs[@]%%:*}" | awk '!seen[$0]++' | paste -sd, -)
    group_types=$(printf '%s\n' "${configs[@]##*:}" | awk '!seen[$0]++' | paste -sd, -)
    limit=$((time_limit * ${#configs[@]}))
    group=$cell-p$degree
    echo "[$((i + 1))/${#groups[@]}] $cell p=$degree, ${#configs[@]} configurations ($SECONDS s)"
    processes=$((processes + 1))
    run "$limit" "$logs/$group.log" --families "$group_families" --cells "$cell" \
        --degrees "$degree" --types "$group_types"
    if [[ $rc -eq 0 ]]; then
        rm -f "$logs/$group.log"
        continue
    fi
    if [[ $rc -eq 124 || $rc -eq 137 ]]; then
        echo "$group: stopped after $limit s; the rest one configuration a process"
        stopped=$((stopped + 1))
    else
        echo "$group: degree_sweep exited with $rc (log $logs/$group.log);" \
            "the rest one configuration a process"
        failed=$((failed + 1))
    fi
    # The configurations of the group, each in its own process; `--resume` skips those the
    # group recorded.
    for config in "${configs[@]}"; do
        family=${config%%:*}
        type=${config##*:}
        cfg=$family-$cell-p$degree-$type
        filters=(--families "$family" --cells "$cell" --degrees "$degree" --types "$type")
        processes=$((processes + 1))
        run "$time_limit" "$logs/$cfg.log" "${filters[@]}"
        if [[ $rc -eq 124 || $rc -eq 137 ]]; then
            echo "$cfg: stopped after $time_limit s"
            stopped=$((stopped + 1))
            "$bin" "${filters[@]}" "${options[@]+"${options[@]}"}" --out "$out" \
                --stopped-after "$time_limit" ||
                die "$cfg: cannot record the stopped configurations"
        elif [[ $rc -ne 0 ]]; then
            echo "$cfg: degree_sweep exited with $rc (log $logs/$cfg.log)"
            failed=$((failed + 1))
        else
            rm -f "$logs/$cfg.log"
        fi
    done
done
echo "degree-sweep.sh: ${#groups[@]} (cell, degree) groups in $processes processes," \
    "$stopped stopped, $failed failed; records in $out"
[[ $failed -eq 0 ]] || exit 1
