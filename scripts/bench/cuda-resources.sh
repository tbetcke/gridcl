#!/usr/bin/env bash
# Registers, spill and local memory per thread of the M9h degree-sweep kernels on CUDA
# (docs/plans/M9h.md, T2; ADR-0042, item 6), from `ptxas -v` on the PTX that cubecl-cuda
# writes under CUBECL_CUDA_DUMP_PTX, as `m9h-cuda-resources-<target>.json`
# (gridcl_bench::record::CudaResources).
#
# Usage: scripts/bench/cuda-resources.sh [--families LIST] [--cells LIST] [--degrees LIST]
#            [--types LIST] [--strategies LIST] [--timeout SECONDS] [--scalar-only] [--sanitize]
#            [--ptxas-only DIR] [--out FILE]
#   --families, --cells, --degrees, --types, --strategies: comma-separated, as degree_sweep
#       names them (default every family and cell type, degrees 1-8 with pyramids to 3, f32
#       and f64, direct).
#   --timeout: time limit of one configuration, and of `ptxas` on one kernel, in seconds
#       (default 600).
#   --scalar-only: launch vector size 1 only, not also the policy's.
#   --sanitize: run each configuration under compute-sanitizer, which names a failing driver
#       call (cuFuncSetAttribute, cuLaunchKernel); slow.
#   --ptxas-only: launch nothing; run `ptxas` again on the PTX and launch records an earlier
#       run left in DIR (its `cuda-resources/` work directory), e.g. after a driver update made
#       that run's `ptxas` too old.
#   --out: the JSON file (default results/$JOB_ID/m9h-cuda-resources-<target>.json in a
#       scripts/hpc job, else target/cuda-resources/m9h-cuda-resources-<target>.json).
#
# Each (family, cell, degree, type, strategy, vector size) runs as its own
# `degree_sweep --launch-only` process with its own dump directory: cubecl-cuda names a dumped
# file by the tail of the kernel id, which kernels differing only in comptime settings share
# (docs/cubecl-upstream.md, C19). The vector sizes are 1 and the policy's (1 only with
# --scalar-only). ncu is not used:
# Locust restricts GPU performance counters to administrators (ERR_NVGPUCTRPERM).
#
# A record joins a family kernel's PTX entry (its name, with the vector size `_n_<N>_`) to the
# configuration's launch record. Family kernels are matched by name: `evector` (EVector),
# `interpolant` (BasisEval), `pull_back` (PullBack) and `factor` (the factor families); kernel
# names of new strategies keep these words. The maximum threads per block follow from the
# register count (65536 registers per block, allocated per warp in units of 256, at most 1024
# threads; compute capability 7.0 to 9.0).
#
# The work files (records, logs, PTX, ptxas output) stay in <out dir>/cuda-resources/; rec/
# holds each configuration's launch record (`degree_sweep --launch-only`, one new kernel per
# process, so with CUDA_CACHE_DISABLE=1 its compile time is that kernel's first use; a process
# stopped by --timeout gets a `too slow to compile` record), which make the CUDA launch file
# `m9h-launch-cuda-<target>.json` (docs/plans/M9h.md, T11a). Prints
# `[k/N]` before each configuration (shown by scripts/hpc/status.sh while the job runs),
# then one line per kernel, and exits 1 if a configuration launched (status ok) without a
# record.
# `ptxas` is $PTXAS, else the one on PATH; it must be at least the driver's CUDA version, since
# cubecl-cuda emits the newest PTX version the driver loads (Locust: /usr/local/cuda-13.5 for
# driver 620.70, CUDA 13.5, from 2026-11; docs/dev-environment.md).
# Needs bash 5 (EPOCHREALTIME, associative arrays).
set -euo pipefail

cd "$(dirname "$0")/../.."

die() {
    echo "cuda-resources.sh: $*" >&2
    exit 2
}

families=EVector,Points,Jacobian,DetJ,Pinv,Normals,FacetNormals,PullBack,BasisEval
cells=Interval,Triangle,Quadrilateral,Tetrahedron,Hexahedron,Prism,Pyramid
degrees=1,2,3,4,5,6,7,8
types=f32,f64
strategies=direct
time_limit=600
sanitize=no
scalar_only=no
reuse=""
out=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --families | --cells | --degrees | --types | --strategies | --timeout | --out | --ptxas-only)
            [[ $# -ge 2 ]] || die "$1 needs a value"
            case $1 in
                --families) families=$2 ;;
                --cells) cells=$2 ;;
                --degrees) degrees=$2 ;;
                --types) types=$2 ;;
                --strategies) strategies=$2 ;;
                --timeout) time_limit=$2 ;;
                --out) out=$2 ;;
                --ptxas-only) reuse=$2 ;;
            esac
            shift 2
            ;;
        --sanitize)
            sanitize=yes
            shift
            ;;
        --scalar-only)
            scalar_only=yes
            shift
            ;;
        *) die "unknown option $1" ;;
    esac
done
[[ "$time_limit" =~ ^[1-9][0-9]*$ ]] || die "--timeout '$time_limit' is not a positive integer"

target=${GRIDCL_HPC_TARGET:-local}
if [[ -z "$out" ]]; then
    if [[ -n "${JOB_ID:-}" ]]; then
        out=results/$JOB_ID/m9h-cuda-resources-$target.json
    else
        out=target/cuda-resources/m9h-cuda-resources-$target.json
    fi
fi
work=$(dirname "$out")/cuda-resources
mkdir -p "$work/ptx" "$work/rec" "$work/log" "$work/ptxas"

ptxas=${PTXAS:-ptxas}
command -v "$ptxas" >/dev/null || die "$ptxas not found"
command -v timeout >/dev/null || die "timeout not found"
if [[ $sanitize == yes ]]; then
    command -v compute-sanitizer >/dev/null || die "compute-sanitizer not found"
fi
[[ -z "$reuse" || -d "$reuse/rec" ]] || die "--ptxas-only $reuse: no rec/ directory"
echo "driver: $(nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader 2>/dev/null || echo unknown)"
# `CUDA Version: 12.7` before driver 580, `CUDA UMD Version: 13.5` from then on.
driver_cuda=$(nvidia-smi 2>/dev/null | sed -n 's/.*CUDA \(UMD \)\{0,1\}Version: *\([0-9.]*\).*/\2/p' | head -n 1)
ptxas_cuda=$("$ptxas" --version | sed -n 's/.*release \([0-9.]*\).*/\1/p')
echo "driver CUDA: $driver_cuda"
echo "ptxas: $ptxas_cuda ($(command -v "$ptxas"))"
if [[ -n "$driver_cuda" && -n "$ptxas_cuda" ]] &&
    [[ $(printf '%s\n%s\n' "$ptxas_cuda" "$driver_cuda" | sort -V | head -n 1) != "$driver_cuda" ]]; then
    die "ptxas $ptxas_cuda is older than the driver's CUDA $driver_cuda; set PTXAS"
fi

bin=${CARGO_TARGET_DIR:-target}/release/degree_sweep
if [[ -z "$reuse" ]]; then
    cargo build --release -p gridcl-bench --bin degree_sweep --features gridcl-testing/cuda
    [[ -x "$bin" ]] || die "$bin not built"
fi

# The field `key` (a string or a number) of the first record in a degree_sweep file `file`.
field() {
    sed -n -e "s/^ *\"$2\": \"\(.*\)\",\{0,1\}\$/\1/p" \
        -e "s/^ *\"$2\": \([0-9][0-9.e+-]*\),\{0,1\}\$/\1/p" "$1" | head -n 1
}

# Whether the PTX entry `entry` is a kernel of the family `family`.
family_kernel() {
    case $1 in
        EVector) [[ $2 == *evector* ]] ;;
        BasisEval) [[ $2 == *interpolant* ]] ;;
        PullBack) [[ $2 == *pull_back* ]] ;;
        *) [[ $2 == *factor* ]] ;;
    esac
}

# Maximum threads per block for `registers` registers per thread.
max_threads() {
    local per_warp=$((($1 * 32 + 255) / 256 * 256))
    local warps=$((65536 / per_warp))
    local threads=$((warps * 32))
    echo $((threads < 1024 ? threads : 1024))
}

records=()
declare -A seen=()
missing=0
# Runs one configuration; arguments: family cell degree type strategy vector-sizes.
run_config() {
    local family=$1 cell=$2 degree=$3 type=$4 strategy=$5 sizes=$6
    local cfg=$family-$cell-p$degree-$type-$strategy-$sizes
    local rec=$work/rec/$cfg.json log=$work/log/$cfg.log dump=$work/ptx/$cfg
    local cmd=(env "CUBECL_CUDA_DUMP_PTX=$dump")
    if [[ $sanitize == yes ]]; then
        cmd+=(compute-sanitizer --tool memcheck --print-limit 20 --log-file "$work/log/$cfg.sanitizer.log")
    fi
    local start=$SECONDS rc=0
    if [[ -n "$reuse" ]]; then
        rec=$reuse/rec/$cfg.json
        dump=$reuse/ptx/$cfg
        [[ -s "$rec" ]] || rc=1
    else
        rm -rf "$dump" "$rec"
        timeout -k 10 "$time_limit" "${cmd[@]}" "$bin" --launch-only --families "$family" \
            --cells "$cell" --degrees "$degree" --types "$type" --strategies "$strategy" \
            --vector-sizes "$sizes" --out "$rec" >"$log" 2>&1 || rc=$?
    fi
    local status=ok
    [[ $rc -eq 0 ]] || status="exit $rc"
    [[ $rc -ne 124 ]] || status="timeout after $time_limit s"
    if [[ $rc -eq 124 && ! -s "$rec" ]]; then
        # The launch record of a stopped process (`too slow to compile`), for the launch files.
        "$bin" --launch-only --families "$family" --cells "$cell" --degrees "$degree" \
            --types "$type" --strategies "$strategy" --vector-sizes "$sizes" --out "$rec" \
            --stopped-after "$time_limit" >>"$log" 2>&1 || rm -f "$rec"
    fi
    if [[ -s "$rec" ]]; then
        status=$(field "$rec" status)
        [[ $rc -eq 0 ]] || status="$status (exit $rc)"
    fi
    local found=0 f entry arch line regs stack spill t0 ptxas_s vector ran key
    ran=$strategy
    [[ ! -s "$rec" ]] || ran=$(field "$rec" strategy_ran)
    for f in "$dump"/*.ptx; do
        [[ -e "$f" ]] || continue
        entry=$(sed -n 's/^\.visible \.entry \([A-Za-z0-9_]*\)(.*/\1/p' "$f" | head -n 1)
        family_kernel "$family" "$entry" || continue
        found=$((found + 1))
        # The policy's run plans vector size 1 again where the register budget limits it.
        key=$family-$cell-$degree-$type-$strategy-$entry
        [[ -z "${seen[$key]:-}" ]] || continue
        seen[$key]=1
        arch=$(sed -n 's/^\.target \([a-z0-9_]*\).*/\1/p' "$f" | head -n 1)
        t0=$EPOCHREALTIME
        timeout -k 10 "$time_limit" "$ptxas" "-arch=$arch" -v "$f" -o /dev/null \
            >"$work/ptxas/$cfg-$entry.txt" 2>&1 || true
        ptxas_s=$(awk -v a="$t0" -v b="$EPOCHREALTIME" 'BEGIN { printf "%.1f", b - a }')
        line=$(cat "$work/ptxas/$cfg-$entry.txt")
        regs=$(sed -n 's/.*Used \([0-9]*\) registers.*/\1/p' <<<"$line" | head -n 1)
        stack=$(sed -n 's/.* \([0-9]*\) bytes stack frame.*/\1/p' <<<"$line" | head -n 1)
        spill=$(sed -n 's/.* \([0-9]*\) bytes spill stores.*/\1/p' <<<"$line" | head -n 1)
        if [[ -z "$regs" || -z "$stack" || -z "$spill" ]]; then
            echo "$cfg: ptxas failed on $entry (see $work/ptxas/$cfg-$entry.txt)"
            continue
        fi
        vector=$(sed -n 's/.*_n_\([0-9]*\)_[0-9a-f]*$/\1/p' <<<"$entry")
        [[ -n "$vector" ]] || vector=1
        records+=("$(printf '  {"target": "%s", "kernel": "%s", "family": "%s", "cell": "%s", "degree": %s, "scalar": "%s", "strategy": "%s", "vector_size": %s, "registers": %s, "spill_bytes": %s, "local_bytes": %s, "max_threads_per_block": %s, "source": "ptxas"}' \
            "$target" "$entry" "$family" "$cell" "$degree" "$type" "$ran" "$vector" "$regs" \
            "$spill" "$stack" "$(max_threads "$regs")")")
        printf '%-48s %s: %s registers, %s B spill, %s B stack, max %s threads; ptxas %s s; run %d s\n' \
            "$cfg" "$entry" "$regs" "$spill" "$stack" "$(max_threads "$regs")" "$ptxas_s" \
            $((SECONDS - start))
    done
    if [[ $found -eq 0 ]]; then
        echo "$cfg: no family kernel (status: $status; log $log)"
        [[ $status != ok ]] || missing=$((missing + 1))
    elif [[ $status != ok ]]; then
        echo "$cfg: status: $status"
    fi
    if [[ $sanitize == yes ]]; then
        grep -h "CUDA API call to\|ERROR SUMMARY" "$work/log/$cfg.sanitizer.log" 2>/dev/null |
            sed "s/^/$cfg: /" | head -n 5 || true
    fi
    # The policy's vector size, for the caller: 1 when it equals the scalar run's.
    policy_vector=1
    [[ ! -s "$rec" ]] || policy_vector=$(field "$rec" policy_vector_size)
}

IFS=, read -r -a family_list <<<"$families"
IFS=, read -r -a cell_list <<<"$cells"
IFS=, read -r -a degree_list <<<"$degrees"
IFS=, read -r -a type_list <<<"$types"
IFS=, read -r -a strategy_list <<<"$strategies"
configs=()
for strategy in "${strategy_list[@]}"; do
    for family in "${family_list[@]}"; do
        for cell in "${cell_list[@]}"; do
            if [[ $family == Normals ]] && [[ $cell =~ ^(Tetrahedron|Hexahedron|Prism|Pyramid)$ ]]; then
                continue
            fi
            for degree in "${degree_list[@]}"; do
                [[ $cell != Pyramid || $degree -le 3 ]] || continue
                for type in "${type_list[@]}"; do
                    configs+=("$family $cell $degree $type $strategy")
                done
            done
        done
    done
done
# Progress: `[k/N]` counts configurations; the policy's run follows its scalar run.
for i in "${!configs[@]}"; do
    read -r family cell degree type strategy <<<"${configs[$i]}"
    echo "[$((i + 1))/${#configs[@]}] $family $cell p=$degree $type $strategy (${SECONDS} s)"
    run_config "$family" "$cell" "$degree" "$type" "$strategy" scalar
    if [[ $scalar_only == no && ${policy_vector:-1} != 1 ]]; then
        run_config "$family" "$cell" "$degree" "$type" "$strategy" policy
    fi
done

{
    echo "["
    for i in "${!records[@]}"; do
        if [[ $i -lt $((${#records[@]} - 1)) ]]; then
            echo "${records[$i]},"
        else
            echo "${records[$i]}"
        fi
    done
    echo "]"
} >"$out"
echo "${#records[@]} records in $out"
if [[ $missing -gt 0 ]]; then
    echo "cuda-resources.sh: $missing launched configurations without a record" >&2
    exit 1
fi
