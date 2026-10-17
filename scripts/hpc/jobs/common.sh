#!/usr/bin/env bash
# Shared part of the job scripts. remote-test.sh appends it to the rendered
# jobs/<target>.sh, after the scheduler lines and the job settings:
#   GRIDCL_HPC_TARGET, GRIDCL_HPC_SCHEDULER (sge | direct), GRIDCL_HPC_TREE, GRIDCL_HPC_BRANCH, GRIDCL_HPC_COMMIT, GRIDCL_HPC_DIRTY,
#   GRIDCL_HPC_COMMAND, GRIDCL_HPC_BUILD (may be empty), GRIDCL_HPC_MODULES,
#   GRIDCL_HPC_CUDA_PATH (may be empty), and the exported CARGO_TARGET_DIR,
#   CARGO_NET_OFFLINE=true, GRIDCL_TEST_DEVICE=cuda and, for --ranks N, GRIDCL_RANKS=N.
# JOB_ID comes from the scheduler, or from the runner on direct targets.
#
# The job runs the module lines, exports CUDA_PATH, records the prologue, builds (when
# GRIDCL_HPC_BUILD is set), runs the command in the tree with bash, and writes
# <tree>/results/<JOB_ID>/:
#   prologue.txt   host, date, slots, module lines, nvidia-smi, CUDA toolkit, internet
#                  access, toolchain versions
#   times.env      start, build_start, build_end, build_status, run_start, end (epoch s)
#   output.log     stdout and stderr of the build and the command
#   exit_status    the command's exit status (the build's, when the build failed)
#   node-load-before.txt, node-load-after.txt
#                  direct targets only (ADR-0002): node-load snapshots taken immediately
#                  before and after the command; the last line is node_load=idle or
#                  node_load=busy: <reasons>
# submit.env (written at submission) and sge.log (moved in afterwards), or job.sh,
# direct.log and direct_status on direct targets, complete it.
# The job itself always exits 0: SGE treats the exit statuses 99 (requeue) and 100 (error
# state) specially, so the command's status is passed on only through exit_status.

gridcl_job_prologue() {
    echo "== nvidia-smi"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi
        echo "== nvidia-smi --query-gpu"
        nvidia-smi --query-gpu=index,name,compute_cap,driver_version,memory.total --format=csv
    else
        echo "nvidia-smi: not found"
    fi
    echo "== CUDA toolkit"
    echo "CUDA_PATH: ${CUDA_PATH:-}"
    if [ -n "${CUDA_PATH:-}" ] && [ -f "$CUDA_PATH/version.json" ]; then
        cat "$CUDA_PATH/version.json"
    elif command -v nvcc >/dev/null 2>&1; then
        nvcc --version
    else
        echo "toolkit: not found"
    fi
    echo "== internet access"
    if ! command -v curl >/dev/null 2>&1; then
        echo "internet: unknown (no curl)"
    elif curl -sS -o /dev/null --connect-timeout 5 --max-time 10 https://index.crates.io/config.json; then
        echo "internet: yes"
    else
        echo "internet: no"
    fi
    echo "== toolchain"
    rustc --version
    cargo --version
    echo "== modules"
    if type module >/dev/null 2>&1; then
        module list
    else
        echo "module: not available"
    fi
}

# gridcl_job_cpu_others: CPU cores used by other users' processes (root included), sampled
# over 1 s from /proc on Linux; elsewhere the sum of ps's %cpu. Empty when unknown.
gridcl_job_cpu_others() {
    local uid hz
    uid=$(id -u)
    if [ -r /proc/self/stat ]; then
        hz=$(getconf CLK_TCK 2>/dev/null)
        {
            echo "@A"
            cat /proc/[0-9]*/stat 2>/dev/null
            sleep 1
            echo "@B"
            cat /proc/[0-9]*/stat 2>/dev/null
            echo "@U"
            ps -e -o uid=,pid= 2>/dev/null
        } | awk -v me="$uid" -v hz="${hz:-100}" '
            /^@[ABU]$/ { sec = $0; next }
            sec == "@U" { owner[$2] = $1; owners++; next }
            {
                pid = $1 + 0
                line = $0
                sub(/^.*\) /, "", line) # fields after "pid (comm) "
                split(line, f, " ")
                t = f[12] + f[13]        # utime + stime, in clock ticks
                if (sec == "@A") a[pid] = t; else b[pid] = t
            }
            END {
                for (p in b) if ((p in a) && (p in owner) && owner[p] != me) s += b[p] - a[p]
                if (owners == 0) exit 1
                printf "%.2f\n", s / hz
            }'
    else
        ps -A -o uid=,pcpu= 2>/dev/null |
            awk -v me="$uid" '$1 != me { s += $2; n++ } END { if (n == 0) exit 1; printf "%.2f\n", s / 100 }'
    fi
}

# gridcl_job_node_load <file>: a node-load snapshot (ADR-0002). The node is idle if no other
# user has a GPU compute process, every GPU's utilisation is below 5%, other users'
# processes use less than one core and the 1-minute load average is below 2; a value that
# cannot be read counts as busy.
gridcl_job_node_load() {
    local file=$1 uid reasons="" out idx util pid rest owner load others gpus=0
    uid=$(id -u)
    : >"$file"
    {
        echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "user: $(id -un) (uid $uid)"
        echo "== GPUs: index, utilisation (%), memory used (MiB), memory total (MiB)"
        if ! command -v nvidia-smi >/dev/null 2>&1; then
            echo "nvidia-smi: not found"
            reasons="GPU load unknown (no nvidia-smi)"
        elif ! out=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total \
            --format=csv,noheader,nounits 2>&1); then
            printf '%s\n' "$out"
            reasons="GPU utilisation unknown"
        else
            printf '%s\n' "$out"
            while IFS=', ' read -r idx util rest; do
                case $idx in
                    '' | *[!0-9]*) continue ;;
                esac
                gpus=$((gpus + 1))
                case $util in
                    '' | *[!0-9]*) reasons="${reasons:+$reasons, }GPU $idx utilisation unknown" ;;
                    *) [ "$util" -lt 5 ] || reasons="${reasons:+$reasons, }GPU $idx utilisation $util%" ;;
                esac
            done <<<"$out"
            [ "$gpus" -gt 0 ] || reasons="${reasons:+$reasons, }GPU utilisation unknown"
            echo "== GPU compute processes: pid, memory used (MiB), name, owner"
            if ! out=$(nvidia-smi --query-compute-apps=pid,used_memory,process_name \
                --format=csv,noheader,nounits 2>&1); then
                printf '%s\n' "$out"
                reasons="${reasons:+$reasons, }GPU processes unknown"
            else
                while IFS=', ' read -r pid rest; do
                    case $pid in
                        '' | *[!0-9]*) continue ;;
                    esac
                    owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
                    echo "$pid, $rest, uid ${owner:-unknown}"
                    [ "$owner" = "$uid" ] ||
                        reasons="${reasons:+$reasons, }GPU process $pid of uid ${owner:-unknown}"
                done <<<"$out"
            fi
        fi
        if [ -r /proc/loadavg ]; then
            load=$(cut -d ' ' -f 1 /proc/loadavg)
        else
            load=$(sysctl -n vm.loadavg 2>/dev/null | tr -d '{}' | awk '{ print $1 }')
        fi
        echo "== load average (1 min): ${load:-unknown}"
        case $load in
            '' | *[!0-9.]*) reasons="${reasons:+$reasons, }load average unknown" ;;
            *) awk -v l="$load" 'BEGIN { exit !(l < 2) }' || reasons="${reasons:+$reasons, }load average $load" ;;
        esac
        others=$(gridcl_job_cpu_others)
        echo "== CPU cores used by other users' processes: ${others:-unknown}"
        case $others in
            '') reasons="${reasons:+$reasons, }other users' CPU use unknown" ;;
            *) awk -v c="$others" 'BEGIN { exit !(c < 1) }' ||
                reasons="${reasons:+$reasons, }other users' processes use $others cores" ;;
        esac
        if [ -z "$reasons" ]; then
            echo "node_load=idle"
        else
            echo "node_load=busy: $reasons"
        fi
    } >>"$file" 2>&1 </dev/null
}

gridcl_job_main() {
    local results times log status=0
    if [ -z "${JOB_ID:-}" ] || [ -z "${GRIDCL_HPC_TREE:-}" ]; then
        echo "common.sh: JOB_ID or GRIDCL_HPC_TREE not set; submit with scripts/hpc/remote-test.sh" >&2
        return 1
    fi
    results=$GRIDCL_HPC_TREE/results/$JOB_ID
    times=$results/times.env
    log=$results/output.log
    mkdir -p "$results" || return 1
    echo "start=$(date +%s)" >"$times"
    cd "$GRIDCL_HPC_TREE" || return 1

    {
        echo "host: $(hostname)"
        echo "date: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
        echo "target: ${GRIDCL_HPC_TARGET:-}"
        echo "job id: $JOB_ID"
        echo "commit: ${GRIDCL_HPC_COMMIT:-} (${GRIDCL_HPC_BRANCH:-}), dirty: ${GRIDCL_HPC_DIRTY:-}"
        echo "slots: ${NSLOTS:-unknown}; ranks: ${GRIDCL_RANKS:-none}"
        echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-}"
        echo "== module lines"
        printf '%s\n' "${GRIDCL_HPC_MODULES:-}"
        eval "${GRIDCL_HPC_MODULES:-}"
        echo "module lines exit status: $?"
    } >"$results/prologue.txt" 2>&1 </dev/null
    if [ -n "${GRIDCL_HPC_CUDA_PATH:-}" ]; then
        CUDA_PATH=$GRIDCL_HPC_CUDA_PATH
        export CUDA_PATH
        if [ -d "$CUDA_PATH/bin" ]; then PATH=$CUDA_PATH/bin:$PATH; fi
    fi
    if [ -d "$HOME/.cargo/bin" ]; then PATH=$HOME/.cargo/bin:$PATH; fi
    export PATH
    gridcl_job_prologue >>"$results/prologue.txt" 2>&1 </dev/null

    : >"$log"
    if [ -n "${GRIDCL_HPC_BUILD:-}" ]; then
        echo "build_start=$(date +%s)" >>"$times"
        bash -c "$GRIDCL_HPC_BUILD" >>"$log" 2>&1 </dev/null
        status=$?
        echo "build_end=$(date +%s)" >>"$times"
        echo "build_status=$status" >>"$times"
    fi
    if [ "$status" = 0 ]; then
        if [ "${GRIDCL_HPC_SCHEDULER:-}" = direct ]; then
            gridcl_job_node_load "$results/node-load-before.txt"
        fi
        echo "run_start=$(date +%s)" >>"$times"
        bash -c "$GRIDCL_HPC_COMMAND" >>"$log" 2>&1 </dev/null
        status=$?
        echo "end=$(date +%s)" >>"$times"
        if [ "${GRIDCL_HPC_SCHEDULER:-}" = direct ]; then
            gridcl_job_node_load "$results/node-load-after.txt"
        fi
    else
        echo "end=$(date +%s)" >>"$times"
    fi
    echo "$status" >"$results/exit_status"
}

gridcl_job_main || echo "common.sh: job setup failed" >&2
exit 0
