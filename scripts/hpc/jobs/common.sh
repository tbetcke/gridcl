#!/usr/bin/env bash
# Shared part of the job scripts. remote-test.sh appends it to the rendered
# jobs/<target>.sh, after the scheduler lines and the job settings:
#   GRIDCL_HPC_TARGET, GRIDCL_HPC_TREE, GRIDCL_HPC_BRANCH, GRIDCL_HPC_COMMIT, GRIDCL_HPC_DIRTY,
#   GRIDCL_HPC_COMMAND, GRIDCL_HPC_BUILD (may be empty), GRIDCL_HPC_MODULES,
#   GRIDCL_HPC_CUDA_PATH (may be empty), and the exported CARGO_TARGET_DIR,
#   CARGO_NET_OFFLINE=true, GRIDCL_TEST_DEVICE=cuda and, for --ranks N, GRIDCL_RANKS=N.
# JOB_ID comes from the scheduler.
#
# The job runs the module lines, exports CUDA_PATH, records the prologue, builds (when
# GRIDCL_HPC_BUILD is set), runs the command in the tree with bash, and writes
# <tree>/results/<JOB_ID>/:
#   prologue.txt   host, date, slots, module lines, nvidia-smi, CUDA toolkit, internet
#                  access, toolchain versions
#   times.env      start, build_start, build_end, build_status, run_start, end (epoch s)
#   output.log     stdout and stderr of the build and the command
#   exit_status    the command's exit status (the build's, when the build failed)
# submit.env (written at submission) and sge.log (moved in afterwards) complete it.
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
        echo "run_start=$(date +%s)" >>"$times"
        bash -c "$GRIDCL_HPC_COMMAND" >>"$log" 2>&1 </dev/null
        status=$?
    fi
    echo "end=$(date +%s)" >>"$times"
    echo "$status" >"$results/exit_status"
}

gridcl_job_main || echo "common.sh: job setup failed" >&2
exit 0
