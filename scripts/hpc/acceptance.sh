#!/usr/bin/env bash
# usage: scripts/hpc/acceptance.sh <target> [step ...]
# Live E0 acceptance driver for a remote target (docs/plans/E0.md, T7-T9). Runs the steps
# through remote-test.sh, never more than one job at a time:
#   turnaround  env-check x3 on one GPU (--expect-sm, --expect-cuda-major); queue wait and
#               run time of each, and their medians
#   nvrtc       env-check with the cuda-cpp feature (NVRTC path), plus the CPU runtime
#   gpus2       --gpus 2 --ranks 2: mpi_allreduce_of_rank_ids, one GPU per rank
#   fail        a deliberately failing command: exit status 3 and its error as first failure
#   max-wait    a --max-wait overrun: deleted, exit status 124
# then checks that status.sh lists no job, and prints the rows for docs/dev-environment.md.
# Without steps, all of them run. Exit status 0 iff every step met its expectation.
#
# Summaries and the rows: target/hpc/<target>/acceptance-<UTC time>/.
# Environment: GRIDCL_ACCEPT_CUDA_MAJOR overrides the expected CUDA major version (empty: no
# check); GRIDCL_ACCEPT_MAX_WAIT the --max-wait of the regular steps (default 04:00:00).
# shellcheck source-path=SCRIPTDIR
set -u
set -o pipefail

HPC_DIR=$(cd "$(dirname "$0")" && pwd)
HPC_ROOT=$(cd "$HPC_DIR/../.." && pwd)
ALL_STEPS="turnaround nvrtc gpus2 fail max-wait"
MANIFEST=tools/env-check/Cargo.toml

usage() {
    sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

msg() { echo "acceptance.sh: $*" >&2; }

# Per-target expectations: compute capability, CUDA major version, features of the main and
# the NVRTC runs, GPUs of the MPI step.
target_settings() {
    case $1 in
        myriad-v100)
            SM=70 CUDA_MAJOR=12 FEATURES=cuda NVRTC_FEATURES=cuda-cpp,cpu MPI_GPUS=2
            ;;
        myriad-a100)
            # Myriad's newest CUDA module is 12.2.2 (no CUDA 13; docs/dev-environment.md).
            SM=80 CUDA_MAJOR=12 FEATURES=cuda NVRTC_FEATURES=cuda-cpp,cpu MPI_GPUS=2
            ;;
        locust)
            SM=90 CUDA_MAJOR="" FEATURES=cuda,cpu NVRTC_FEATURES=cuda-cpp MPI_GPUS=1
            ;;
        *) return 1 ;;
    esac
    CUDA_MAJOR=${GRIDCL_ACCEPT_CUDA_MAJOR-$CUDA_MAJOR}
    EXPECT="--expect-sm $SM${CUDA_MAJOR:+ --expect-cuda-major $CUDA_MAJOR}"
}

# run_step <name> <expected status> <runner options and command ...>: one remote-test.sh
# run; its output goes to $OUT/<name>.txt, its job id to $OUT/<name>.id. Returns 0 iff the
# runner's exit status is the expected one.
run_step() {
    local name=$1 want=$2 status id
    shift 2
    msg "step $name: remote-test.sh $TARGET $*"
    "$HPC_DIR/remote-test.sh" "$TARGET" "$@" >"$OUT/$name.txt" 2>"$OUT/$name.err"
    status=$?
    id=$(sed -n 's/^job id: //p' "$OUT/$name.txt" | head -n 1)
    printf '%s\n' "$id" >"$OUT/$name.id"
    printf '%s\n' "$status" >"$OUT/$name.status"
    msg "step $name: job ${id:-none}, exit status $status (expected $want)"
    if [ "$status" != "$want" ]; then
        sed -n '1,40p' "$OUT/$name.txt" >&2
        tail -n 5 "$OUT/$name.err" >&2
        return 1
    fi
}

step_turnaround() {
    local i rc=0
    for i in 1 2 3; do
        # A plain cargo command, so the job records build and command times separately.
        # shellcheck disable=SC2086 # EXPECT is a list of options
        run_step "turnaround-$i" 0 --max-wait "$MAX_WAIT" -- \
            cargo run --release --manifest-path "$MANIFEST" --features "$FEATURES" -- $EXPECT || rc=1
    done
    return "$rc"
}

step_nvrtc() {
    # A new feature set builds most of CubeCL again: 8 cores and a longer limit.
    # shellcheck disable=SC2086
    run_step nvrtc 0 --cores 8 --time 02:00:00 --max-wait "$MAX_WAIT" -- \
        cargo run --release --manifest-path "$MANIFEST" --features "$NVRTC_FEATURES" -- $EXPECT
}

step_gpus2() {
    # The per-rank reports are copied into the job's results for the record (MPI version,
    # CUDA-awareness). $JOB_ID and $CARGO_TARGET_DIR expand in the job.
    # shellcheck disable=SC2016
    run_step gpus2 0 --gpus "$MPI_GPUS" --ranks 2 --cores 8 --time 02:00:00 --max-wait "$MAX_WAIT" -- \
        'cargo test --release --manifest-path '"$MANIFEST"' --features cuda,mpi --test acceptance -- mpi_allreduce_of_rank_ids;' \
        's=$?; cp "$CARGO_TARGET_DIR"/tmp/mpi_allreduce_of_rank_ids/rank-*.json "results/$JOB_ID/" 2>/dev/null; exit $s'
}

step_fail() {
    local first
    run_step fail 3 --time 00:10:00 --max-wait "$MAX_WAIT" -- \
        'echo "acceptance.sh: this command fails on purpose"; echo "error: deliberate failure (acceptance.sh)"; exit 3' ||
        return 1
    first=$(sed -n 's/^first failure: //p' "$OUT/fail.txt")
    case $first in
        "error: deliberate failure (acceptance.sh)") ;;
        *)
            msg "step fail: first failure is '$first', not the command's error"
            return 1
            ;;
    esac
}

step_max_wait() {
    run_step max-wait 124 --time 00:15:00 --max-wait 00:02:00 -- 'sleep 600'
}

# status_empty: status.sh lists no job; a deleted job may linger in qstat for a while.
status_empty() {
    local i out
    for i in 1 2 3 4 5 6; do
        if out=$("$HPC_DIR/status.sh" 2>"$OUT/status.err") && [ -z "$out" ]; then
            return 0
        fi
        [ "$i" -lt 6 ] && sleep 30
    done
    printf '%s\n' "$out" >"$OUT/status.txt"
    msg "status.sh still lists jobs (or failed):"
    printf '%s\n' "$out" >&2
    return 1
}

# print_rows: the rows of docs/dev-environment.md, from the summaries and job results.
print_rows() {
    python3 -I - "$HPC_ROOT/target/hpc/$TARGET" "$OUT" "$TARGET" <<'EOF'
import json, os, re, statistics, sys

results, out, target = sys.argv[1:4]

def text(path):
    try:
        with open(path) as f:
            return f.read()
    except OSError:
        return ""

def job(step):
    return text(os.path.join(out, step + ".id")).strip()

def summary(step, key):
    m = re.search("^" + re.escape(key) + ": (.*)$", text(os.path.join(out, step + ".txt")), re.M)
    return m.group(1) if m else ""

def report_from_log(step):
    """The env-check JSON report printed on stdout (from a line '{' to a line '}')."""
    log = text(os.path.join(results, job(step), "output.log"))
    m = re.search(r"^\{\n.*?^\}$", log, re.M | re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except ValueError:
        return None

def seconds(value):
    m = re.match(r"(\d+) s", value)
    return int(m.group(1)) if m else None

def prologue(step, section):
    lines = text(os.path.join(results, job(step), "prologue.txt")).splitlines()
    try:
        i = lines.index("== " + section)
    except ValueError:
        return []
    body = []
    for line in lines[i + 1:]:
        if line.startswith("== "):
            break
        body.append(line)
    return body

rows = []
def row(item, value):
    rows.append("| %s | %s |" % (item, value))

main = report_from_log("turnaround-1") or report_from_log("turnaround-2") or report_from_log("turnaround-3")
nvrtc = report_from_log("nvrtc")
if main:
    host, cuda = main["host"], main.get("cuda") or {}
    gpu = next((d for d in main["devices"] if d["runtime"] == "cuda"), None)
    row("OS and architecture", "%s, %s" % (host["os_version"], host["arch"]))
    row("Rust toolchain", host["rustc_version"])
    modules = prologue("turnaround-1", "module lines")
    row("Module lines for jobs", "`%s`" % modules[0] if modules else "?")
    row("CUDA toolkit and `CUDA_PATH`", "%s (%s), `%s`" % (cuda.get("toolkit_version"), cuda.get("toolkit_source"), cuda.get("cuda_path")))
    row("Driver version and its CUDA version (`nvidia-smi`)", "%s, CUDA %s; toolkit not newer: %s" % (cuda.get("driver_version"), cuda.get("driver_cuda_version"), cuda.get("toolkit_not_newer_than_driver")))
    if gpu:
        row("Compute capability (env-check)", "%s (`%s`; nvidia-smi %s)" % (", ".join(cuda.get("compute_capability") or []), gpu["fingerprint"], ", ".join(cuda.get("nvidia_smi_compute_capability") or [])))
        checks = ", ".join("%s %s x%d %s %s ms" % (c["kernel"], c["precision"], c["vector_size"], c["status"], c["millis"]) for c in gpu["checks"])
        row("f64 and u64 support (env-check)", "f64 %s, u64 %s, u32 atomic add %s; %s" % (gpu["supports_f64"], gpu["supports_u64"], gpu["atomic_u32_add"], checks))
        row("`max_page_size`, `max_bindings`", "%d B, %d; plane size %d-%d; IO-optimized sizes f32 %s, f64 %s" % (gpu["max_page_size"], gpu["max_bindings"], gpu["plane_size_min"], gpu["plane_size_max"], gpu["io_vector_sizes_f32"], gpu["io_vector_sizes_f64"]))
    simd = [f for f in host["simd_flags"] if f.startswith(("avx", "sve", "asimd", "fma"))]
    row("CPU model and SIMD extensions", "%s; %s" % (host["cpu_model"], " ".join(simd)))
    internet = [l for l in prologue("turnaround-1", "internet access") if l.startswith("internet:")]
    row("Internet access on compute nodes", internet[0] if internet else "?")
if nvrtc:
    cpu = next((d for d in nvrtc["devices"] if d["runtime"] == "cpu"), None)
    cuda = nvrtc.get("cuda") or {}
    gpu = next((d for d in nvrtc["devices"] if d["runtime"] == "cuda"), None)
    if gpu:
        row("NVRTC path (`cuda-cpp`)", "compiler %s, passed %s" % (cuda.get("compiler"), gpu["passed"]))
    if cpu:
        checks = ", ".join("%s %s x%d %s %s ms" % (c["kernel"], c["precision"], c["vector_size"], c["status"], c["millis"]) for c in cpu["checks"])
        row("CPU runtime: `load_width`, vector registers, hardware threads", "%s bits, %s, %s (job slots); plane size %d; IO-optimized sizes f32 %s, f64 %s; %s" % (cpu["load_width"], cpu["vector_register_count"], nvrtc["host"]["hardware_threads"], cpu["plane_size_min"], cpu["io_vector_sizes_f32"], cpu["io_vector_sizes_f64"], checks))
ranks = []
mpi_dir = os.path.join(results, job("gpus2")) if job("gpus2") else ""
for name in sorted(os.listdir(mpi_dir)) if os.path.isdir(mpi_dir) else []:
    if re.match(r"rank-\d+\.json$", name):
        ranks.append(json.loads(text(os.path.join(mpi_dir, name))))
if ranks:
    mpi = ranks[0]["mpi"]
    devices = ["rank %d: %s %d (%s)" % (r["mpi"]["rank"], d["runtime"], d["index"], "pass" if d["passed"] else "FAIL") for r in ranks for d in r["devices"]]
    row("MPI implementation and version", "%s; %d-rank allreduce %s; %s" % (mpi["library_version"].splitlines()[0], mpi["size"], "passes" if all(r["mpi"]["allreduce_passed"] for r in ranks) else "FAILS", "; ".join(devices)))
    row("CUDA-aware MPI available", "%s (%s)" % (mpi["cuda_aware"], mpi["cuda_aware_method"]))
builds = []
for step in ("turnaround-1", "turnaround-2", "turnaround-3", "nvrtc", "gpus2"):
    run = summary(step, "run time")
    if run:
        builds.append("%s: %s" % (step, run))
row("Build and run times per step", "; ".join(builds))
queue = [seconds(summary("turnaround-%d" % i, "queue wait")) for i in (1, 2, 3)]
run = [seconds(summary("turnaround-%d" % i, "run time")) for i in (1, 2, 3)]
if all(v is not None for v in queue + run):
    row("Queue wait and run time (median of 3)", "%d s and %d s (queue %s s, run %s s; jobs %s)" % (statistics.median(queue), statistics.median(run), queue, run, ", ".join(job("turnaround-%d" % i) for i in (1, 2, 3))))
for step in ("fail", "max-wait"):
    if job(step) or os.path.exists(os.path.join(out, step + ".status")):
        row("Runner step " + step, "job %s, exit status %s; first failure: %s" % (job(step) or "none", text(os.path.join(out, step + ".status")).strip(), summary(step, "first failure")))
print("| Item | %s |" % target)
print("| --- | --- |")
print("\n".join(rows))
EOF
}

main() {
    local steps step rc=0 failed=""
    case ${1:-} in
        '' | -h | --help)
            usage
            [ $# -gt 0 ] && return 0
            return 2
            ;;
    esac
    TARGET=$1
    shift
    target_settings "$TARGET" || {
        msg "unknown target '$TARGET'"
        return 2
    }
    steps=${*:-$ALL_STEPS}
    for step in $steps; do
        case " $ALL_STEPS " in
            *" $step "*) ;;
            *)
                msg "unknown step '$step' (steps: $ALL_STEPS)"
                return 2
                ;;
        esac
    done
    MAX_WAIT=${GRIDCL_ACCEPT_MAX_WAIT:-04:00:00}
    OUT=$HPC_ROOT/target/hpc/$TARGET/acceptance-$(date -u +%Y%m%dT%H%M%SZ)
    mkdir -p "$OUT" || return 125
    msg "target $TARGET, steps: $steps; summaries in ${OUT#"$HPC_ROOT"/}/"

    for step in $steps; do
        if ! "step_${step//-/_}"; then
            failed="$failed $step"
            rc=1
        fi
    done
    if ! status_empty; then
        failed="$failed status"
        rc=1
    fi
    print_rows | tee "$OUT/rows.md"
    if [ "$rc" -eq 0 ]; then
        msg "passed: $steps; status.sh empty"
    else
        msg "FAILED:$failed"
    fi
    return "$rc"
}

main "$@"
