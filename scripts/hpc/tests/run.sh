#!/usr/bin/env bash
# Tests of the remote-test runner (scripts/hpc) against fake ssh, rsync, SGE and Slurm.
#
# usage: scripts/hpc/tests/run.sh [sge | direct | slurm | <case>...]
#   sge      the SGE-path cases (E0 T3)
#   direct   the direct-kind, status, cancel and pruning cases (E0 T4)
#   slurm    the Slurm-path cases of the kathleen target (M3s T19, ADR-0016)
#   default  every case
# KEEP=1 keeps the scratch directories; CASE_TIMEOUT (seconds, default 120) limits a case.
#
# Each case gets a scratch directory with
#   local/        a git repository holding a copy of scripts/hpc: the working tree to sync
#   remote/       the fake remote file system: the base directories of the targets
#   state/        the fake schedulers (state/sge/, state/slurm/) and the logs of the shims
#   targets.env   personal settings, passed to the runner through GRIDCL_TARGETS_ENV
# fake-bin/ stands in for ssh (runs the remote command locally), rsync (drops the host
# part), qsub, qstat, qdel, sbatch, squeue and scancel; fake-bin/remote/ holds the commands of the remote nodes
# (cargo, rustup, rustc, nvidia-smi, curl), which log to state/. GRIDCL_HPC_POLL sets the
# runner's polling interval in seconds.
#
# Conventions the cases rely on (docs/design.md, "Development environment and test
# targets", and docs/plans/E0.md):
#   - the summary is printed on stdout, at most 40 lines, starting with the lines
#     "target:", "job id:", "commit:", "dirty:", "command:", "exit status:",
#     "queue wait:", "run time:" and "first failure:"; progress goes to stderr;
#   - the tree of branch B is <BASE>/<B with '/' replaced by '_'>, results of job <id>
#     are in <tree>/results/<id>/ and are copied to target/hpc/<target>/<id>/ locally;
#   - a direct job's id is <timestamp>-<pid>, and its remote pid file is named <id>.pid;
#   - a direct job writes node-load-before.txt and node-load-after.txt, each ending in a
#     line node_load=idle or node_load=busy: <reasons>, and its summary has a line
#     "node load: before ...; after ..." (ADR-0002);
#   - status.sh prints nothing when no job is queued or running.
set -u

here=$(cd "$(dirname "$0")" && pwd)
hpc=$(cd "$here/.." && pwd)
repo=$(cd "$hpc/../.." && pwd)
orig_path=$PATH

SGE_CASES="usage missing_targets_env job_script_sge_lines exit_status_passthrough
failing_command_first_error summary_header_and_length max_wait_deletes_job results_copied
results_survive_resync busy_tree_refused cubecl_toml_written myriad_cpu_job_script
myriad_cpu_rejects_gpus"
DIRECT_CASES="status_lists_jobs cancel_deletes_job direct_remote_timeout prune_merged_branch_trees"
SLURM_CASES="kathleen_job_script kathleen_two_nodes kathleen_rejects kathleen_max_wait_cancels
kathleen_status_and_cancel cancel_only_own_jobs"
CASE_TIMEOUT=${CASE_TIMEOUT:-120}

# --- assertions -------------------------------------------------------------------

failed=0
fail() {
    echo "    FAIL: $*"
    failed=1
}

expect_status() { # expect_status <want> <what>
    [ "$RC" = "$1" ] || fail "$2: exit status $RC, want $1"
}

expect_nonzero() { # expect_nonzero <what>
    [ "$RC" != 0 ] || fail "$1: exit status 0, want non-zero"
}

expect_grep() { # expect_grep <file> <extended regex> <what>
    grep -Eq -- "$2" "$1" 2>/dev/null || fail "$3: /$2/ not in $(basename "$1")"
}

expect_no_grep() { # expect_no_grep <file> <extended regex> <what>
    if grep -Eq -- "$2" "$1" 2>/dev/null; then fail "$3: /$2/ found in $(basename "$1")"; fi
}

expect_line() { # expect_line <file> <exact line> <what>
    grep -Fxq -- "$2" "$1" 2>/dev/null || fail "$3: line '$2' not in $(basename "$1")"
}

expect_exists() { # expect_exists <path> <what>
    [ -e "$1" ] || fail "$2: $1 does not exist"
}

expect_absent() { # expect_absent <path> <what>
    [ ! -e "$1" ] || fail "$2: $1 exists"
}

# --- fixture ----------------------------------------------------------------------

g() {
    git -C "$LOCAL" -c commit.gpgsign=false -c core.hooksPath=/dev/null "$@"
}

commit_file() { # commit_file <path> <content>
    echo "$2" >"$LOCAL/$1"
    g add "$1"
    g commit -q -m "add $1"
}

write_targets_env() {
    {
        for t in MYRIAD_V100 MYRIAD_A100 MYRIAD_CPU; do
            echo "${t}_HOST=fake-myriad"
            echo "${t}_SCHEDULER=sge"
            echo "${t}_BASE='$REMOTE/myriad'"
            echo "${t}_LOGIN_MODULES=:"
            echo "${t}_MODULES=:"
            echo "${t}_CORES_REQUEST='-pe smp {cores}'"
            echo "${t}_MEM_REQUEST='-l mem=4G'"
        done
        echo "MYRIAD_V100_CUDA_PATH='$REMOTE/cuda-12'"
        echo "MYRIAD_A100_CUDA_PATH='$REMOTE/cuda-13'"
        echo "MYRIAD_CPU_CUDA_PATH="
        echo "LOCUST_HOST=fake-locust"
        echo "LOCUST_SCHEDULER=direct"
        echo "LOCUST_BASE='$REMOTE/locust'"
        echo "LOCUST_LOGIN_MODULES=:"
        echo "LOCUST_MODULES=:"
        echo "LOCUST_CUDA_PATH='$REMOTE/cuda-12'"
        echo "LOCUST_CORES_REQUEST="
        echo "LOCUST_MEM_REQUEST="
        echo "KATHLEEN_HOST=fake-kathleen"
        echo "KATHLEEN_SCHEDULER=slurm"
        echo "KATHLEEN_BASE='$REMOTE/kathleen'"
        echo "KATHLEEN_LOGIN_MODULES=:"
        echo "KATHLEEN_MODULES=:"
        echo "KATHLEEN_CUDA_PATH="
        echo "KATHLEEN_CORES_REQUEST="
        echo "KATHLEEN_MEM_REQUEST="
    } >"$T/targets.env"
}

setup() {
    T=$(mktemp -d "${TMPDIR:-/tmp}/hpc-tests.XXXXXX")
    LOCAL=$T/local
    REMOTE=$T/remote
    FAKE_STATE=$T/state
    mkdir -p "$LOCAL/scripts" "$REMOTE" "$FAKE_STATE"
    rsync -a --exclude tests/ --exclude targets.env "$hpc/" "$LOCAL/scripts/hpc/"
    cp "$repo/.gitignore" "$repo/rust-toolchain.toml" "$LOCAL/"
    echo "fixture" >"$LOCAL/README.md"
    export GIT_AUTHOR_NAME=hpc-tests GIT_AUTHOR_EMAIL=hpc-tests@example.invalid
    export GIT_COMMITTER_NAME=hpc-tests GIT_COMMITTER_EMAIL=hpc-tests@example.invalid
    git init -q -b main "$LOCAL"
    g add -A
    g commit -q -m init
    g checkout -q -b dev/e0-test
    TREE=$REMOTE/myriad/dev_e0-test
    write_targets_env
    export FAKE_STATE GRIDCL_TARGETS_ENV=$T/targets.env GRIDCL_HPC_POLL=0.2
    export PATH="$here/fake-bin:$orig_path"
    unset FAKE_SGE_HOLD FAKE_SGE_QUEUE_DELAY FAKE_SSH_UNREACHABLE FAKE_SLURM_HOLD FAKE_SLURM_QUEUE_DELAY
    RT=$LOCAL/scripts/hpc/remote-test.sh
    STATUS=$LOCAL/scripts/hpc/status.sh
    CANCEL=$LOCAL/scripts/hpc/cancel.sh
}

teardown() {
    local pid
    # shellcheck disable=SC2013 # one pid per line
    for pid in $(cat "$T/bg.pids" 2>/dev/null); do
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
    done
    # shellcheck disable=SC2013 # one pid per file
    for pid in $(cat "$FAKE_STATE"/sge/jobs/*/pid 2>/dev/null); do
        pkill -TERM -P "$pid" 2>/dev/null
        kill -TERM "$pid" 2>/dev/null
    done
    # shellcheck disable=SC2013,SC2046 # one pid per file
    for pid in $(cat $(pid_files) /dev/null 2>/dev/null); do
        kill -TERM "$pid" 2>/dev/null # timeout passes it on to the direct job
    done
    if [ -n "${KEEP:-}" ]; then
        echo "    kept $T"
    else
        rm -rf "$T"
    fi
}

# rt <args...>: runs remote-test.sh; sets RC, stdout in $T/stdout, stderr in $T/stderr.
rt() {
    "$RT" "$@" >"$T/stdout" 2>"$T/stderr"
    RC=$?
}

# bg_rt <name> <args...>: runs remote-test.sh in the background; sets BG to its pid.
bg_rt() {
    local name=$1
    shift
    "$RT" "$@" >"$T/$name.stdout" 2>"$T/$name.stderr" &
    BG=$!
    echo "$BG" >>"$T/bg.pids"
}

job_id() { # the job id of the last summary
    sed -n 's/^job id: //p' "$T/stdout" | head -n 1
}

queued_jobs() {
    # shellcheck disable=SC2012 # job ids are numbers
    ls "$FAKE_STATE/sge/jobs" 2>/dev/null | sort -n
}

wait_for_sge_job() { # prints the id of the first queued job; fails after 20 s
    local i=0 id
    while [ $i -lt 100 ]; do
        id=$(queued_jobs | head -n 1)
        if [ -n "$id" ]; then
            echo "$id"
            return 0
        fi
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

pid_files() { # remote pid files of direct jobs
    find "$REMOTE" -name '*.pid' 2>/dev/null
}

wait_for_pid_file() { # prints the first remote pid file; fails after 20 s
    local i=0 f
    while [ $i -lt 100 ]; do
        f=$(pid_files | head -n 1)
        if [ -n "$f" ]; then
            echo "$f"
            return 0
        fi
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

wait_exit() { # wait_exit <pid> <seconds>: waits for a background runner; sets RC (or 999)
    local i=0 limit=$(($2 * 5))
    while kill -0 "$1" 2>/dev/null && [ $i -lt $limit ]; do
        sleep 0.2
        i=$((i + 1))
    done
    if kill -0 "$1" 2>/dev/null; then
        RC=999
    else
        wait "$1"
        RC=$?
    fi
}

no_process() { # no_process <pattern>: no process matches the pattern (after a short grace)
    sleep 1
    ! pgrep -f "$1" >/dev/null 2>&1
}

# --- SGE path (T3) ----------------------------------------------------------------

case_usage() {
    rt
    expect_status 2 "no arguments"
    cat "$T/stdout" "$T/stderr" >"$T/both"
    expect_grep "$T/both" 'targets: myriad-v100 \| myriad-a100 \| myriad-cpu \| locust \| kathleen' "usage lists the targets"
    rt nowhere -- true
    expect_status 2 "unknown target"
    rt myriad-v100 true
    expect_status 2 "command without --"
    rt myriad-v100 --
    expect_status 2 "empty command"
    rt myriad-v100 --gpus two -- true
    expect_status 2 "non-numeric --gpus"
    rt myriad-v100 --time 1h -- true
    expect_status 2 "malformed --time"
    rt myriad-v100 --bogus -- true
    expect_status 2 "unknown option"
    rt locust --gpus 2 -- true
    expect_status 2 "--gpus 2 on locust (one GPU)"
    expect_absent "$FAKE_STATE/ssh.log" "nothing contacted on usage errors"
}

case_missing_targets_env() {
    GRIDCL_TARGETS_ENV=$T/none.env rt myriad-v100 -- true
    expect_nonzero "missing targets.env"
    expect_grep "$T/stderr" 'targets\.example\.env' "message points at targets.example.env"
    grep -v '^MYRIAD_A100_' "$T/targets.env" >"$T/partial.env"
    GRIDCL_TARGETS_ENV=$T/partial.env rt myriad-a100 -- true
    expect_nonzero "target without settings"
    expect_grep "$T/stderr" 'MYRIAD_A100_HOST' "message names the missing key"
    expect_absent "$FAKE_STATE/ssh.log" "nothing contacted without settings"
}

case_job_script_sge_lines() {
    rt myriad-v100 --gpus 2 --ranks 2 --cores 36 --time 01:30:00 -- true
    expect_status 0 "v100 run"
    local s
    s=$FAKE_STATE/sge/submitted/$(job_id).sh
    expect_exists "$s" "rendered job script submitted"
    expect_line "$s" '#$ -l gpu=2' "gpu request"
    expect_line "$s" '#$ -ac allow=EF' "V100 node classes"
    expect_line "$s" '#$ -l h_rt=01:30:00' "wall-clock request"
    expect_line "$s" '#$ -pe smp 36' "cores request from CORES_REQUEST"
    expect_line "$s" '#$ -l mem=4G' "memory request from MEM_REQUEST"
    expect_grep "$s" 'GRIDCL_RANKS=2' "GRIDCL_RANKS for --ranks"
    expect_grep "$s" 'CARGO_NET_OFFLINE=true' "offline cargo"
    expect_grep "$s" 'GRIDCL_TEST_DEVICE=cuda' "test device"
    expect_grep "$s" 'CARGO_TARGET_DIR=' "per-target CARGO_TARGET_DIR"
    expect_grep "$s" "$REMOTE/cuda-12" "CUDA_PATH from targets.env"
    bash -n "$s" || fail "rendered script is not valid bash"
    # Every #$ line comes before the first command, where SGE reads them.
    awk '/^#\$/ { if (cmd) bad = 1 } /^[^#]/ && NF { cmd = 1 } END { exit bad }' "$s" ||
        fail "a #$ line follows a command"

    rt myriad-a100 --gpus 1 -- true
    expect_status 0 "a100 run"
    s=$FAKE_STATE/sge/submitted/$(job_id).sh
    expect_line "$s" '#$ -l gpu=1' "a100 gpu request"
    expect_line "$s" '#$ -ac allow=L' "A100 node class"
    expect_grep "$s" '^#\$ -l h_rt=[0-9]+:[0-9]{2}:[0-9]{2}$' "default wall-clock request"
    expect_grep "$s" "$REMOTE/cuda-13" "a100 CUDA_PATH"

    # Fetch on the login node, then an offline build in the job.
    rt myriad-v100 -- cargo run --release --manifest-path tools/env-check/Cargo.toml --features cuda
    expect_status 0 "cargo command"
    expect_grep "$FAKE_STATE/rustup.log" 'toolchain install' "toolchain installed on the login node"
    expect_grep "$FAKE_STATE/cargo.log" '^offline= .*fetch.*--manifest-path tools/env-check/Cargo.toml' \
        "cargo fetch online for the command's manifest"
    expect_grep "$FAKE_STATE/cargo.log" '^offline= .*args=check --manifest-path tools/env-check/Cargo.toml --features cuda$' \
        "build scripts (LLVM bundle) run on the login node with the command's features"
    expect_grep "$FAKE_STATE/cargo.log" '^offline=true target_dir=[^ ]+ args=run --release' \
        "command runs offline with a target dir"

    # Without a feature option cubecl-llvm is not activated: no check on the login node.
    : >"$FAKE_STATE/cargo.log"
    rt myriad-v100 -- cargo run --release --manifest-path tools/env-check/Cargo.toml -- --features cuda
    expect_status 0 "cargo command without features"
    if grep -Eq '^offline= .*args=(check|build)' "$FAKE_STATE/cargo.log"; then
        fail "login-node check although no feature activates cubecl-llvm"
    fi
}

case_myriad_cpu_job_script() {
    # shellcheck disable=SC2016 # expanded by the job, not here
    rt myriad-cpu --cores 36 --time 00:30:00 -- 'echo "device=$GRIDCL_TEST_DEVICE"'
    expect_status 0 "myriad-cpu run"
    local s id
    id=$(job_id)
    s=$FAKE_STATE/sge/submitted/$id.sh
    expect_exists "$s" "rendered job script submitted"
    expect_no_grep "$s" '^#\$ -l gpu' "no GPU request on a CPU-only node"
    expect_line "$s" '#$ -ac allow=D' "D nodes"
    expect_no_grep "$s" '^#\$ -ac allow=(EF|L)$' "no GPU node class"
    expect_line "$s" '#$ -pe smp 36' "whole node from CORES_REQUEST"
    expect_line "$s" '#$ -l mem=4G' "memory request from MEM_REQUEST"
    expect_line "$s" '#$ -l h_rt=00:30:00' "wall-clock request"
    expect_line "$s" 'export GRIDCL_TEST_DEVICE=cpu' "CPU test device"
    expect_no_grep "$s" 'GRIDCL_TEST_DEVICE=cuda' "no CUDA test device"
    expect_line "$s" "GRIDCL_HPC_CUDA_PATH=''" "no CUDA_PATH"
    expect_line "$s" 'export GRIDCL_HPC_TARGET=myriad-cpu' "target exported for gridcl-bench"
    expect_grep "$s" "CARGO_TARGET_DIR=$TREE/target/myriad-cpu" "own CARGO_TARGET_DIR in the shared tree"
    bash -n "$s" || fail "rendered script is not valid bash"
    awk '/^#\$/ { if (cmd) bad = 1 } /^[^#]/ && NF { cmd = 1 } END { exit bad }' "$s" ||
        fail "a #$ line follows a command"
    expect_grep "$T/stdout" '^target: myriad-cpu$' "summary target"
    expect_no_grep "$T/stdout" '^node load:' "no node-load line on SGE targets"
    local dir=$LOCAL/target/hpc/myriad-cpu/$id
    grep -q 'device=cpu' "$dir/output.log" 2>/dev/null || fail "the command did not see GRIDCL_TEST_DEVICE=cpu"
    expect_grep "$dir/prologue.txt" '^slots: 36;' "prologue records the slots"
    expect_grep "$dir/prologue.txt" '^cpus: nproc ' "prologue records the CPUs"

    # Without --cores the job has one slot; the runner says so.
    rt myriad-cpu -- true
    expect_status 0 "myriad-cpu run without --cores"
    expect_grep "$T/stderr" 'myriad-cpu without --cores' "note about the slots"
    s=$FAKE_STATE/sge/submitted/$(job_id).sh
    expect_no_grep "$s" '^#\$ -pe ' "no cores request"
    expect_line "$s" '#$ -ac allow=D' "D nodes without --cores"
}

case_myriad_cpu_rejects_gpus() {
    rt myriad-cpu --gpus 1 -- true
    expect_status 2 "--gpus 1 on myriad-cpu"
    expect_grep "$T/stderr" 'CPU-only' "message names the CPU-only nodes"
    rt myriad-cpu --cores 36 --gpus=2 -- true
    expect_status 2 "--gpus=2 on myriad-cpu"
    expect_absent "$FAKE_STATE/ssh.log" "nothing contacted on usage errors"
}

case_exit_status_passthrough() {
    rt myriad-v100 -- exit 3
    expect_status 3 "exit 3"
    expect_line "$T/stdout" 'exit status: 3' "summary exit status"
    rt myriad-v100 -- true
    expect_status 0 "true"
}

case_failing_command_first_error() {
    rt myriad-v100 -- 'echo compiling; echo "error[E0425]: first failure here" >&2; echo "error: later failure"; exit 1'
    expect_status 1 "failing command"
    expect_grep "$T/stdout" '^first failure: .*first failure here' "first error in the summary header"
    local first later
    first=$(grep -n 'first failure here' "$T/stdout" | head -n 1 | cut -d: -f1)
    later=$(grep -n 'later failure' "$T/stdout" | head -n 1 | cut -d: -f1)
    if [ -n "$later" ] && [ -n "$first" ] && [ "$later" -lt "$first" ]; then
        fail "a later error is printed before the first one"
    fi
}

case_summary_header_and_length() {
    echo "local change" >>"$LOCAL/README.md"
    # shellcheck disable=SC2016 # expanded by the job, not here
    rt myriad-v100 -- 'for i in $(seq 1 500); do echo "line $i"; done'
    expect_status 0 "long output"
    local lines
    lines=$(wc -l <"$T/stdout")
    [ "$lines" -le 40 ] || fail "summary has $lines lines, at most 40 allowed"
    expect_grep "$T/stdout" '^target: myriad-v100$' "target"
    expect_grep "$T/stdout" '^job id: [0-9]+$' "job id"
    expect_grep "$T/stdout" "^commit: $(g rev-parse --short=7 HEAD)" "commit"
    expect_grep "$T/stdout" '^dirty: yes' "dirty flag set"
    expect_grep "$T/stdout" '^command: .*seq 1 500' "command"
    expect_grep "$T/stdout" '^exit status: 0$' "exit status"
    expect_grep "$T/stdout" '^queue wait: [0-9]' "queue wait"
    expect_grep "$T/stdout" '^run time: [0-9]' "run time"
    expect_grep "$T/stdout" '^first failure: ' "first failure"
    expect_no_grep "$T/stdout" '^node load:' "no node-load line on SGE targets"
    local n key prev=0
    for key in 'target:' 'job id:' 'commit:' 'dirty:' 'command:' 'exit status:' \
        'queue wait:' 'run time:' 'first failure:'; do
        n=$(grep -n "^$key" "$T/stdout" | head -n 1 | cut -d: -f1)
        [ -n "$n" ] || continue
        [ "$n" -gt "$prev" ] || fail "header line '$key' out of order"
        [ "$n" -le 12 ] || fail "header line '$key' is not at the top"
        prev=$n
    done

    g checkout -q -- README.md
    rt myriad-v100 -- true
    expect_grep "$T/stdout" '^dirty: no' "clean tree"
}

case_max_wait_deletes_job() {
    local start=$SECONDS
    FAKE_SGE_HOLD=1 rt myriad-v100 --max-wait 00:00:02 -- true
    expect_status 124 "max-wait overrun"
    [ $((SECONDS - start)) -lt 30 ] || fail "runner waited $((SECONDS - start)) s for a 2 s max-wait"
    expect_exists "$FAKE_STATE/sge/qdel.log" "qdel called"
    [ -z "$(queued_jobs)" ] || fail "jobs left in the queue: $(queued_jobs)"
}

case_results_copied() {
    mkdir -p "$LOCAL/target/debug" "$LOCAL/.venv"
    echo junk >"$LOCAL/target/debug/junk"
    echo junk >"$LOCAL/.venv/junk"
    rt myriad-v100 -- 'echo hello-results'
    expect_status 0 "run"
    local id dir
    id=$(job_id)
    dir=$LOCAL/target/hpc/myriad-v100/$id
    [ -n "$id" ] || fail "no job id in the summary"
    expect_exists "$dir" "local results directory"
    grep -rq hello-results "$dir" 2>/dev/null || fail "command output not in $dir"
    grep -rq 'CUDA Version' "$dir" 2>/dev/null || fail "nvidia-smi record not in $dir"
    [ "$(cat "$dir/exit_status" 2>/dev/null)" = 0 ] || fail "exit_status file is not 0"
    expect_exists "$TREE/results/$id" "remote results directory"
    expect_exists "$TREE/README.md" "tracked file synced"
    expect_exists "$TREE/rust-toolchain.toml" "toolchain file synced"
    expect_absent "$TREE/target/debug/junk" "ignored target/ not synced"
    expect_absent "$TREE/.venv/junk" "ignored .venv/ not synced"
}

case_results_survive_resync() {
    commit_file extra.txt extra
    rt myriad-v100 -- 'echo first-run'
    expect_status 0 "first run"
    local id1
    id1=$(job_id)
    expect_exists "$TREE/extra.txt" "committed file synced"
    g rm -q extra.txt
    g commit -q -m "remove extra.txt"
    rt myriad-v100 -- 'echo second-run'
    expect_status 0 "second run"
    expect_exists "$TREE/results/$id1" "results of the first job survive the second sync"
    grep -rq first-run "$TREE/results/$id1" 2>/dev/null || fail "first job's output lost"
    expect_absent "$TREE/extra.txt" "deleted file removed from the tree"
}

case_cubecl_toml_written() {
    # A local cubecl.toml is git-ignored: the sync neither copies nor deletes the job's own.
    echo 'local = "not synced"' >"$LOCAL/cubecl.toml"
    rt myriad-v100 -- 'echo first'
    expect_status 0 "first run"
    local id
    id=$(job_id)
    expect_line "$TREE/cubecl.toml" '[environment]' "cubecl.toml written by the job"
    expect_line "$TREE/cubecl.toml" 'path = "global"' "database path follows XDG_CACHE_HOME"
    expect_no_grep "$TREE/cubecl.toml" 'not synced' "local cubecl.toml not synced"
    expect_grep "$TREE/results/$id/prologue.txt" '^kernel caches: .*XDG_CACHE_HOME=.*/gridcl-cache/xdg .*cubecl database=.*/gridcl-cache/xdg/cubecl' \
        "prologue records the database location"
    commit_file second.txt second
    rt myriad-v100 -- 'test -f cubecl.toml && echo still-there'
    expect_status 0 "second run"
    grep -rq still-there "$TREE/results/$(job_id)" 2>/dev/null || fail "cubecl.toml deleted by the resync"
}

case_busy_tree_refused() {
    FAKE_SGE_HOLD=1 bg_rt held myriad-v100 --max-wait 00:01:00 -- true
    local id
    id=$(wait_for_sge_job) || { fail "held job never queued"; return; }
    local submitted
    submitted=$(find "$FAKE_STATE/sge/submitted" -type f | wc -l)

    echo "local change" >>"$LOCAL/README.md"
    rt myriad-v100 -- true
    expect_nonzero "sync of a changed tree with a pending job"
    cat "$T/stdout" "$T/stderr" >"$T/both"
    expect_grep "$T/both" '[Pp]ending' "refusal names the pending job"
    [ "$(find "$FAKE_STATE/sge/submitted" -type f | wc -l)" = "$submitted" ] || fail "a job was submitted anyway"
    expect_no_grep "$TREE/README.md" 'local change' "remote sources unchanged"

    g checkout -q -- README.md
    rt myriad-v100 -- true
    expect_status 0 "unchanged tree may be reused while a job is pending"

    qdel "$id" >/dev/null 2>&1
    wait_exit "$BG" 20
}

# --- direct kind, status, cancel, pruning (T4) ------------------------------------

case_status_lists_jobs() {
    "$STATUS" >"$T/status" 2>"$T/status.err"
    RC=$?
    expect_status 0 "status with no jobs"
    [ ! -s "$T/status" ] || fail "status prints something with no jobs: $(head -n 3 "$T/status")"

    FAKE_SGE_HOLD=1 bg_rt held myriad-v100 --max-wait 00:01:00 -- true
    local sge_bg=$BG id
    id=$(wait_for_sge_job) || { fail "held job never queued"; return; }
    bg_rt direct locust --max-wait 00:01:00 -- 'sleep 47'
    local direct_bg=$BG pidfile did
    pidfile=$(wait_for_pid_file) || { fail "direct job never started"; return; }
    did=$(basename "$pidfile" .pid)

    "$STATUS" >"$T/status" 2>"$T/status.err"
    RC=$?
    expect_status 0 "status with jobs"
    expect_grep "$T/status" "(^|[^0-9])$id([^0-9]|$)" "SGE job listed"
    expect_grep "$T/status" "$did" "direct job listed"

    "$CANCEL" "$id" >/dev/null 2>&1
    "$CANCEL" "$did" >/dev/null 2>&1
    wait_exit "$sge_bg" 20
    wait_exit "$direct_bg" 20
    "$STATUS" >"$T/status" 2>"$T/status.err"
    [ ! -s "$T/status" ] || fail "status not empty after cancelling: $(head -n 3 "$T/status")"
}

case_cancel_deletes_job() {
    FAKE_SGE_HOLD=1 bg_rt held myriad-v100 --max-wait 00:01:00 -- true
    local id
    id=$(wait_for_sge_job) || { fail "held job never queued"; return; }
    "$CANCEL" "$id" >"$T/cancel" 2>&1
    RC=$?
    expect_status 0 "cancel SGE job"
    expect_line "$FAKE_STATE/sge/qdel.log" "$id" "qdel called with the id"
    [ -z "$(queued_jobs)" ] || fail "jobs left in the queue: $(queued_jobs)"
    wait_exit "$BG" 20
    expect_nonzero "runner of a cancelled job"
    [ "$RC" != 999 ] || fail "runner still waiting after its job was cancelled"

    bg_rt direct locust --max-wait 00:01:00 -- 'sleep 53'
    local pidfile did
    pidfile=$(wait_for_pid_file) || { fail "direct job never started"; return; }
    did=$(basename "$pidfile" .pid)
    "$CANCEL" "$did" >"$T/cancel" 2>&1
    RC=$?
    expect_status 0 "cancel direct job"
    expect_absent "$pidfile" "pid file removed"
    wait_exit "$BG" 20
    expect_nonzero "runner of a cancelled direct job"
    no_process 'sleep 53' || fail "direct job still running"

    "$CANCEL" 999999 >"$T/cancel" 2>&1
    RC=$?
    expect_nonzero "cancel of an unknown job"
    "$CANCEL" >"$T/cancel" 2>&1
    RC=$?
    expect_nonzero "cancel without a job id"
}

# --- Slurm path (M3s T19) --------------------------------------------------------

slurm_jobs() {
    # shellcheck disable=SC2012 # job ids are numbers
    ls "$FAKE_STATE/slurm/jobs" 2>/dev/null | sort -n
}

wait_for_slurm_job() { # prints the id of the first pending job; fails after 20 s
    local i=0 id
    while [ $i -lt 100 ]; do
        id=$(slurm_jobs | head -n 1)
        if [ -n "$id" ]; then
            echo "$id"
            return 0
        fi
        sleep 0.2
        i=$((i + 1))
    done
    return 1
}

case_kathleen_job_script() {
    local ktree=$REMOTE/kathleen/dev_e0-test
    # shellcheck disable=SC2016 # expanded by the job, not here
    rt kathleen --cores 36 --time 00:30:00 -- \
        'echo "device=$GRIDCL_TEST_DEVICE step=${SLURM_STEP_ID:-unset} tid=$PSM2_TID flags=${GRIDCL_MPIRUN_FLAGS:-none} job=${JOB_ID:-unset}"'
    expect_status 0 "kathleen run"
    expect_grep "$T/stderr" '--cores ignored' "--cores is ignored on whole nodes"
    local s id dir
    id=$(job_id)
    s=$FAKE_STATE/slurm/submitted/$id.sh
    expect_exists "$s" "rendered job script submitted"
    expect_line "$s" '#SBATCH --nodes=1' "one node by default"
    expect_line "$s" '#SBATCH --qos=singlenode' "QoS singlenode for one node"
    expect_line "$s" '#SBATCH --time=00:30:00' "wall-clock request"
    expect_line "$s" '#SBATCH --ntasks-per-node=40' "a slot per core for mpirun"
    expect_line "$s" "#SBATCH --chdir=$ktree" "job starts in the tree"
    expect_line "$s" 'export GRIDCL_TEST_DEVICE=cpu' "CPU test device"
    expect_no_grep "$s" '^#\$' "no SGE lines"
    expect_no_grep "$s" 'GRIDCL_MPIRUN_FLAGS="' "no mapping flags on one node"
    expect_no_grep "$s" '@[A-Z_]+@' "every placeholder replaced"
    bash -n "$s" || fail "rendered script is not valid bash"
    awk '/^#SBATCH/ { if (cmd) bad = 1 } /^[^#]/ && NF { cmd = 1 } END { exit bad }' "$s" ||
        fail "an #SBATCH line follows a command"
    expect_grep "$T/stdout" '^target: kathleen$' "summary target"
    expect_line "$T/stdout" 'exit status: 0' "summary exit status"
    dir=$LOCAL/target/hpc/kathleen/$id
    expect_grep "$dir/output.log" "device=cpu step=unset tid=0 flags=none job=$id" \
        "the command sees the CPU device, no batch step id, PSM2_TID=0 and JOB_ID"
    expect_grep "$dir/prologue.txt" '^slots: 40;' "prologue records the tasks"
    expect_grep "$dir/prologue.txt" '^slurm: qos singlenode, 1 node' "prologue records QoS and nodes"
    expect_exists "$dir/slurm.log" "Slurm output moved into the results"
    expect_absent "$ktree/results/slurm-$id.log" "Slurm output not left in results/"
}

case_kathleen_two_nodes() {
    # shellcheck disable=SC2016 # expanded by the job, not here
    rt kathleen --nodes 2 --ranks 4 -- 'echo "flags=$GRIDCL_MPIRUN_FLAGS ranks=$GRIDCL_RANKS"'
    expect_status 0 "two-node run"
    local s id
    id=$(job_id)
    s=$FAKE_STATE/slurm/submitted/$id.sh
    expect_line "$s" '#SBATCH --nodes=2' "two nodes"
    expect_line "$s" '#SBATCH --qos=test' "QoS test for two nodes up to 1 h"
    grep -q 'flags=--map-by node ranks=4' "$LOCAL/target/hpc/kathleen/$id/output.log" 2>/dev/null ||
        fail "mpirun does not map the ranks by node"
    expect_grep "$LOCAL/target/hpc/kathleen/$id/prologue.txt" '^slots: 80;' "80 tasks on two nodes"
    rt kathleen --nodes=2 --time 02:00:00 -- true
    expect_status 0 "two-node run beyond 1 h"
    expect_line "$FAKE_STATE/slurm/submitted/$(job_id).sh" '#SBATCH --qos=small' "QoS small beyond 1 h"
}

case_kathleen_rejects() {
    rt kathleen --gpus 1 -- true
    expect_status 2 "--gpus on kathleen"
    rt kathleen --nodes 3 -- true
    expect_status 2 "three nodes"
    expect_grep "$T/stderr" 'at most 2 Kathleen nodes' "message names the node limit"
    rt kathleen --ranks 41 -- true
    expect_status 2 "41 ranks on one node"
    rt kathleen --nodes 2 --ranks 81 -- true
    expect_status 2 "81 ranks on two nodes"
    rt kathleen --time 06:00:01 -- true
    expect_status 2 "one node beyond 6 h"
    rt kathleen --nodes 2 --time 48:00:01 -- true
    expect_status 2 "two nodes beyond 48 h"
    rt myriad-cpu --nodes 1 -- true
    expect_status 2 "--nodes on an SGE target"
    expect_absent "$FAKE_STATE/ssh.log" "nothing contacted on usage errors"
}

case_kathleen_max_wait_cancels() {
    local start=$SECONDS
    FAKE_SLURM_HOLD=1 rt kathleen --max-wait 00:00:02 -- true
    expect_status 124 "max-wait overrun"
    [ $((SECONDS - start)) -lt 30 ] || fail "runner waited $((SECONDS - start)) s for a 2 s max-wait"
    expect_exists "$FAKE_STATE/slurm/scancel.log" "scancel called"
    [ -z "$(slurm_jobs)" ] || fail "jobs left in the queue: $(slurm_jobs)"
}

case_kathleen_status_and_cancel() {
    FAKE_SLURM_HOLD=1 bg_rt held kathleen --max-wait 00:01:00 -- true
    local id
    id=$(wait_for_slurm_job) || { fail "held job never queued"; return; }
    "$STATUS" >"$T/status" 2>&1
    RC=$?
    expect_status 0 "status with a pending Slurm job"
    expect_grep "$T/status" '^== kathleen \(fake-kathleen, slurm\)$' "status names the Slurm host"
    expect_grep "$T/status" "^$id singlenode gridcl-kathleen PENDING" "status lists the job"
    "$CANCEL" "$id" >"$T/cancel" 2>&1
    RC=$?
    expect_status 0 "cancel Slurm job"
    expect_line "$FAKE_STATE/slurm/scancel.log" "$id" "scancel called with the id"
    [ -z "$(slurm_jobs)" ] || fail "jobs left in the queue: $(slurm_jobs)"
    wait_exit "$BG" 20
    expect_nonzero "runner of a cancelled job"
    [ "$RC" != 999 ] || fail "runner still waiting after its job was cancelled"
    "$STATUS" >"$T/status" 2>&1
    RC=$?
    expect_status 0 "status without jobs"
    [ ! -s "$T/status" ] || fail "status printed output without jobs: $(head -n 3 "$T/status")"
}

case_cancel_only_own_jobs() {
    # A Slurm job whose id SGE also has, for another user: cancel.sh must skip the SGE job and
    # cancel the Slurm one; an id only another user has is not ours to delete.
    FAKE_SLURM_HOLD=1 bg_rt held kathleen --max-wait 00:01:00 -- true
    local id other
    id=$(wait_for_slurm_job) || { fail "held job never queued"; return; }
    for other in "$id" 4242; do
        mkdir -p "$FAKE_STATE/sge/jobs/$other"
        echo someone-else >"$FAKE_STATE/sge/jobs/$other/owner"
        echo foreign >"$FAKE_STATE/sge/jobs/$other/name"
        echo r >"$FAKE_STATE/sge/jobs/$other/state"
    done
    "$CANCEL" "$id" >"$T/cancel" 2>&1
    RC=$?
    expect_status 0 "cancel a Slurm job whose id another user has on SGE"
    expect_line "$FAKE_STATE/slurm/scancel.log" "$id" "scancel called with the id"
    expect_absent "$FAKE_STATE/sge/qdel.log" "no qdel of another user's job"
    expect_exists "$FAKE_STATE/sge/jobs/$id" "the other user's SGE job kept"
    wait_exit "$BG" 20
    expect_nonzero "runner of a cancelled job"
    "$CANCEL" 4242 >"$T/cancel" 2>&1
    RC=$?
    expect_status 1 "cancel of another user's job"
    expect_absent "$FAKE_STATE/sge/qdel.log" "no qdel of another user's job"
}

case_direct_remote_timeout() {
    rt locust -- 'echo direct-ok'
    expect_status 0 "direct run"
    local id
    id=$(job_id)
    echo "$id" | grep -Eq '^[0-9]+-[0-9]+$' || fail "direct job id '$id' is not <timestamp>-<pid>"
    expect_grep "$T/stdout" '^target: locust$' "summary target"
    grep -rq direct-ok "$LOCAL/target/hpc/locust/$id" 2>/dev/null || fail "direct results not copied"
    expect_grep "$FAKE_STATE/ssh.log" '^fake-locust .*timeout' "remote side runs timeout"
    local dir=$LOCAL/target/hpc/locust/$id
    expect_grep "$dir/node-load-before.txt" '^node_load=(idle|busy)' "node-load snapshot before the command"
    expect_grep "$dir/node-load-after.txt" '^node_load=(idle|busy)' "node-load snapshot after the command"
    expect_grep "$T/stdout" '^node load: before (idle|busy).*; after (idle|busy)' "node-load line in the summary"
    [ "$(wc -l <"$T/stdout")" -le 40 ] || fail "direct summary longer than 40 lines"

    rt locust -- exit 5
    expect_status 5 "direct exit status"

    local start=$SECONDS
    rt locust --max-wait 00:00:02 -- 'sleep 59'
    expect_status 124 "direct max-wait overrun"
    [ $((SECONDS - start)) -lt 30 ] || fail "runner took $((SECONDS - start)) s for a 2 s max-wait"
    [ -z "$(pid_files)" ] || fail "remote pid file left: $(pid_files)"
    no_process 'sleep 59' || fail "timed-out direct job still running"
}

case_prune_merged_branch_trees() {
    local base=$REMOTE/myriad
    g checkout -q -b feature/old main
    commit_file old.txt old
    rt myriad-v100 -- true
    expect_status 0 "run on feature/old"
    expect_exists "$base/feature_old" "tree of feature/old"

    g checkout -q -b feature/wip main
    commit_file wip.txt wip
    rt myriad-v100 -- true
    expect_exists "$base/feature_wip" "tree of feature/wip"

    g checkout -q -b feature/busy main
    commit_file busy.txt busy
    FAKE_SGE_HOLD=1 bg_rt held myriad-v100 --max-wait 00:01:00 -- true
    local id
    id=$(wait_for_sge_job) || { fail "held job never queued"; return; }

    g checkout -q main
    g merge -q --no-ff -m "merge old" feature/old
    g merge -q --no-ff -m "merge busy" feature/busy
    g branch -q -d feature/old
    mkdir -p "$base/not-a-tree"
    echo keep >"$base/not-a-tree/keep"

    g checkout -q -b feature/new main
    rt myriad-v100 -- true
    expect_status 0 "run on feature/new"
    expect_absent "$base/feature_old" "tree of merged (and deleted) feature/old pruned"
    expect_exists "$base/feature_busy" "merged tree with a pending job kept"
    expect_exists "$base/feature_wip" "unmerged tree kept"
    expect_exists "$base/feature_new" "current tree kept"
    expect_exists "$base/not-a-tree/keep" "directory that is not a runner tree kept"

    qdel "$id" >/dev/null 2>&1
    wait_exit "$BG" 20
}

# --- driver -----------------------------------------------------------------------

run_case() { # run_case <name>: prints ok/FAIL; returns non-zero on failure
    local name=$1 log pid waited=0 rc
    log=$(mktemp "${TMPDIR:-/tmp}/hpc-case.XXXXXX")
    (
        trap teardown EXIT
        trap 'exit 143' TERM
        setup
        "case_$name"
        if [ "$failed" != 0 ]; then
            echo "    --- runner stdout (last 15 lines) ---"
            tail -n 15 "$T/stdout" 2>/dev/null | sed 's/^/    /'
            echo "    --- runner stderr (last 15 lines) ---"
            tail -n 15 "$T/stderr" 2>/dev/null | sed 's/^/    /'
        fi
        exit "$failed"
    ) >"$log" 2>&1 &
    pid=$!
    while kill -0 "$pid" 2>/dev/null; do
        if [ "$waited" -ge "$CASE_TIMEOUT" ]; then
            kill -TERM "$pid" 2>/dev/null
            echo "    FAIL: case timed out after $CASE_TIMEOUT s" >>"$log"
            break
        fi
        sleep 1
        waited=$((waited + 1))
    done
    wait "$pid"
    rc=$?
    if [ "$rc" = 0 ]; then
        echo "ok   $name"
    else
        echo "FAIL $name"
        cat "$log"
    fi
    rm -f "$log"
    return "$rc"
}

main() {
    local cases="" arg
    if [ $# -eq 0 ]; then
        cases="$SGE_CASES $DIRECT_CASES $SLURM_CASES"
    fi
    for arg in "$@"; do
        case $arg in
            sge) cases="$cases $SGE_CASES" ;;
            direct) cases="$cases $DIRECT_CASES" ;;
            slurm) cases="$cases $SLURM_CASES" ;;
            -h | --help)
                sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'
                return 0
                ;;
            *)
                if ! type "case_$arg" >/dev/null 2>&1; then
                    echo "run.sh: unknown case '$arg'" >&2
                    return 2
                fi
                cases="$cases $arg"
                ;;
        esac
    done

    local name passed=0 failures=""
    for name in $cases; do
        if run_case "$name"; then
            passed=$((passed + 1))
        else
            failures="$failures $name"
        fi
    done
    if [ -n "$failures" ]; then
        echo "hpc runner tests: $passed passed; failed:$failures"
        return 1
    fi
    echo "hpc runner tests: $passed passed"
}

main "$@"
