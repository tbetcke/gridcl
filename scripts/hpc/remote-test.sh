#!/usr/bin/env bash
# Run a command on a remote test target (docs/design.md, "Development environment and test
# targets"): sync the working tree, fetch on the login node, submit the job, wait, report.
# See usage() below; the steps live in lib/{config,sync,submit,direct,report}.sh.
#
# Environment: GRIDCL_TARGETS_ENV overrides the settings file (default
# scripts/hpc/targets.env); GRIDCL_HPC_POLL sets the polling interval in seconds (default 30).
# shellcheck source-path=SCRIPTDIR
set -u
set -o pipefail

HPC_PROG=remote-test.sh
HPC_DIR=$(cd "$(dirname "$0")" && pwd)
HPC_ROOT=$(cd "$HPC_DIR/../.." && pwd)

# shellcheck source=lib/config.sh
. "$HPC_DIR/lib/config.sh"
# shellcheck source=lib/sync.sh
. "$HPC_DIR/lib/sync.sh"
# shellcheck source=lib/submit.sh
. "$HPC_DIR/lib/submit.sh"
# shellcheck source=lib/direct.sh
. "$HPC_DIR/lib/direct.sh"
# shellcheck source=lib/report.sh
. "$HPC_DIR/lib/report.sh"

DEFAULT_TIME=01:00:00
DEFAULT_MAX_WAIT=03:00:00

usage() {
    cat <<EOF
usage: scripts/hpc/remote-test.sh <target> [--gpus N] [--ranks N] [--cores N] [--time HH:MM:SS] [--max-wait HH:MM:SS] -- <command>
targets: myriad-v100 | myriad-a100 | locust

Syncs the working tree to the target, fetches crates on its login node, runs <command> in
a job from the root of the remote tree (offline, GRIDCL_TEST_DEVICE=cuda) and prints a
summary of at most 40 lines. The words after -- are joined with spaces and run by bash.
On SGE targets (Myriad) the job is queued; on direct targets (Locust: no scheduler) it starts
at once under timeout (the smaller of --time and --max-wait) and records the node load
before and after the command (summary line "node load:").

  --gpus N           GPUs of the job (default 1; at most 2 on myriad-v100, 4 on myriad-a100)
  --ranks N          reserve N slots and export GRIDCL_RANKS=N; the command starts mpirun
  --cores N          CPU cores of the job (36: a whole Myriad node, for CPU benchmarks);
                     ignored on direct targets, which reserve nothing
  --time HH:MM:SS    wall-clock limit of the job (default $DEFAULT_TIME)
  --max-wait HH:MM:SS  wait at most this long, queue time included (default $DEFAULT_MAX_WAIT);
                     then the job is deleted and the exit status is 124

Results: target/hpc/<target>/<job id>/. Exit status: the command's; 124 after --max-wait;
125 when the runner fails (settings, ssh, sync, fetch, submission) or the job ends without
an exit status (cancelled, stopped at --time); 2 on usage errors.
Settings: scripts/hpc/targets.env (from targets.example.env), or \$GRIDCL_TARGETS_ENV.
EOF
}

usage_error() {
    hpc_msg "$*"
    usage >&2
    exit 2
}

check_count() { # check_count <option> <value>: a positive integer
    case $2 in
        '' | *[!0-9]*) usage_error "$1 needs a positive integer, not '$2'" ;;
    esac
    [ $((10#$2)) -ge 1 ] || usage_error "$1 needs a positive integer, not '$2'"
}

check_duration() { # check_duration <option> <value>: HH:MM:SS
    printf '%s\n' "$2" | grep -Eq '^[0-9]+:[0-5][0-9]:[0-5][0-9]$' ||
        usage_error "$1 needs HH:MM:SS, not '$2'"
}

parse_args() {
    local opt val dashdash=""
    [ $# -gt 0 ] || usage_error "no target given"
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
    esac
    TARGET=$1
    shift
    hpc_known_target "$TARGET" || usage_error "unknown target '$TARGET'"
    OPT_GPUS=1 OPT_RANKS="" OPT_CORES="" OPT_TIME=$DEFAULT_TIME OPT_MAX_WAIT=$DEFAULT_MAX_WAIT
    while [ $# -gt 0 ]; do
        opt=$1
        case $opt in
            --)
                shift
                dashdash=1
                break
                ;;
            -h | --help)
                usage
                exit 0
                ;;
            --gpus=* | --ranks=* | --cores=* | --time=* | --max-wait=*)
                val=${opt#*=}
                opt=${opt%%=*}
                shift
                ;;
            --gpus | --ranks | --cores | --time | --max-wait)
                [ $# -ge 2 ] || usage_error "$opt needs a value"
                val=$2
                shift 2
                ;;
            -*) usage_error "unknown option '$opt'" ;;
            *) usage_error "'$opt' before --: put the command after --" ;;
        esac
        case $opt in
            --gpus) check_count "$opt" "$val" && OPT_GPUS=$((10#$val)) ;;
            --ranks) check_count "$opt" "$val" && OPT_RANKS=$((10#$val)) ;;
            --cores) check_count "$opt" "$val" && OPT_CORES=$((10#$val)) ;;
            --time) check_duration "$opt" "$val" && OPT_TIME=$val ;;
            --max-wait) check_duration "$opt" "$val" && OPT_MAX_WAIT=$val ;;
        esac
    done
    [ -n "$dashdash" ] || usage_error "no -- before the command"
    [ $# -gt 0 ] || usage_error "no command after --"
    CMD="$*"
    case $CMD in
        *[![:space:]]*) ;;
        *) usage_error "empty command" ;;
    esac
    case $TARGET in
        myriad-v100) [ "$OPT_GPUS" -le 2 ] || usage_error "--gpus $OPT_GPUS: an E or F node has 2 V100s" ;;
        myriad-a100) [ "$OPT_GPUS" -le 4 ] || usage_error "--gpus $OPT_GPUS: an L node has 4 A100s" ;;
    esac
    if [ -n "$OPT_RANKS" ] && [ -n "$OPT_CORES" ] && [ "$OPT_CORES" -lt "$OPT_RANKS" ]; then
        usage_error "--cores $OPT_CORES is less than --ranks $OPT_RANKS"
    fi
    # The slots of the job: --cores, or one per rank.
    JOB_CORES=${OPT_CORES:-$OPT_RANKS}
    MAX_WAIT_SECONDS=$(hpc_seconds "$OPT_MAX_WAIT")
    TIME_SECONDS=$(hpc_seconds "$OPT_TIME")
}

# shellcheck disable=SC2329 # invoked by the traps in run_job
on_signal() { # on_signal <status>: deletes a submitted job that has not ended
    trap - INT TERM HUP
    if [ -n "${JOB_ACTIVE:-}" ]; then
        hpc_msg "interrupted: deleting job $JOB_ACTIVE"
        submit_delete >/dev/null 2>&1
        hpc_remote r_collect "$TREE" "$JOB_ACTIVE" >/dev/null 2>&1
    fi
    exit "$1"
}

run_job() {
    sync_check_rsync
    sync_git_state
    TREE=$CFG_BASE/$TREE_NAME
    LOCAL_RESULTS=""
    hpc_msg "target $TARGET ($CFG_HOST, $CFG_SCHEDULER), tree $TREE, commit $(printf '%s' "$GIT_COMMIT" | cut -c 1-12), dirty: $GIT_DIRTY"

    sync_prepare
    sync_tree
    submit_fetch
    BUILD_CMD=$(submit_build_command "$CMD")

    trap 'on_signal 130' INT
    trap 'on_signal 143' TERM
    trap 'on_signal 129' HUP
    case $CFG_SCHEDULER in
        sge) submit_job ;;
        direct) direct_submit ;;
    esac
    JOB_ACTIVE=$JOB_ID
    submit_wait
    if [ "$JOB_STATE" = timeout ] && [ -z "$JOB_ENDED" ]; then
        hpc_msg "--max-wait $OPT_MAX_WAIT exceeded: deleting job $JOB_ID"
        submit_delete || hpc_msg "deleting job $JOB_ID failed; check scripts/hpc/status.sh"
    elif [ "$JOB_STATE" = timeout ]; then
        hpc_msg "--max-wait $OPT_MAX_WAIT exceeded: job $JOB_ID was stopped on the node"
    fi
    JOB_ACTIVE=""
    trap - INT TERM HUP

    LOCAL_RESULTS=$HPC_ROOT/target/hpc/$TARGET/$JOB_ID
    report_collect
    report_summary "$JOB_STATE"
    exit "$REPORT_STATUS"
}

main() {
    parse_args "$@"
    HPC_POLL=${GRIDCL_HPC_POLL:-30}
    printf '%s\n' "$HPC_POLL" | grep -Eq '^[0-9]+([.][0-9]+)?$' || HPC_POLL=30
    config_load "$TARGET"
    if [ "$CFG_SCHEDULER" = direct ]; then
        if [ -n "$OPT_CORES" ]; then
            hpc_msg "--cores ignored: $TARGET has no scheduler to reserve cores (the node load is recorded instead)"
        fi
        JOB_CORES=""
    elif [ -n "$OPT_CORES$OPT_RANKS" ]; then
        case $CFG_CORES_REQUEST in
            *'{cores}'*) ;;
            *) hpc_die 125 "--cores/--ranks need ${CFG_PREFIX}_CORES_REQUEST with {cores} in targets.env" ;;
        esac
    fi
    run_job
}

main "$@"
