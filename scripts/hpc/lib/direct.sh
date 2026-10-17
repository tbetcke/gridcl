# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Direct targets (scripts/hpc, step 4, scheduler kind `direct`): nodes without a scheduler,
# such as Locust. The job runs at once on the node the runner reaches over ssh:
#   1. r_direct_submit writes the rendered job script to <tree>/results/<id>/job.sh, the job
#      file .gridcl-hpc/jobs/<id>, submit.env and the wrapper .gridcl-hpc/run/<id>.wrap. The
#      job id is <timestamp>-<pid>: YYYYmmddHHMMSS and the pid of the remote shell.
#   2. direct_start runs `ssh <host> ... timeout -k 30 <limit> bash results/<id>/job.sh`
#      detached (nohup) under the wrapper, which writes the pid of timeout to
#      .gridcl-hpc/run/<id>.pid, waits, writes timeout's exit status to
#      results/<id>/direct_status and removes the pid file. Output of the wrapper goes to
#      results/<id>/direct.log.
#   3. The runner polls like on SGE (r_poll): the job runs while the pid file names a live
#      timeout process of this job.
# <limit> is the smaller of --time and --max-wait in seconds. timeout ends the job on the
# node itself, so a lost connection cannot leave it running. cancel.sh, and the runner after
# --max-wait or on a signal, send SIGTERM to timeout, which passes it to the job's process
# group; SIGKILL follows after 10 s.

# direct_submit: writes the job to the remote tree and starts it; sets JOB_ID,
# JOB_SUBMITTED and JOB_SUBMITTED_LOCAL.
direct_submit() {
    local script meta wrapper out limit
    script=$(submit_render) || exit 125
    meta="$(printf 'target=%q\nscheduler=direct\n' "$TARGET")"$'\n'"$(sync_state_env)"
    wrapper="$(declare -f r_direct_wrapper)"$'\n''r_direct_wrapper "$@"'
    out=$(hpc_remote r_direct_submit "$TREE" "$script" "$meta" "$(printf 'command=%q' "$CMD")" "$wrapper")
    hpc_remote_check $? "writing the job"
    JOB_ID=$(printf '%s\n' "$out" | sed -n 's/^job_id=//p' | head -n 1)
    JOB_SUBMITTED=$(printf '%s\n' "$out" | sed -n 's/^submitted=//p' | head -n 1)
    printf '%s\n' "$JOB_ID" | grep -Eqx '[0-9]+-[0-9]+' || hpc_die 125 "no job id from $CFG_HOST: $out"
    limit=$TIME_SECONDS
    [ "$MAX_WAIT_SECONDS" -ge "$limit" ] || limit=$MAX_WAIT_SECONDS
    direct_start "$limit"
    JOB_SUBMITTED_LOCAL=$SECONDS
    hpc_msg "started job $JOB_ID on $TARGET (stopped on the node after $limit s)"
}

# direct_start <limit>: starts job $JOB_ID detached under timeout. The command reaches the
# remote login shell as one `bash -c '...'` word, so the tree path must need no quoting.
direct_start() {
    local cmd rc
    printf '%s\n' "$TREE" | grep -Eqx '/[A-Za-z0-9._/+-]+' ||
        hpc_die 125 "the tree path $TREE needs quoting; use a simpler ${CFG_PREFIX}_BASE"
    cmd="cd $TREE && JOB_ID=$JOB_ID nohup bash .gridcl-hpc/run/$JOB_ID.wrap $TREE $JOB_ID"
    cmd="$cmd timeout -k 10 $1 bash results/$JOB_ID/job.sh"
    cmd="$cmd </dev/null >>results/$JOB_ID/direct.log 2>&1 &"
    # shellcheck disable=SC2086,SC2029 # options list; $cmd is meant to expand here
    ssh $HPC_SSH_OPTS "$CFG_HOST" "bash -c '$cmd'" </dev/null
    rc=$?
    hpc_remote_check "$rc" "starting job $JOB_ID"
}

# --- remote side ------------------------------------------------------------------

# r_direct_submit <tree> <job script> <job lines> <command line> <wrapper>: writes the job;
# prints job_id= and submitted=.
r_direct_submit() {
    local tree=$1 id now
    r_env direct
    id=$(date +%Y%m%d%H%M%S)-$$
    mkdir -p "$tree/results/$id" "$tree/.gridcl-hpc/jobs" "$tree/.gridcl-hpc/run" || return 1
    printf '%s\n' "$2" >"$tree/results/$id/job.sh" || return 1
    printf '%s\n' "$5" >"$tree/.gridcl-hpc/run/$id.wrap" || return 1
    now=$(date +%s)
    printf '%s\nsubmitted=%s\n' "$3" "$now" >"$tree/.gridcl-hpc/jobs/$id"
    printf '%s\n%s\nsubmitted=%s\n' "$3" "$4" "$now" >"$tree/results/$id/submit.env"
    echo "job_id=$id"
    echo "submitted=$now"
}

# r_direct_wrapper <tree> <job id> <command...>: runs the command (timeout ...) in the
# background, records its pid while it runs and its exit status when it ends. Written to
# .gridcl-hpc/run/<id>.wrap and started by direct_start.
r_direct_wrapper() {
    local tree=$1 id=$2 pid rc
    shift 2
    "$@" &
    pid=$!
    echo "$pid" >"$tree/.gridcl-hpc/run/$id.pid.tmp" &&
        mv -f "$tree/.gridcl-hpc/run/$id.pid.tmp" "$tree/.gridcl-hpc/run/$id.pid"
    wait "$pid"
    rc=$?
    echo "$rc" >"$tree/results/$id/direct_status"
    rm -f "$tree/.gridcl-hpc/run/$id.pid"
}

# r_direct_alive <tree> <job id>: the job runs, or was written less than 60 s ago and has
# not yet recorded its pid. The pid must belong to a process whose arguments name the job
# (against pid reuse): from /proc on Linux, else from ps when ps is usable.
r_direct_alive() {
    local pidf=$1/.gridcl-hpc/run/$2.pid pid args submitted
    if [ -f "$pidf" ]; then
        pid=$(cat "$pidf" 2>/dev/null)
        case $pid in
            '' | *[!0-9]*) return 1 ;;
        esac
        if [ -r "/proc/$pid/cmdline" ]; then
            tr '\0' ' ' <"/proc/$pid/cmdline" | grep -q -- "$2"
            return
        fi
        kill -0 "$pid" 2>/dev/null || return 1
        args=$(ps -ww -p "$pid" -o args= 2>/dev/null) || return 0
        case $args in
            *"$2"*) return 0 ;;
        esac
        return 1
    fi
    [ ! -f "$1/results/$2/direct_status" ] && [ ! -f "$1/results/$2/exit_status" ] || return 1
    submitted=$(sed -n 's/^submitted=//p' "$1/.gridcl-hpc/jobs/$2" 2>/dev/null | head -n 1)
    case $submitted in
        '' | *[!0-9]*) return 1 ;;
    esac
    [ $(($(date +%s) - submitted)) -lt 60 ]
}

# r_direct_stop <tree> <job id> [marker]: stops a running direct job; with a marker, writes
# it to results/<id>/cancelled first. Status 1 when the job is not running.
r_direct_stop() {
    local tree=$1 id=$2 pidf pid i=0
    pidf=$tree/.gridcl-hpc/run/$id.pid
    if ! r_direct_alive "$tree" "$id"; then
        rm -f "$pidf" "$tree/.gridcl-hpc/jobs/$id"
        return 1
    fi
    if [ -n "${3:-}" ]; then
        mkdir -p "$tree/results/$id" && echo "$3" >"$tree/results/$id/cancelled"
    fi
    while [ ! -f "$pidf" ] && [ $i -lt 50 ]; do
        sleep 0.2
        i=$((i + 1))
    done
    pid=$(cat "$pidf" 2>/dev/null)
    case $pid in
        '' | *[!0-9]*)
            echo "job $id recorded no pid" >&2
            return 1
            ;;
    esac
    kill -TERM "$pid" 2>/dev/null
    i=0
    while kill -0 "$pid" 2>/dev/null && [ $i -lt 50 ]; do
        sleep 0.2
        i=$((i + 1))
    done
    if kill -0 "$pid" 2>/dev/null; then
        # timeout leads its own process group: end every process of the job.
        kill -KILL -- "-$pid" 2>/dev/null
        kill -KILL "$pid" 2>/dev/null
    fi
    rm -f "$pidf" "$tree/.gridcl-hpc/jobs/$id"
}

# r_direct_cancel <job id> <marker> <base>...: stops the job in whichever tree below the
# bases holds it. Status 0 stopped, 1 already ended, 4 unknown.
r_direct_cancel() {
    local id=$1 marker=$2 b f
    shift 2
    for b in "$@"; do
        for f in "$b"/*/.gridcl-hpc/jobs/"$id" "$b"/*/.gridcl-hpc/run/"$id".pid; do
            [ -f "$f" ] || continue
            r_direct_stop "${f%/.gridcl-hpc/*}" "$id" "$marker" && return 0
            echo "job $id has already ended" >&2
            return 1
        done
    done
    return 4
}

# r_direct_status <base>...: one line per running direct job: id, tree, command.
r_direct_status() {
    local b f id tree cmd
    for b in "$@"; do
        for f in "$b"/*/.gridcl-hpc/run/*.pid; do
            [ -f "$f" ] || continue
            id=${f##*/}
            id=${id%.pid}
            tree=${f%/.gridcl-hpc/run/*}
            r_direct_alive "$tree" "$id" || continue
            cmd=$(sed -n 's/^command=//p' "$tree/results/$id/submit.env" 2>/dev/null | head -n 1)
            printf '%s  running  %s  %s\n' "$id" "${tree##*/}" "$cmd"
        done
    done
}
