# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Results and summary (scripts/hpc, step 6).
#
# A job writes <tree>/results/<job id>/ (see jobs/common.sh); report_collect copies it to
# target/hpc/<target>/<job id>/ and report_summary prints at most 40 lines on stdout:
# the header lines target, job id, commit, dirty, command, exit status, queue wait,
# run time, first failure, then node and results, then an excerpt of output.log that
# starts at the first failure.

# Lines that count as a failure in output.log; the first one is the "first failure".
REPORT_FAILURE_RE='(^|[^A-Za-z_])(error(\[E[0-9]+\])?:|ERROR([^A-Za-z_]|$)|FAILED|panicked at|CUDA_ERROR_[A-Z_]+|[Ss]egmentation fault|command not found|Killed$)'

# report_collect: finishes the remote side of the job and copies its results to
# $LOCAL_RESULTS; sets HAVE_RESULTS (yes|no).
report_collect() {
    local out
    out=$(hpc_remote r_collect "$TREE" "$JOB_ID")
    hpc_remote_check $? "collecting the results"
    HAVE_RESULTS=no
    case $out in
        *results=yes*)
            mkdir -p "$LOCAL_RESULTS" || hpc_die 125 "cannot create $LOCAL_RESULTS"
            rsync -az -e "ssh $HPC_SSH_OPTS" "$CFG_HOST:$TREE/results/$JOB_ID/" "$LOCAL_RESULTS/" ||
                hpc_die 125 "copying the results of job $JOB_ID failed"
            HAVE_RESULTS=yes
            ;;
    esac
}

report_value() { # report_value <file> <key>: the last value of key=... in a file
    sed -n "s/^$2=//p" "$1" 2>/dev/null | tail -n 1
}

report_duration() { # report_duration <from> <to>: "N s", or empty when a value is missing
    case "$1$2" in
        '' | *[!0-9]*) return 0 ;;
    esac
    [ -n "$1" ] && [ -n "$2" ] && echo "$(($2 - $1)) s"
}

report_one_line() { # report_one_line <text>: one summary line of at most 300 characters
    printf '%s' "$1" | tr '\n\t' '  ' | cut -c 1-300
}

# report_summary <state>: prints the summary; sets REPORT_STATUS, the runner's exit status.
report_summary() {
    local state=$1 log times status_line failure="" fline="" queue run start end bstart bend rstart
    local total node
    log=$LOCAL_RESULTS/output.log
    times=$LOCAL_RESULTS/times.env
    start=$(report_value "$times" start)
    end=$(report_value "$times" end)
    bstart=$(report_value "$times" build_start)
    bend=$(report_value "$times" build_end)
    rstart=$(report_value "$times" run_start)

    case $state in
        timeout)
            REPORT_STATUS=124
            status_line="124 (--max-wait $OPT_MAX_WAIT exceeded; job $JOB_ID deleted)"
            if [ -n "$start" ]; then
                failure="--max-wait $OPT_MAX_WAIT exceeded while the job was running"
            else
                failure="--max-wait $OPT_MAX_WAIT exceeded while the job was queued"
            fi
            ;;
        done)
            REPORT_STATUS=$(cat "$LOCAL_RESULTS/exit_status" 2>/dev/null)
            case $REPORT_STATUS in
                '' | *[!0-9]*)
                    REPORT_STATUS=125
                    status_line="none (unreadable exit_status file)"
                    failure="unreadable exit_status file"
                    ;;
                *) status_line=$REPORT_STATUS ;;
            esac
            ;;
        *)
            REPORT_STATUS=125
            status_line="none (the job ended without writing one: deleted, or killed by the scheduler; see qacct -j $JOB_ID)"
            failure="job ended without an exit status"
            ;;
    esac
    if [ -z "$failure" ]; then
        if [ "$REPORT_STATUS" = 0 ]; then
            failure=none
        elif [ -s "$log" ]; then
            fline=$(grep -n -E -m 1 -- "$REPORT_FAILURE_RE" "$log" | cut -d: -f1)
            if [ -n "$fline" ]; then
                failure=$(sed -n "${fline}p" "$log")
            else
                failure="(no error line; last line) $(grep -v '^[[:space:]]*$' "$log" | tail -n 1)"
            fi
        else
            failure="(no output)"
        fi
    fi

    if [ -n "$start" ]; then
        queue=$(report_duration "$JOB_SUBMITTED" "$start")
        run=$(report_duration "$start" "$end")
        [ -n "$run" ] || run="unknown (job did not finish)"
        if [ -n "$bstart" ] && [ -n "$bend" ]; then
            run="$run (build $(report_duration "$bstart" "$bend")${rstart:+, command $(report_duration "$rstart" "$end")})"
        fi
    else
        queue="$((SECONDS - JOB_SUBMITTED_LOCAL)) s (never started)"
        run="0 s (never started)"
    fi
    node=$(sed -n 's/^host: //p' "$LOCAL_RESULTS/prologue.txt" 2>/dev/null | head -n 1)

    echo "target: $TARGET"
    echo "job id: $JOB_ID"
    echo "commit: $(git -C "$HPC_ROOT" rev-parse --short=12 "$GIT_COMMIT") ($GIT_BRANCH)"
    if [ "$GIT_DIRTY" = yes ]; then
        echo "dirty: yes (changes hashed $(printf '%s' "$GIT_DIRTY_HASH" | cut -c 1-12))"
    else
        echo "dirty: no"
    fi
    echo "command: $(report_one_line "$CMD")"
    echo "exit status: $status_line"
    echo "queue wait: $queue"
    echo "run time: $run"
    echo "first failure: $(report_one_line "$failure")"
    if [ "$HAVE_RESULTS" = yes ]; then
        echo "node: ${node:-unknown}; results: ${LOCAL_RESULTS#"$HPC_ROOT"/}/"
    else
        echo "node: ${node:-none}; results: none"
    fi

    [ -s "$log" ] || return 0
    total=$(wc -l <"$log" | tr -d ' ')
    if [ -n "$fline" ]; then
        if [ $((total - fline)) -lt 26 ]; then
            echo "--- output.log, lines $fline-$total ---"
            sed -n "${fline},${total}p" "$log" | cut -c 1-300
        else
            echo "--- output.log from line $fline (first failure) ---"
            sed -n "${fline},$((fline + 11))p" "$log" | cut -c 1-300
            echo "--- output.log, last 12 of $total lines ---"
            tail -n 12 "$log" | cut -c 1-300
        fi
    else
        echo "--- output.log, last $([ "$total" -lt 12 ] && echo "$total" || echo 12) of $total lines ---"
        tail -n 12 "$log" | cut -c 1-300
    fi
}

# --- remote side ------------------------------------------------------------------

r_collect() { # r_collect <tree> <job id>: moves the scheduler log in, drops the job file
    local tree=$1 id=$2
    if [ -f "$tree/results/sge-$id.log" ]; then
        mkdir -p "$tree/results/$id" && mv -f "$tree/results/sge-$id.log" "$tree/results/$id/sge.log"
    fi
    rm -f "$tree/.gridcl-hpc/jobs/$id"
    if [ -d "$tree/results/$id" ]; then echo results=yes; else echo results=no; fi
}
