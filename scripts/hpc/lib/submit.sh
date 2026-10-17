# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Fetch, job script, submission and waiting (scripts/hpc, steps 3-5), SGE kind.

# submit_manifests <command>: the Cargo manifests to fetch for, one per line: every
# --manifest-path of the command, or "." (the root Cargo.toml) when it names none and the
# working tree has a root Cargo.toml.
submit_manifests() {
    local w prev="" found=""
    set -f
    # shellcheck disable=SC2086 # split the command into words
    for w in $1; do
        case $prev in
            --manifest-path) found="$found ${w%%[;&|)]*}" ;;
        esac
        case $w in
            --manifest-path=*)
                w=${w#--manifest-path=}
                found="$found ${w%%[;&|)]*}"
                ;;
        esac
        prev=$w
    done
    set +f
    if [ -z "$found" ] && [ -f "$HPC_ROOT/Cargo.toml" ]; then
        found=.
    fi
    # shellcheck disable=SC2086 # one word per manifest
    [ -z "$found" ] || printf '%s\n' $found | awk '!seen[$0]++'
}

# submit_build_command <command>: for a plain `cargo run|test|bench ...` (no shell syntax),
# the cargo command that builds it without running (`cargo build ...`, `cargo test --no-run
# ...`), so the job records build and run times separately; empty otherwise.
submit_build_command() {
    local sub pre="" w nl=$'\n'
    case $1 in
        *[\;\&\|\<\>\(\)\$\`\\\"\'\*\?\[\]\{\}\~\#]* | *"$nl"*) return 0 ;;
    esac
    set -f
    # shellcheck disable=SC2086 # split the command into words
    set -- $1
    set +f
    [ "${1:-}" = cargo ] || return 0
    sub=${2:-}
    shift 2 || return 0
    for w in "$@"; do
        [ "$w" = -- ] && break
        pre="$pre $w"
    done
    case $sub in
        run) echo "cargo build$pre" ;;
        test | bench) echo "cargo $sub --no-run$pre" ;;
    esac
}

# submit_render: the job script of this submission on stdout: jobs/<target>.sh with its
# placeholders replaced and the requests of targets.env added, the job settings, and
# jobs/common.sh.
submit_render() {
    local template=$HPC_DIR/jobs/$TARGET.sh requests="" rendered
    [ -f "$template" ] || hpc_die 125 "no job script $template"
    if [ -n "$JOB_CORES" ]; then
        requests="#\$ $(hpc_replace "$CFG_CORES_REQUEST" '{cores}' "$JOB_CORES")"
    fi
    if [ -n "$CFG_MEM_REQUEST" ]; then
        requests="${requests:+$requests$'\n'}#\$ $CFG_MEM_REQUEST"
    fi
    rendered=$(
        R_NAME="gridcl-$TARGET" R_GPUS=$OPT_GPUS R_TIME=$OPT_TIME R_TREE=$TREE \
            R_REQUESTS=$requests awk '
            function rep(s, key, val,    out, i) {
                out = ""
                while ((i = index(s, key)) > 0) {
                    out = out substr(s, 1, i - 1) val
                    s = substr(s, i + length(key))
                }
                return out s
            }
            $0 == "#@REQUESTS@" {
                if (ENVIRON["R_REQUESTS"] != "") print ENVIRON["R_REQUESTS"]
                next
            }
            {
                l = rep($0, "@NAME@", ENVIRON["R_NAME"])
                l = rep(l, "@GPUS@", ENVIRON["R_GPUS"])
                l = rep(l, "@TIME@", ENVIRON["R_TIME"])
                l = rep(l, "@TREE@", ENVIRON["R_TREE"])
                print l
            }' "$template"
    ) || hpc_die 125 "cannot render $template"
    if printf '%s\n' "$rendered" | grep -Eq '^#\$.*@[A-Z_]+@'; then
        hpc_die 125 "$template has a placeholder the runner does not know"
    fi
    printf '%s\n' "$rendered"
    echo
    echo "# --- job settings, rendered by scripts/hpc/remote-test.sh ---"
    printf 'GRIDCL_HPC_TARGET=%q\n' "$TARGET"
    printf 'GRIDCL_HPC_TREE=%q\n' "$TREE"
    printf 'GRIDCL_HPC_BRANCH=%q\n' "$GIT_BRANCH"
    printf 'GRIDCL_HPC_COMMIT=%q\n' "$GIT_COMMIT"
    printf 'GRIDCL_HPC_DIRTY=%q\n' "$GIT_DIRTY"
    printf 'GRIDCL_HPC_COMMAND=%q\n' "$CMD"
    printf 'GRIDCL_HPC_BUILD=%q\n' "$BUILD_CMD"
    printf 'GRIDCL_HPC_MODULES=%q\n' "$CFG_MODULES"
    printf 'GRIDCL_HPC_CUDA_PATH=%q\n' "$CFG_CUDA_PATH"
    printf 'export CARGO_TARGET_DIR=%q\n' "$TREE/target/$TARGET"
    echo "export CARGO_NET_OFFLINE=true"
    echo "export GRIDCL_TEST_DEVICE=cuda"
    if [ -n "$OPT_RANKS" ]; then
        echo "export GRIDCL_RANKS=$OPT_RANKS"
    fi
    echo
    echo "# --- scripts/hpc/jobs/common.sh ---"
    awk 'NR > 1 || !/^#!/' "$HPC_DIR/jobs/common.sh" || hpc_die 125 "cannot read jobs/common.sh"
}

# submit_fetch: rustup toolchain, crates and the LLVM bundle on the login node.
submit_fetch() {
    local manifests
    manifests=$(submit_manifests "$CMD")
    hpc_msg "fetching on $CFG_HOST (log: $TREE/.gridcl-hpc/fetch-$TARGET.log)"
    # shellcheck disable=SC2086 # one word per manifest
    hpc_remote r_fetch "$TREE" "$TARGET" "$CFG_LOGIN_MODULES" $manifests
    hpc_remote_check $? "the fetch step"
}

# submit_job: submits the rendered job script; sets JOB_ID and JOB_SUBMITTED.
submit_job() {
    local script meta out
    script=$(submit_render) || exit 125
    meta="$(printf 'target=%q\nscheduler=sge\n' "$TARGET")"$'\n'"$(sync_state_env)"
    out=$(hpc_remote r_submit "$TREE" "$script" "$meta" "$(printf 'command=%q' "$CMD")")
    hpc_remote_check $? "qsub"
    JOB_ID=$(printf '%s\n' "$out" | sed -n 's/^job_id=//p' | head -n 1)
    JOB_SUBMITTED=$(printf '%s\n' "$out" | sed -n 's/^submitted=//p' | head -n 1)
    case $JOB_ID in
        '' | *[!0-9]*) hpc_die 125 "qsub returned no job id: $out" ;;
    esac
    JOB_SUBMITTED_LOCAL=$SECONDS
    hpc_msg "submitted job $JOB_ID to $TARGET"
}

# submit_wait: polls until the job has ended or --max-wait has passed since submission;
# sets JOB_STATE to done (exit status written), gone (ended without one) or timeout.
submit_wait() {
    local deadline=$((JOB_SUBMITTED_LOCAL + MAX_WAIT_SECONDS)) state rc last="" gone=0 fails=0
    while :; do
        state=$(hpc_remote r_poll "$TREE" "$JOB_ID")
        rc=$?
        if [ "$rc" != 0 ]; then
            fails=$((fails + 1))
            if [ "$fails" -ge 5 ]; then
                JOB_ACTIVE=""
                hpc_die 125 "lost contact with $CFG_HOST while waiting; job $JOB_ID may still" \
                    "be queued: check scripts/hpc/status.sh and cancel it with scripts/hpc/cancel.sh $JOB_ID"
            fi
        else
            fails=0
            case $state in
                done)
                    JOB_STATE="done"
                    return 0
                    ;;
                gone)
                    gone=$((gone + 1))
                    if [ "$gone" -ge 3 ]; then
                        JOB_STATE=gone
                        return 0
                    fi
                    ;;
                *)
                    gone=0
                    [ "$state" = "$last" ] || hpc_msg "job $JOB_ID $state"
                    last=$state
                    ;;
            esac
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            JOB_STATE=timeout
            return 0
        fi
        sleep "$HPC_POLL" &
        wait $!
    done
}

# submit_delete: deletes the job (after --max-wait or on a signal).
submit_delete() {
    hpc_remote r_delete "$TREE" "$JOB_ID" >/dev/null
}

# --- remote side ------------------------------------------------------------------

r_fetch_steps() { # r_fetch_steps <tree> <target> <login modules> [manifest...]
    local tree=$1 target=$2 m
    shift 3
    cd "$tree" || return 1
    unset CARGO_NET_OFFLINE
    CARGO_TARGET_DIR=$tree/target/$target-login
    export CARGO_TARGET_DIR
    if [ -f rust-toolchain.toml ] || [ -f rust-toolchain ]; then
        if ! command -v rustup >/dev/null 2>&1; then
            echo "rustup not found on the login node; install it in the remote home"
            return 1
        fi
        echo "+ rustup toolchain install"
        rustup toolchain install || rustup show active-toolchain || return 1
    fi
    for m in "$@"; do
        if [ "$m" = . ]; then
            set --
        else
            set -- --manifest-path "$m"
        fi
        echo "+ cargo fetch $*"
        cargo fetch "$@" || return 1
        if cargo pkgid "$@" cubecl-llvm >/dev/null 2>&1; then
            echo "+ cargo build -p cubecl-llvm $*"
            cargo build -p cubecl-llvm "$@" || return 1
        else
            echo "no cubecl-llvm in the dependency graph of ${m}; LLVM bundle not needed"
        fi
    done
}

r_fetch() { # r_fetch <tree> <target> <login modules> [manifest...]: log in .gridcl-hpc/
    local log=$1/.gridcl-hpc/fetch-$2.log rc
    r_env sge
    mkdir -p "$1/.gridcl-hpc" || return 1
    (
        eval "$3"
        r_fetch_steps "$@"
    ) >"$log" 2>&1 </dev/null
    rc=$?
    if [ "$rc" != 0 ]; then
        echo "fetch step failed (status $rc); last lines of $log:" >&2
        tail -n 30 "$log" >&2
        return 1
    fi
}

r_submit() { # r_submit <tree> <job script> <job lines> <command line>: prints job_id=, submitted=
    local tree=$1 file out id now rc
    r_env sge
    mkdir -p "$tree/results" "$tree/.gridcl-hpc/jobs" || return 1
    file=$tree/.gridcl-hpc/submit-$$.sh
    printf '%s\n' "$2" >"$file" || return 1
    out=$(cd "$tree" && qsub -terse "$file" </dev/null)
    rc=$?
    rm -f "$file"
    if [ "$rc" != 0 ]; then
        echo "qsub failed (status $rc): $out" >&2
        return 1
    fi
    id=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\([.].*\)\{0,1\}$/\1/p' | tail -n 1)
    if [ -z "$id" ]; then
        echo "qsub -terse printed no job id: $out" >&2
        return 1
    fi
    now=$(date +%s)
    printf '%s\nsubmitted=%s\n' "$3" "$now" >"$tree/.gridcl-hpc/jobs/$id"
    mkdir -p "$tree/results/$id"
    printf '%s\n%s\nsubmitted=%s\n' "$3" "$4" "$now" >"$tree/results/$id/submit.env"
    echo "job_id=$id"
    echo "submitted=$now"
}

r_poll() { # r_poll <tree> <job id>: prints queued, running, done or gone
    local results=$1/results/$2
    r_env sge
    if [ -f "$results/exit_status" ]; then
        echo "done"
    elif qstat -j "$2" >/dev/null 2>&1; then
        if [ -f "$results/times.env" ]; then echo running; else echo queued; fi
    elif [ -f "$results/exit_status" ]; then
        echo "done"
    else
        echo gone
    fi
}

r_delete() { # r_delete <tree> <job id>
    r_env sge
    qdel "$2" </dev/null
}
