# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Fetch, job script, submission and waiting (scripts/hpc, steps 3-5). Submission is SGE's
# (qsub) or Slurm's (sbatch) here; direct targets start their job with lib/direct.sh and
# share the rest.

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

# submit_cargo_features <command>: the feature options of the command's cargo invocation
# (--features, -F, --all-features, --no-default-features), as one line of words; words after
# a lone `--` belong to the program and are skipped. The fetch step passes them on, since a
# package such as cubecl-llvm is in the lock file but activated only by some features.
submit_cargo_features() {
    local w prev="" found=""
    set -f
    # shellcheck disable=SC2086 # split the command into words
    for w in $1; do
        w=${w%%[;&|)]*}
        case $prev in
            --features | -F) found="$found --features $w" ;;
        esac
        case $w in
            --) break ;;
            --features=* | -F?*)
                w=${w#--features=}
                found="$found --features ${w#-F}"
                ;;
            --all-features | --no-default-features) found="$found $w" ;;
        esac
        prev=$w
    done
    set +f
    printf '%s\n' "${found# }"
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
# placeholders replaced and the requests of targets.env added (SGE), the job settings, and
# jobs/common.sh. GRIDCL_TEST_DEVICE is cuda, or cpu on myriad-cpu and kathleen
# (hpc_target_device). Slurm jobs get @NODES@ and @QOS@; on two nodes mpirun maps the ranks
# by node (GRIDCL_MPIRUN_FLAGS), so that a two-node run spreads over both nodes.
submit_render() {
    local template=$HPC_DIR/jobs/$TARGET.sh requests="" rendered
    [ -f "$template" ] || hpc_die 125 "no job script $template"
    if [ -n "$JOB_CORES" ] && [ "$CFG_SCHEDULER" = sge ]; then
        requests="#\$ $(hpc_replace "$CFG_CORES_REQUEST" '{cores}' "$JOB_CORES")"
    fi
    if [ -n "$CFG_MEM_REQUEST" ] && [ "$CFG_SCHEDULER" = sge ]; then
        requests="${requests:+$requests$'\n'}#\$ $CFG_MEM_REQUEST"
    fi
    rendered=$(
        R_NAME="gridcl-$TARGET" R_GPUS=$OPT_GPUS R_TIME=$OPT_TIME R_TREE=$TREE \
            R_NODES=${OPT_NODES:-1} R_QOS=${JOB_QOS:-} R_REQUESTS=$requests awk '
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
                l = rep(l, "@NODES@", ENVIRON["R_NODES"])
                l = rep(l, "@QOS@", ENVIRON["R_QOS"])
                print l
            }' "$template"
    ) || hpc_die 125 "cannot render $template"
    if printf '%s\n' "$rendered" | grep -Eq '^#(\$|SBATCH).*@[A-Z_]+@'; then
        hpc_die 125 "$template has a placeholder the runner does not know"
    fi
    printf '%s\n' "$rendered"
    echo
    echo "# --- job settings, rendered by scripts/hpc/remote-test.sh ---"
    printf 'GRIDCL_HPC_TARGET=%q\n' "$TARGET"
    printf 'GRIDCL_HPC_SCHEDULER=%q\n' "$CFG_SCHEDULER"
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
    echo "export GRIDCL_TEST_DEVICE=$(hpc_target_device "$TARGET")"
    if [ -n "$OPT_RANKS" ]; then
        echo "export GRIDCL_RANKS=$OPT_RANKS"
    fi
    if [ "${OPT_NODES:-1}" -gt 1 ]; then
        # shellcheck disable=SC2016 # expanded by the job, not here
        echo 'export GRIDCL_MPIRUN_FLAGS="${GRIDCL_MPIRUN_FLAGS:+$GRIDCL_MPIRUN_FLAGS }--map-by node"'
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
    hpc_remote r_fetch "$CFG_SCHEDULER" "$TREE" "$TARGET" "$CFG_LOGIN_MODULES" \
        "$(submit_cargo_features "$CMD")" $manifests
    hpc_remote_check $? "the fetch step"
}

# submit_job: submits the rendered job script (qsub on SGE, sbatch on Slurm); sets JOB_ID
# and JOB_SUBMITTED.
submit_job() {
    local script meta out tool=qsub
    [ "$CFG_SCHEDULER" != slurm ] || tool=sbatch
    script=$(submit_render) || exit 125
    meta="$(printf 'target=%q\nscheduler=%s\n' "$TARGET" "$CFG_SCHEDULER")"$'\n'"$(sync_state_env)"
    out=$(hpc_remote r_submit "$TREE" "$script" "$meta" "$(printf 'command=%q' "$CMD")" "$CFG_SCHEDULER")
    hpc_remote_check $? "$tool"
    JOB_ID=$(printf '%s\n' "$out" | sed -n 's/^job_id=//p' | head -n 1)
    JOB_SUBMITTED=$(printf '%s\n' "$out" | sed -n 's/^submitted=//p' | head -n 1)
    case $JOB_ID in
        '' | *[!0-9]*) hpc_die 125 "$tool returned no job id: $out" ;;
    esac
    JOB_SUBMITTED_LOCAL=$SECONDS
    hpc_msg "submitted job $JOB_ID to $TARGET"
}

# submit_wait: polls until the job has ended or --max-wait has passed since submission;
# sets JOB_STATE to done (exit status written), gone (ended without one), cancelled
# (cancel.sh), walltime (a direct job stopped at --time) or timeout, and JOB_ENDED=yes when
# the job is known to have ended. On direct targets `timeout` on the node enforces
# --max-wait; the runner's own deadline follows 30 s later as a backstop.
submit_wait() {
    local deadline=$((JOB_SUBMITTED_LOCAL + MAX_WAIT_SECONDS)) state rc last="" gone=0 fails=0
    JOB_ENDED=""
    [ "$CFG_SCHEDULER" != direct ] || deadline=$((deadline + 30))
    while :; do
        state=$(hpc_remote r_poll "$TREE" "$JOB_ID" "$CFG_SCHEDULER")
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
                done | cancelled)
                    JOB_STATE=$state
                    JOB_ENDED=yes
                    return 0
                    ;;
                limit)
                    # timeout on the node stopped the job at the smaller of --time and --max-wait.
                    if [ "$TIME_SECONDS" -lt "$MAX_WAIT_SECONDS" ]; then
                        JOB_STATE=walltime
                    else
                        JOB_STATE=timeout
                    fi
                    JOB_ENDED=yes
                    return 0
                    ;;
                gone)
                    gone=$((gone + 1))
                    if [ "$gone" -ge 3 ]; then
                        JOB_STATE=gone
                        JOB_ENDED=yes
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
    case $CFG_SCHEDULER in
        sge | slurm) hpc_remote r_delete "$TREE" "$JOB_ID" "$CFG_SCHEDULER" >/dev/null ;;
        direct) hpc_remote r_direct_stop "$TREE" "$JOB_ID" >/dev/null ;;
    esac
}

# --- remote side ------------------------------------------------------------------

r_fetch_steps() { # r_fetch_steps <tree> <target> <login modules> <features> [manifest...]
    local tree=$1 target=$2 features=$4 m
    shift 4
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
        # The LLVM bundle is downloaded by a build script below cubecl-llvm. cubecl-llvm may
        # be in the lock file without being activated by the command's features, and
        # `cargo build -p` accepts no features for it, so `cargo check` with the command's
        # features runs the build scripts (no code generation). `cargo tree` exits 0
        # either way, so its output decides whether that is needed.
        # shellcheck disable=SC2086 # the features are a list of words
        if cargo tree "$@" $features -e normal,build -i cubecl-llvm --depth 0 2>/dev/null |
            grep -q '^cubecl-llvm '; then
            echo "+ cargo check $* $features"
            # shellcheck disable=SC2086
            cargo check "$@" $features || return 1
        else
            echo "cubecl-llvm not activated for ${m} ${features}; LLVM bundle not needed"
        fi
    done
}

r_fetch() { # r_fetch <scheduler> <tree> <target> <login modules> <features> [manifest...]: log in .gridcl-hpc/
    local log rc
    r_env "$1"
    shift
    log=$1/.gridcl-hpc/fetch-$2.log
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

# r_submit <tree> <job script> <job lines> <command line> [scheduler]: prints job_id=,
# submitted=. The scheduler is sge (qsub -terse; the default) or slurm (sbatch --parsable,
# which prints <id> or <id>;<cluster>).
r_submit() {
    local tree=$1 scheduler=${5:-sge} file out id now rc tool
    r_env "$scheduler"
    mkdir -p "$tree/results" "$tree/.gridcl-hpc/jobs" || return 1
    file=$tree/.gridcl-hpc/submit-$$.sh
    printf '%s\n' "$2" >"$file" || return 1
    if [ "$scheduler" = slurm ]; then
        tool="sbatch --parsable"
        out=$(cd "$tree" && sbatch --parsable "$file" </dev/null)
    else
        tool="qsub -terse"
        out=$(cd "$tree" && qsub -terse "$file" </dev/null)
    fi
    rc=$?
    rm -f "$file"
    if [ "$rc" != 0 ]; then
        echo "$tool failed (status $rc): $out" >&2
        return 1
    fi
    id=$(printf '%s\n' "$out" | sed -n 's/^\([0-9][0-9]*\)\([.;].*\)\{0,1\}$/\1/p' | tail -n 1)
    if [ -z "$id" ]; then
        echo "$tool printed no job id: $out" >&2
        return 1
    fi
    now=$(date +%s)
    printf '%s\nsubmitted=%s\n' "$3" "$now" >"$tree/.gridcl-hpc/jobs/$id"
    mkdir -p "$tree/results/$id"
    printf '%s\n%s\nsubmitted=%s\n' "$3" "$4" "$now" >"$tree/results/$id/submit.env"
    echo "job_id=$id"
    echo "submitted=$now"
}

# r_poll <tree> <job id> <scheduler>: prints queued, running, done, cancelled (cancel.sh),
# limit (a direct job stopped by timeout on the node) or gone (ended without an exit status)
r_poll() {
    local results=$1/results/$2
    r_env "$3"
    if [ -f "$results/exit_status" ]; then
        echo "done"
    elif r_job_alive "$1" "$2" "$3"; then
        if [ -f "$results/times.env" ] || [ "$3" = direct ]; then echo running; else echo queued; fi
    elif [ -f "$results/exit_status" ]; then
        echo "done"
    elif [ -f "$results/cancelled" ]; then
        echo cancelled
    elif [ "$3" = direct ] && grep -Eqx '124|137' "$results/direct_status" 2>/dev/null; then
        echo limit
    else
        echo gone
    fi
}

r_delete() { # r_delete <tree> <job id> [scheduler]: qdel (sge, the default) or scancel (slurm)
    r_env "${3:-sge}"
    if [ "${3:-sge}" = slurm ]; then
        scancel "$2" </dev/null
    else
        qdel "$2" </dev/null
    fi
}

r_status_sge() { # r_status_sge: the user's SGE jobs (qstat; nothing when there are none)
    r_env sge
    qstat -u "${USER:-$(id -un)}" </dev/null
}

r_status_slurm() { # r_status_slurm: the user's Slurm jobs (squeue; nothing when there are none)
    local out
    r_env slurm
    out=$(squeue -h -u "${USER:-$(id -un)}" -o '%.10i %.9q %.12j %.8T %.10M %.6D %R' </dev/null) || return 1
    [ -n "$out" ] || return 0
    printf '%10s %9s %12s %8s %10s %6s %s\n' JOBID QOS NAME STATE TIME NODES 'NODELIST(REASON)'
    printf '%s\n' "$out"
}

# r_cancel_sge <job id> <marker> <base>...: writes the marker to results/<id>/cancelled of the
# tree below the bases that submitted the job (so its runner reports the cancellation), then
# deletes the job. Status 0 deleted, 1 qdel failed, 4 no such job.
r_cancel_sge() {
    local id=$1 marker=$2 b f tree
    shift 2
    r_env sge
    qstat -j "$id" >/dev/null 2>&1 || return 4
    for b in "$@"; do
        for f in "$b"/*/.gridcl-hpc/jobs/"$id"; do
            [ -f "$f" ] || continue
            tree=${f%/.gridcl-hpc/*}
            mkdir -p "$tree/results/$id" && echo "$marker" >"$tree/results/$id/cancelled"
        done
    done
    qdel "$id" </dev/null >/dev/null
}

# r_cancel_slurm <job id> <marker> <base>...: as r_cancel_sge, with squeue and scancel.
r_cancel_slurm() {
    local id=$1 marker=$2 b f tree
    shift 2
    r_env slurm
    [ -n "$(squeue -h -j "$id" -o %i 2>/dev/null </dev/null)" ] || return 4
    for b in "$@"; do
        for f in "$b"/*/.gridcl-hpc/jobs/"$id"; do
            [ -f "$f" ] || continue
            tree=${f%/.gridcl-hpc/*}
            mkdir -p "$tree/results/$id" && echo "$marker" >"$tree/results/$id/cancelled"
        done
    done
    scancel "$id" </dev/null >/dev/null
}
