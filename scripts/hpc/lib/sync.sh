# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Sync of the working tree to the remote tree of its branch (scripts/hpc, step 2).
#
# Remote layout, below the target's BASE:
#   <BASE>/<branch with '/' -> '_'>/       the tree: tracked and untracked, not ignored files
#     results/<job id>/                     job results (never deleted by a sync)
#     target/<target>/                      CARGO_TARGET_DIR of the jobs of <target>
#     target/<target>-login/                CARGO_TARGET_DIR of the login-node fetch step
#     .gridcl-hpc/tree.env                  marker and state of the last sync (below)
#     .gridcl-hpc/jobs/<job id>             one file per job not yet collected (below)
#     .gridcl-hpc/run/<job id>.{pid,wrap}   pid file and wrapper of a direct job (lib/direct.sh)
#     .gridcl-hpc/fetch-<target>.log        log of the last fetch step
# tree.env and the job files are bash-sourceable key=value lines, values quoted with
# printf %q. tree.env: gridcl_hpc_tree=1, branch, commit, dirty (yes|no), dirty_hash (empty
# when clean), synced (epoch seconds). A job file: target, scheduler, branch, commit,
# dirty, dirty_hash, submitted. A tree without .gridcl-hpc/tree.env is not a runner tree.

# sync_git_state: sets GIT_BRANCH, GIT_COMMIT, GIT_DIRTY (yes|no), GIT_DIRTY_HASH (empty
# when clean) and TREE_NAME for the working tree at $HPC_ROOT.
sync_git_state() {
    local status
    GIT_COMMIT=$(git -C "$HPC_ROOT" rev-parse --verify -q HEAD) ||
        hpc_die 125 "$HPC_ROOT is not a git working tree with a commit"
    GIT_BRANCH=$(git -C "$HPC_ROOT" symbolic-ref --quiet --short HEAD) ||
        GIT_BRANCH="detached-$(git -C "$HPC_ROOT" rev-parse --short=12 HEAD)"
    status=$(git -C "$HPC_ROOT" status --porcelain --untracked-files=all) ||
        hpc_die 125 "git status failed in $HPC_ROOT"
    if [ -n "$status" ]; then
        GIT_DIRTY=yes
        GIT_DIRTY_HASH=$(sync_dirty_hash) || hpc_die 125 "cannot hash the local changes"
    else
        GIT_DIRTY=no
        GIT_DIRTY_HASH=""
    fi
    TREE_NAME=$(printf '%s' "$GIT_BRANCH" | tr '/' '_' | tr -c 'A-Za-z0-9._-' '_')
    case $TREE_NAME in
        "" | .*) hpc_die 125 "branch '$GIT_BRANCH' gives no usable tree name" ;;
    esac
}

# sync_dirty_hash: a hash of the local changes against HEAD: the diff of tracked files and
# the names and contents of untracked files that are not ignored.
sync_dirty_hash() {
    (
        cd "$HPC_ROOT" || exit 1
        {
            git diff --binary HEAD
            git ls-files -z --others --exclude-standard |
                while IFS= read -r -d '' f; do
                    printf '%s\n' "$f"
                    git hash-object -- "$f"
                done
        } | git hash-object --stdin
    )
}

sync_check_rsync() { # rsync 3.x is needed for the filter rules
    local major
    command -v rsync >/dev/null 2>&1 || hpc_die 125 "rsync not found; install rsync 3.x (brew install rsync)"
    major=$(rsync --version 2>/dev/null | sed -n 's/.*version \([0-9][0-9]*\)\..*/\1/p' | head -n 1)
    case $major in
        '' | 0 | 1 | 2)
            hpc_die 125 "$(command -v rsync) is not rsync 3.x; install it (brew install rsync)" \
                "and put it first on PATH"
            ;;
    esac
}

# sync_state_env: the state of the working tree as sourceable lines (see the header).
sync_state_env() {
    printf 'branch=%q\ncommit=%q\ndirty=%q\ndirty_hash=%q\n' \
        "$GIT_BRANCH" "$GIT_COMMIT" "$GIT_DIRTY" "$GIT_DIRTY_HASH"
}

# sync_prepare: creates the remote tree and refuses (exit 125) when a job of the tree is
# still pending (queued or running) and the working tree differs from the one it uses.
sync_prepare() {
    local rc
    hpc_remote r_prepare "$TREE" "$CFG_SCHEDULER" "$GIT_COMMIT" "$GIT_DIRTY_HASH"
    rc=$?
    if [ "$rc" = 3 ]; then
        hpc_die 125 "refusing to sync: the working tree (commit $(git -C "$HPC_ROOT" rev-parse --short=12 HEAD)," \
            "dirty: $GIT_DIRTY) differs from the one the pending job(s) use. Wait for them," \
            "or cancel them with scripts/hpc/cancel.sh <job id>."
    fi
    hpc_remote_check "$rc" "preparing $TREE"
}

# sync_tree: copies the working tree to the remote tree. .gitignore rules apply on both
# sides, so ignored files are neither sent nor deleted; results/ and .gridcl-hpc/ are
# protected explicitly.
sync_tree() {
    hpc_msg "syncing $HPC_ROOT to $CFG_HOST:$TREE"
    rsync -az --delete \
        --filter='P /results/' --filter='- /results/' \
        --filter='P /.gridcl-hpc/' --filter='- /.gridcl-hpc/' \
        --filter='- /.git' \
        --filter=':- .gitignore' \
        -e "ssh $HPC_SSH_OPTS" \
        "$HPC_ROOT/" "$CFG_HOST:$TREE/" ||
        hpc_die 125 "rsync to $CFG_HOST:$TREE failed"
    hpc_remote r_record_tree "$TREE" "$(sync_state_env)"
    hpc_remote_check $? "recording the tree state"
    sync_prune
}

# sync_prune: removes the trees of merged branches below $CFG_BASE; called after every
# sync. A tree is removed only if it is a runner tree (.gridcl-hpc/tree.env) other than the
# current one, its branch is not main, its recorded commit is a proper ancestor of the local
# main (origin/main when there is no local main), and no job of it is pending. Merged
# means: the commit is known locally, is an ancestor of main and is not main's tip, so the
# tree of a branch just created from main is kept. Failures only print a warning.
sync_prune() {
    local main trees name commit branch merged="" out line
    main=$(git -C "$HPC_ROOT" rev-parse --verify -q 'refs/heads/main^{commit}' ||
        git -C "$HPC_ROOT" rev-parse --verify -q 'refs/remotes/origin/main^{commit}') || return 0
    if ! trees=$(hpc_remote r_list_trees "$CFG_BASE"); then
        hpc_msg "warning: cannot list the trees in $CFG_BASE; none pruned"
        return 0
    fi
    while read -r name commit branch; do
        [ -n "$name" ] && [ -n "$commit" ] || continue
        [ "$name" != "$TREE_NAME" ] && [ "$branch" != main ] && [ "$commit" != "$main" ] || continue
        git -C "$HPC_ROOT" cat-file -e "$commit^{commit}" 2>/dev/null || continue
        git -C "$HPC_ROOT" merge-base --is-ancestor "$commit" "$main" || continue
        merged="$merged $name"
    done <<<"$trees"
    [ -n "$merged" ] || return 0
    # shellcheck disable=SC2086 # one word per tree name
    if ! out=$(hpc_remote r_prune "$CFG_SCHEDULER" "$CFG_BASE" "$TREE_NAME" $merged); then
        hpc_msg "warning: pruning merged trees in $CFG_BASE failed"
    fi
    while IFS= read -r line; do
        [ -z "$line" ] || hpc_msg "prune: $line"
    done <<<"$out"
}

# --- remote side ------------------------------------------------------------------

r_env() { # r_env <scheduler>: the login environment of a non-interactive ssh session
    local f
    if { { [ "$1" = sge ] && ! command -v qsub >/dev/null 2>&1; } ||
        { [ "$1" = slurm ] && ! command -v sbatch >/dev/null 2>&1; }; } && [ -r /etc/profile ]; then
        # shellcheck source=/dev/null
        . /etc/profile >/dev/null 2>&1
    fi
    # The `module` function of the login shell is not exported to `bash -s` (Myriad).
    if ! type module >/dev/null 2>&1; then
        for f in "${MODULESHOME:+$MODULESHOME/init/bash}" /etc/profile.d/modules.sh; do
            if [ -n "$f" ] && [ -r "$f" ]; then
                # shellcheck source=/dev/null
                . "$f" >/dev/null 2>&1
                break
            fi
        done
    fi
    if [ -d "$HOME/.cargo/bin" ]; then
        PATH=$HOME/.cargo/bin:$PATH
        export PATH
    fi
}

r_job_alive() { # r_job_alive <tree> <job id> <scheduler>: the job is queued or running
    case $3 in
        sge) qstat -j "$2" >/dev/null 2>&1 ;;
        slurm) [ -n "$(squeue -h -j "$2" -o %i 2>/dev/null)" ] ;;
        direct) r_direct_alive "$1" "$2" ;;
        *) return 0 ;; # unknown kinds count as pending
    esac
}

r_job_pending() { # r_job_pending <job file>: the job is queued or running
    local scheduler
    scheduler=$(sed -n 's/^scheduler=//p' "$1" | head -n 1)
    r_job_alive "${1%/.gridcl-hpc/jobs/*}" "${1##*/}" "$scheduler"
}

r_pending_jobs() { # r_pending_jobs <tree>: ids of pending jobs; drops files of ended jobs
    local f
    for f in "$1"/.gridcl-hpc/jobs/*; do
        [ -f "$f" ] || continue
        if r_job_pending "$f"; then
            echo "${f##*/}"
        else
            rm -f "$f"
        fi
    done
}

r_prepare() { # r_prepare <tree> <scheduler> <commit> <dirty hash>: status 3 = busy
    local tree=$1 ids
    r_env "$2"
    mkdir -p "$tree/.gridcl-hpc/jobs" "$tree/results" || return 1
    ids=$(r_pending_jobs "$tree")
    [ -n "$ids" ] || return 0
    if [ -f "$tree/.gridcl-hpc/tree.env" ] &&
        (
            commit="" dirty_hash=""
            # shellcheck source=/dev/null
            . "$tree/.gridcl-hpc/tree.env"
            [ "$commit" = "$3" ] && [ "$dirty_hash" = "$4" ]
        ); then
        echo "pending job(s) $(echo "$ids" | tr '\n' ' ')use $tree; its sources are unchanged" >&2
        return 0
    fi
    echo "pending job(s) $(echo "$ids" | tr '\n' ' ')use $tree" >&2
    return 3
}

r_record_tree() { # r_record_tree <tree> <state lines>
    local tree=$1
    {
        echo "gridcl_hpc_tree=1"
        printf '%s\n' "$2"
        echo "synced=$(date +%s)"
    } >"$tree/.gridcl-hpc/tree.env.tmp" &&
        mv -f "$tree/.gridcl-hpc/tree.env.tmp" "$tree/.gridcl-hpc/tree.env"
}

r_list_trees() { # r_list_trees <base>: "<name> <commit> <branch>" for every runner tree
    local d
    for d in "$1"/*/; do
        d=${d%/}
        [ -f "$d/.gridcl-hpc/tree.env" ] && [ ! -L "$d" ] || continue
        (
            commit="" branch=""
            # shellcheck source=/dev/null
            . "$d/.gridcl-hpc/tree.env"
            printf '%s %s %s\n' "${d##*/}" "$commit" "$branch"
        )
    done
}

r_prune() { # r_prune <scheduler> <base> <current tree name> <tree name>...: removes the trees
    local base=$2 current=$3 name tree ids
    r_env "$1"
    shift 3
    case $base in
        /?*) ;;
        *) return 1 ;;
    esac
    for name in "$@"; do
        case $name in
            '' | . | .. | */* | "$current") continue ;;
        esac
        tree=$base/$name
        [ -d "$tree" ] && [ ! -L "$tree" ] && [ -f "$tree/.gridcl-hpc/tree.env" ] || continue
        ids=$(r_pending_jobs "$tree")
        if [ -n "$ids" ]; then
            echo "kept $tree: pending job(s) $(echo "$ids" | tr '\n' ' ')"
            continue
        fi
        if rm -rf -- "$tree"; then
            echo "removed $tree (merged into main)"
        else
            echo "could not remove $tree"
        fi
    done
}
