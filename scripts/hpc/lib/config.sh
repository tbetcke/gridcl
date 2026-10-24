# shellcheck shell=bash disable=SC2034,SC2153 # globals are shared between remote-test.sh and the libs
# Configuration and shared helpers of scripts/hpc, sourced by remote-test.sh.
#
# Settings come from scripts/hpc/targets.env (git-ignored; GRIDCL_TARGETS_ENV overrides the
# path), created from targets.example.env. The file is sourced by bash; the keys of target
# <t> carry the prefix <t> in upper case with '-' replaced by '_' (myriad-v100 ->
# MYRIAD_V100_HOST, ...). myriad-cpu is the one target without GPUs: its jobs run on Myriad's
# CPU-only D nodes with GRIDCL_TEST_DEVICE=cpu (ADR-0014).
#
# Exit statuses of the runner: 2 usage error, 124 --max-wait exceeded (job deleted),
# 125 runner failure (settings, ssh, sync, fetch, submission, job without exit status);
# otherwise the command's exit status.
#
# Remote functions are named r_*. hpc_remote ships every r_* function to the target's host
# (declare -f) and runs one of them there with `bash -s`, so the remote side needs nothing
# but bash; the functions must therefore run on both bash 3.2 and bash 4/5 and must not use
# here-documents.

HPC_TARGETS="myriad-v100 myriad-a100 myriad-cpu locust"
HPC_SSH_OPTS="-o BatchMode=yes"

hpc_msg() { # hpc_msg <text>: progress and errors, on stderr
    printf '%s: %s\n' "${HPC_PROG:-scripts/hpc}" "$*" >&2
}

hpc_die() { # hpc_die <status> <text>
    local status=$1
    shift
    hpc_msg "$*"
    exit "$status"
}

hpc_known_target() { # hpc_known_target <name>
    case " $HPC_TARGETS " in
        *" $1 "*) return 0 ;;
    esac
    return 1
}

hpc_target_device() { # hpc_target_device <target>: the GRIDCL_TEST_DEVICE of its jobs
    case $1 in
        myriad-cpu) printf 'cpu' ;;
        *) printf 'cuda' ;;
    esac
}

hpc_prefix() { # hpc_prefix <target>: the key prefix of a target (myriad-v100 -> MYRIAD_V100)
    printf '%s' "$1" | tr 'abcdefghijklmnopqrstuvwxyz-' 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'
}

hpc_replace() { # hpc_replace <string> <key> <value>: every <key> replaced literally
    local s=$1 k=$2 v=$3 out=""
    while :; do
        case $s in
            *"$k"*)
                out=$out${s%%"$k"*}$v
                s=${s#*"$k"}
                ;;
            *) break ;;
        esac
    done
    printf '%s' "$out$s"
}

hpc_seconds() { # hpc_seconds <HH:MM:SS>: the duration in seconds
    local h m s
    h=${1%%:*}
    m=${1#*:}
    m=${m%%:*}
    s=${1##*:}
    echo $((10#$h * 3600 + 10#$m * 60 + 10#$s))
}

config_value() { # config_value <key>: the value of a settings key, empty when unset
    local name=$1
    printf '%s' "${!name:-}"
}

# config_source: reads the settings file into the shell; sets CFG_FILE.
config_source() {
    CFG_FILE=${GRIDCL_TARGETS_ENV:-$HPC_DIR/targets.env}
    if [ ! -f "$CFG_FILE" ]; then
        hpc_die 125 "no settings file $CFG_FILE: copy scripts/hpc/targets.example.env to" \
            "scripts/hpc/targets.env and fill in the values (docs/dev-environment.md)"
    fi
    # shellcheck source=/dev/null # personal settings, not in the repository
    . "$CFG_FILE" || hpc_die 125 "cannot read the settings file $CFG_FILE"
}

config_has_target() { # config_has_target <target>: the settings file sets its HOST
    config_source
    [ -n "$(config_value "$(hpc_prefix "$1")_HOST")" ]
}

# config_remotes <scheduler>: one line per distinct host among the configured targets of a
# scheduler kind: "<host> <target>[,<target>...] <base>...". Exits 125 on a settings error.
config_remotes() {
    local t
    for t in $HPC_TARGETS; do
        config_has_target "$t" || continue
        config_load "$t"
        [ "$CFG_SCHEDULER" = "$1" ] || continue
        echo "$CFG_HOST $t $CFG_BASE"
    done | awk '
        !($1 in order) { order[$1] = ++n; host[n] = $1 }
        { targets[$1] = targets[$1] (targets[$1] == "" ? "" : ",") $2 }
        !(($1 SUBSEP $3) in seen) { seen[$1, $3] = 1; bases[$1] = bases[$1] " " $3 }
        END { for (i = 1; i <= n; i++) print host[i], targets[host[i]] bases[host[i]] }'
}

# config_load <target>: reads the settings file and sets CFG_HOST, CFG_SCHEDULER, CFG_BASE,
# CFG_LOGIN_MODULES, CFG_MODULES, CFG_CUDA_PATH, CFG_CORES_REQUEST and CFG_MEM_REQUEST.
config_load() {
    local target=$1 file prefix key missing=""
    config_source
    file=$CFG_FILE
    prefix=$(hpc_prefix "$target")
    for key in HOST SCHEDULER BASE; do
        [ -n "$(config_value "${prefix}_$key")" ] || missing="$missing ${prefix}_$key"
    done
    if [ -n "$missing" ]; then
        hpc_die 125 "$file does not set$missing (the keys are listed in scripts/hpc/targets.example.env)"
    fi
    CFG_HOST=$(config_value "${prefix}_HOST")
    CFG_SCHEDULER=$(config_value "${prefix}_SCHEDULER")
    CFG_BASE=$(config_value "${prefix}_BASE")
    CFG_LOGIN_MODULES=$(config_value "${prefix}_LOGIN_MODULES")
    CFG_MODULES=$(config_value "${prefix}_MODULES")
    CFG_CUDA_PATH=$(config_value "${prefix}_CUDA_PATH")
    CFG_CORES_REQUEST=$(config_value "${prefix}_CORES_REQUEST")
    CFG_MEM_REQUEST=$(config_value "${prefix}_MEM_REQUEST")

    case $CFG_HOST in
        *[[:space:]]* | -*) hpc_die 125 "${prefix}_HOST '$CFG_HOST' is not an SSH host alias" ;;
    esac
    case $CFG_SCHEDULER in
        sge | direct) ;;
        *) hpc_die 125 "${prefix}_SCHEDULER is '$CFG_SCHEDULER'; it must be sge or direct" ;;
    esac
    case $CFG_BASE in
        *[[:space:]]* | *[\'\"\$\`\\]*)
            hpc_die 125 "${prefix}_BASE must not contain spaces, quotes, \$, \` or \\"
            ;;
        / | /. | /.. | */./* | */../* | */. | */..)
            hpc_die 125 "${prefix}_BASE '$CFG_BASE' is not a usable base directory"
            ;;
        /*) ;;
        *) hpc_die 125 "${prefix}_BASE '$CFG_BASE' is not an absolute path" ;;
    esac
    CFG_BASE=${CFG_BASE%/}
    CFG_PREFIX=$prefix
}

# hpc_remote <function> [args...]: runs the remote function r_<...> on $CFG_HOST and returns
# its status (255: ssh failed). Remote stdout is this function's stdout; remote stderr goes
# to stderr. The script reaches `bash -s` on stdin; the functions read nothing from it.
hpc_remote() {
    local fn=$1 script arg
    shift
    # shellcheck disable=SC2046 # function names are single words
    script=$(declare -f $(compgen -A function r_))
    script="$script"$'\n'"$fn"
    for arg in "$@"; do
        script="$script $(printf '%q' "$arg")"
    done
    # shellcheck disable=SC2086 # HPC_SSH_OPTS is a list of options
    ssh $HPC_SSH_OPTS "$CFG_HOST" bash -s <<<"$script"
}

# hpc_remote_check <status> <what>: stops with 125 after a failed remote step.
hpc_remote_check() {
    case $1 in
        0) return 0 ;;
        255)
            hpc_die 125 "$2: cannot reach $CFG_HOST over ssh. Open the shared connection" \
                "in a terminal (ssh $CFG_HOST true) and try again."
            ;;
        *) hpc_die 125 "$2 failed on $CFG_HOST (status $1)" ;;
    esac
}
