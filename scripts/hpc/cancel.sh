#!/usr/bin/env bash
# usage: scripts/hpc/cancel.sh <job id>
# Deletes one remote job: an SGE job id (digits) on the SGE targets, a direct job id
# (<timestamp>-<pid>) on the direct targets of targets.env. The runner waiting for the job
# reports it as cancelled. Exit status 0 when deleted, 1 when no such job is queued or
# running, 2 on usage errors, 125 when the settings are unusable or a target is unreachable.
# shellcheck source-path=SCRIPTDIR
set -u
set -o pipefail

HPC_PROG=cancel.sh
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

usage() {
    sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
}

main() {
    local id kind remotes host targets bases marker unreachable=""
    case $# in
        1) id=$1 ;;
        *)
            usage >&2
            return 2
            ;;
    esac
    case $id in
        -h | --help)
            usage
            return 0
            ;;
    esac
    if printf '%s\n' "$id" | grep -Eqx '[0-9]+'; then
        kind=sge
    elif printf '%s\n' "$id" | grep -Eqx '[0-9]+-[0-9]+'; then
        kind=direct
    else
        hpc_msg "'$id' is not a job id (SGE: digits; direct: <timestamp>-<pid>)"
        return 2
    fi
    marker="cancelled with scripts/hpc/cancel.sh at $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    remotes=$(config_remotes "$kind") || return 125
    while read -r host targets bases; do
        [ -n "$host" ] || continue
        CFG_HOST=$host
        # shellcheck disable=SC2086 # one word per base directory
        case $kind in
            sge) hpc_remote r_cancel_sge "$id" "$marker" $bases ;;
            direct) hpc_remote r_direct_cancel "$id" "$marker" $bases ;;
        esac
        case $? in
            0)
                echo "cancelled job $id on $host ($targets)"
                return 0
                ;;
            4) ;;
            255)
                hpc_msg "cannot reach $host ($targets) over ssh; open the connection (ssh $host true)"
                unreachable=1
                ;;
            *)
                hpc_msg "deleting job $id on $host ($targets) failed"
                return 1
                ;;
        esac
    done <<<"$remotes"
    if [ -n "$unreachable" ]; then
        return 125
    fi
    hpc_msg "no $kind job $id is queued or running on the configured targets"
    return 1
}

main "$@"
