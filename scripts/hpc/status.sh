#!/usr/bin/env bash
# usage: scripts/hpc/status.sh
# Lists the user's remote jobs on every target configured in targets.env: qstat on each SGE
# host, squeue on each Slurm host, and the running jobs of the runner on each direct target. Prints nothing when there
# are none. Exit status 0, or 125 when the settings are unusable or a target is unreachable
# (the others are still listed).
# shellcheck source-path=SCRIPTDIR
set -u
set -o pipefail

HPC_PROG=status.sh
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

main() {
    local kind remotes host targets bases out rc=0 found=""
    case ${1:-} in
        '') ;;
        -h | --help)
            sed -n '2,6p' "$0" | sed 's/^# \{0,1\}//'
            return 0
            ;;
        *)
            hpc_msg "no arguments expected"
            return 2
            ;;
    esac
    for kind in sge slurm direct; do
        remotes=$(config_remotes "$kind") || return 125
        while read -r host targets bases; do
            [ -n "$host" ] || continue
            found=1
            CFG_HOST=$host
            # shellcheck disable=SC2086 # one word per base directory
            case $kind in
                sge) out=$(hpc_remote r_status_sge) ;;
                slurm) out=$(hpc_remote r_status_slurm) ;;
                direct) out=$(hpc_remote r_direct_status $bases) ;;
            esac
            case $? in
                0) ;;
                255)
                    hpc_msg "cannot reach $host ($targets) over ssh; open the connection (ssh $host true)"
                    rc=125
                    continue
                    ;;
                *)
                    hpc_msg "listing the jobs on $host ($targets) failed"
                    rc=125
                    continue
                    ;;
            esac
            [ -n "$out" ] || continue
            echo "== $targets ($host, $kind)"
            if [ "$kind" = direct ]; then
                echo "job-ID  state  tree  command"
            fi
            printf '%s\n' "$out"
        done <<<"$remotes"
    done
    if [ -z "$found" ]; then
        hpc_msg "no target is configured in ${CFG_FILE:-targets.env}"
        return 125
    fi
    return "$rc"
}

main "$@"
