#!/usr/bin/env bash
# usage: scripts/hpc/acceptance.sh <target>  - live E0 acceptance driver for a remote target.
# Runs, one job at a time: env-check x3 (turnaround), env-check with cuda-cpp, the 2-GPU MPI test,
# a failing command and a --max-wait overrun; then checks that status.sh is empty.
# Implemented in E0 T7.
echo "$(basename "$0"): not yet implemented" >&2
exit 1
