#!/usr/bin/env bash
# Run a command on a remote test target (docs/design.md, "Development environment and test targets").
# usage: scripts/hpc/remote-test.sh <target> [--gpus N] [--ranks N] [--cores N] [--time HH:MM:SS] [--max-wait HH:MM:SS] -- <command>
# Implemented in E0 T3 (SGE) and T4 (direct).
echo "$(basename "$0"): not yet implemented" >&2
exit 1
