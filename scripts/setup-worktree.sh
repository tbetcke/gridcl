#!/usr/bin/env bash
# Set up the git-ignored local files a new git worktree lacks (docs/dev-environment.md):
# - .venv with the pinned Python packages of scripts/check-local-tools.sh (the M5 oracles and
#   the M4 fixture corpus), created with python3 if missing;
# - scripts/hpc/targets.env, copied from the main checkout if missing there and present here.
# Then runs scripts/check-local-tools.sh. Safe to rerun: existing files are kept.
#
# pip needs pypi.org and files.pythonhosted.org; inside Claude Code's sandbox the proxy breaks
# pip's certificate check, so run this script outside it.
#
# usage: scripts/setup-worktree.sh
set -eu

repo=$(cd "$(dirname "$0")/.." && pwd)
main=$(dirname "$(git -C "$repo" rev-parse --path-format=absolute --git-common-dir)")
pins=$(sed -n 's/^pins="\(.*\)"$/\1/p' "$repo/scripts/check-local-tools.sh")
if [ -z "$pins" ]; then
    echo "setup-worktree.sh: no pins line in scripts/check-local-tools.sh" >&2
    exit 1
fi

if [ ! -x "$repo/.venv/bin/python" ]; then
    echo "setup-worktree.sh: creating .venv"
    python3 -m venv "$repo/.venv"
fi
# shellcheck disable=SC2086 # pins is a space-separated list of requirements
"$repo/.venv/bin/pip" install --quiet --disable-pip-version-check $pins

targets=scripts/hpc/targets.env
if [ -f "$repo/$targets" ]; then
    echo "setup-worktree.sh: $targets present"
elif [ "$main" != "$repo" ] && [ -f "$main/$targets" ]; then
    cp "$main/$targets" "$repo/$targets"
    echo "setup-worktree.sh: copied $targets from the main checkout"
else
    echo "setup-worktree.sh: no $targets here or in the main checkout; remote runs need one" \
        "(scripts/hpc/targets.example.env)"
fi

"$repo/scripts/check-local-tools.sh"
