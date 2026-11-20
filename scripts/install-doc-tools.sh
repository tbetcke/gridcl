#!/usr/bin/env bash
# Installs the pinned book tools (ADR-0067, rule 3): mdBook and the mdbook-katex preprocessor,
# a tested pair of the same mdBook major version (docs/plans/D1.md, question 5): mdbook-katex
# 0.10.0 is built against mdBook 0.5.3 and warns under any other mdBook. Installed with
# `cargo install --locked` into cargo's bin directory (~/.cargo/bin; CI caches it).
#
# Usage: scripts/install-doc-tools.sh [--check]
#   --check   install nothing; exit 1 unless both tools are installed at the pinned versions
#
# Skips a tool that is already installed at its pinned version.
set -euo pipefail

MDBOOK_VERSION=0.5.3
MDBOOK_KATEX_VERSION=0.10.0

# The installed version of tool $1 (its `--version` line's last word, without a leading v), or
# nothing.
installed() {
    local out
    out=$("$1" --version 2>/dev/null) || return 0
    out=${out##* }
    echo "${out#v}"
}

check=
case "${1:-}" in
    "") ;;
    --check) check=1 ;;
    *)
        echo "usage: scripts/install-doc-tools.sh [--check]" >&2
        exit 2
        ;;
esac

status=0
for pair in "mdbook $MDBOOK_VERSION" "mdbook-katex $MDBOOK_KATEX_VERSION"; do
    read -r tool version <<<"$pair"
    have=$(installed "$tool")
    if [[ $have == "$version" ]]; then
        echo "install-doc-tools: $tool $version installed"
    elif [[ -n $check ]]; then
        echo "install-doc-tools: $tool ${have:-missing}, want $version (run scripts/install-doc-tools.sh)" >&2
        status=1
    else
        echo "install-doc-tools: installing $tool $version (have ${have:-none})"
        cargo install --locked --version "=$version" "$tool"
    fi
done
exit $status
