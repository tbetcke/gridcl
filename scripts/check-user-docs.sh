#!/usr/bin/env bash
# The checks of the user documentation (ADR-0067, rule 7): builds the site (the book at site/, the
# API of the nine library crates at site/api/) and runs scripts/doc_audit.py on it.
#
# Usage: scripts/check-user-docs.sh [--no-build] [--audit-only CRATE] [--parmetis] [--verbose]
#   --no-build          check the site, doctest lists and logs of the last build
#   --audit-only CRATE  only the API checks (c), (d) and (e), for CRATE's items
#   --parmetis          the second audit: tetraloma-partition with `parmetis` instead of `metis`
#                       (metis and parmetis exclude each other); builds its API and doctest list
#                       into target/user-docs/parmetis and runs (c), (d) and (e) on it
#   --verbose           print every failure (they are always in target/user-docs/report.txt)
#
# Checks (scripts/doc_audit.py): (a) the site builds without warnings; (b) links and anchors
# resolve; (c) no developer citation in rendered docs; (d) every item of rule 6.2 links to the
# book; (e) every function of rule 6.3 has a running example or an allowlist entry; (f) the book's
# outline is complete and no chapter is a stub; (g) fences tagged, no indented code, every ignore
# block includes a program that run-examples.sh runs; (h) while D1 is open, the library sources
# differ from D1's base commit only in comments.
#
# The cargo steps run under a lock (target/user-docs.lock), so one build runs at a time; the
# checks of --no-build need no lock. Needs the tools of scripts/install-doc-tools.sh and the system
# libraries of the site's features (MPI, a parallel HDF5, Scotch and PT-Scotch;
# docs/dev-environment.md has the local flags).
#
# Exit status: 0 if every check passes, 1 if one fails, 2 on a usage or build-tool error.
set -euo pipefail

cd "$(dirname "$0")/.."

# The features of the site (ADR-0067, rule 2), and of the parmetis audit.
SITE_FEATURES="tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis"
PARMETIS_FEATURES="tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/parmetis"
# The doctests also build the dev-dependency tetraloma-testing, whose test communicator needs its
# own `mpi` feature with tetraloma/mpi.
DOCTEST_EXTRA="tetraloma-testing/mpi"

SITE=${TETRALOMA_SITE_DIR:-site}
WORK=target/user-docs
PYTHON=${TETRALOMA_PYTHON:-python3}

die() {
    echo "check-user-docs.sh: $*" >&2
    exit 2
}

build=1
only=
parmetis=
verbose=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-build) build= ;;
        --audit-only)
            [[ $# -ge 2 ]] || die "--audit-only needs a crate"
            only=$2
            shift
            ;;
        --audit-only=*) only=${1#--audit-only=} ;;
        --parmetis) parmetis=1 ;;
        --verbose) verbose=(--verbose) ;;
        -h | --help)
            sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
    shift
done
if [[ -n $only ]]; then
    case " tetraloma tetraloma-dims tetraloma-refcell tetraloma-device tetraloma-parallel tetraloma-topology tetraloma-geometry tetraloma-partition tetraloma-io " in
        *" ${only//_/-} "*) ;;
        *) die "--audit-only '$only' is not one of the nine library crates" ;;
    esac
fi

mkdir -p "$WORK/logs"

# A lock for the cargo steps: mkdir is atomic, and flock is not on macOS.
lock=target/user-docs.lock
take_lock() {
    local waited=0
    until mkdir "$lock" 2>/dev/null; do
        if [[ $waited -eq 0 ]]; then
            echo "check-user-docs.sh: waiting for $lock (another build; remove it if stale)" >&2
        fi
        sleep 5
        waited=$((waited + 5))
        [[ $waited -lt 7200 ]] || die "gave up waiting for $lock"
    done
    trap 'rmdir "$lock" 2>/dev/null || true' EXIT
}

# The API docs of the crates of $1 (lines "CRATE FEATURES" of doc_audit.py api-plan), one
# `cargo doc` per crate in that order into an empty target/doc, so that each crate links to the
# docs of the crates before it; then copied to $2.
build_api() {
    local plan=$1 out=$2 crate feats status=0
    rm -rf target/doc
    while read -r crate feats; do
        local flags=()
        [[ $feats == - ]] || flags=(--features "$feats")
        echo "check-user-docs.sh: cargo doc --no-deps -p $crate ${flags[*]+"${flags[*]}"}"
        RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }-Dwarnings" \
            cargo doc --no-deps -p "$crate" ${flags[@]+"${flags[@]}"} \
            >"$WORK/logs/rustdoc-$crate.log" 2>&1 || {
            status=$?
            echo "check-user-docs.sh: cargo doc of $crate failed (exit $status); see $WORK/logs/rustdoc-$crate.log" >&2
        }
    done <"$plan"
    rm -rf "$out"
    mkdir -p "$(dirname "$out")"
    cp -R target/doc "$out"
    # The landing page of /api/ (rustdoc's help and settings pages link to it): the crates.
    {
        echo '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
        echo '<title>tetraloma API</title></head><body><h1>tetraloma API</h1>'
        echo '<p>The <a href="../index.html">user book</a>, and the API of each crate:</p><ul>'
        while read -r crate feats; do
            echo "<li><a href=\"${crate//-/_}/index.html\">$crate</a></li>"
        done <"$plan"
        echo '</ul></body></html>'
    } >"$out/index.html"
    return $status
}

# The doctests of the crates of $1 with features $2, listed (all, and the ignored ones) into $3.
list_doctests() {
    local crates=() c
    for c in $1; do crates+=(-p "$c"); done
    cargo test --doc "${crates[@]}" --features "$2" -- --list >"$3.txt" 2>"$WORK/logs/doctest-list.log" ||
        die "cargo test --doc -- --list failed; see $WORK/logs/doctest-list.log"
    cargo test --doc "${crates[@]}" --features "$2" -- --list --ignored >"$3-ignored.txt" \
        2>>"$WORK/logs/doctest-list.log" ||
        die "cargo test --doc -- --list --ignored failed; see $WORK/logs/doctest-list.log"
}

if [[ -n $parmetis ]]; then
    pwork=$WORK/parmetis
    mkdir -p "$pwork"
    if [[ -n $build ]]; then
        take_lock
        "$PYTHON" scripts/doc_audit.py api-plan --features "$PARMETIS_FEATURES" \
            --crates tetraloma-partition >"$pwork/api-order.txt"
        build_api "$pwork/api-order.txt" "$pwork/api" || true
        list_doctests tetraloma-partition "tetraloma-partition/parmetis,tetraloma-partition/scotch,tetraloma-partition/ptscotch" "$pwork/doctests"
    fi
    "$PYTHON" scripts/doc_audit.py check --site "$SITE" --work "$pwork" --api "$pwork/api" \
        --checks cde --crate tetraloma-partition ${verbose[@]+"${verbose[@]}"}
    exit $?
fi

if [[ -n $build ]]; then
    command -v mdbook >/dev/null && command -v mdbook-katex >/dev/null ||
        die "mdbook or mdbook-katex missing: run scripts/install-doc-tools.sh"
    scripts/install-doc-tools.sh --check >/dev/null ||
        die "the doc tools are not at their pinned versions: run scripts/install-doc-tools.sh"
    take_lock
    rm -rf "$SITE"
    echo "check-user-docs.sh: mdbook build docs/user -> $SITE"
    mdbook build docs/user -d "$PWD/$SITE" >"$WORK/logs/mdbook.log" 2>&1 ||
        echo "check-user-docs.sh: mdbook build failed; see $WORK/logs/mdbook.log" >&2
    rm -f "$WORK"/logs/rustdoc-*.log
    "$PYTHON" scripts/doc_audit.py api-plan --features "$SITE_FEATURES" >"$WORK/api-order.txt"
    build_api "$WORK/api-order.txt" "$SITE/api" || true
    list_doctests "$(cut -d' ' -f1 "$WORK/api-order.txt" | tr '\n' ' ')" \
        "$SITE_FEATURES,$DOCTEST_EXTRA" "$WORK/doctests"
fi

[[ -d $SITE ]] || die "no site at $SITE: run without --no-build first"
if [[ -n $only ]]; then
    "$PYTHON" scripts/doc_audit.py check --site "$SITE" --work "$WORK" --checks cde --crate "$only" \
        ${verbose[@]+"${verbose[@]}"}
else
    "$PYTHON" scripts/doc_audit.py check --site "$SITE" --work "$WORK" ${verbose[@]+"${verbose[@]}"}
fi
