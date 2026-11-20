#!/usr/bin/env bash
# Builds the user documentation site (ADR-0067, rules 2 and 3): the book (mdBook, docs/user) at
# OUT/, and at OUT/api/ the rustdoc of the nine published library crates with the site's features.
# The CI job `docs` uploads OUT as the GitHub Pages artifact on main; scripts/check-user-docs.sh
# builds through this script and checks the result.
#
# Usage: scripts/build-site.sh [--out DIR] [--api-only] [--features FEATURES] [--crates C,...]
#                              [--logs DIR]
#   --out DIR            the site directory, emptied first (default: site)
#   --api-only           only the API, into DIR itself (no book; check-user-docs.sh --parmetis)
#   --features FEATURES  the features of the API build (default: the site's, below)
#   --crates C,...       only these library crates (default: all nine)
#   --logs DIR           the build logs (default: target/user-docs/logs): mdbook.log and one
#                        rustdoc-CRATE.log per crate, which check (a) searches for warnings; the
#                        documentation order goes to DIR/../api-order.txt
#
# The API is one `cargo doc --no-deps -p CRATE` per crate, dependencies first (doc_audit.py
# api-plan), each with that crate's features of the site build, into an empty target/doc:
# rustdoc links a crate to the docs of the crates documented before it. No unpublished crate and
# no test-support feature is built into the site.
#
# The build runs under the lock target/user-docs.lock (one build at a time), unless the caller
# holds it and sets TETRALOMA_USER_DOCS_LOCKED=1. Needs the tools of scripts/install-doc-tools.sh
# and the system libraries of the site's features (MPI, a parallel HDF5, Scotch and PT-Scotch;
# docs/dev-environment.md has the local flags).
#
# Exit status: 0 if the site built, 1 if mdBook or a rustdoc run failed (the logs say why), 2 on
# a usage or tool error. Warnings do not fail the build; check (a) of check-user-docs.sh does.
set -euo pipefail

cd "$(dirname "$0")/.."

# The features of the site (ADR-0067, rule 2): `metis`, not `parmetis` (they exclude each other,
# ADR-0033).
SITE_FEATURES="tetraloma/mpi,tetraloma/hdf5,tetraloma/scotch,tetraloma/ptscotch,tetraloma/metis"
PYTHON=${TETRALOMA_PYTHON:-python3}

die() {
    echo "build-site.sh: $*" >&2
    exit 2
}

out=site
api_only=
features=$SITE_FEATURES
crates=
logs=target/user-docs/logs
while [[ $# -gt 0 ]]; do
    case $1 in
        --out)
            [[ $# -ge 2 ]] || die "--out needs a directory"
            out=$2
            shift
            ;;
        --api-only) api_only=1 ;;
        --features)
            [[ $# -ge 2 ]] || die "--features needs a value"
            features=$2
            shift
            ;;
        --crates)
            [[ $# -ge 2 ]] || die "--crates needs a value"
            crates=$2
            shift
            ;;
        --logs)
            [[ $# -ge 2 ]] || die "--logs needs a directory"
            logs=$2
            shift
            ;;
        -h | --help)
            sed -n '2,28p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
    shift
done
[[ -n $out && $out != / && $out != . ]] || die "--out '$out' cannot be emptied"
[[ "$features" =~ ^[A-Za-z0-9_/,-]+$ ]] || die "--features '$features' is not a feature list"
[[ -z $crates || "$crates" =~ ^[A-Za-z0-9_,-]+$ ]] || die "--crates '$crates' is not a crate list"

if [[ -z $api_only ]]; then
    command -v mdbook >/dev/null && command -v mdbook-katex >/dev/null ||
        die "mdbook or mdbook-katex missing: run scripts/install-doc-tools.sh"
    scripts/install-doc-tools.sh --check >/dev/null ||
        die "the doc tools are not at their pinned versions: run scripts/install-doc-tools.sh"
fi

# A lock for the cargo steps: mkdir is atomic, and flock is not on macOS.
if [[ ${TETRALOMA_USER_DOCS_LOCKED:-} != 1 ]]; then
    lock=target/user-docs.lock
    mkdir -p target
    waited=0
    until mkdir "$lock" 2>/dev/null; do
        if [[ $waited -eq 0 ]]; then
            echo "build-site.sh: waiting for $lock (another build; remove it if stale)" >&2
        fi
        sleep 5
        waited=$((waited + 5))
        [[ $waited -lt 7200 ]] || die "gave up waiting for $lock"
    done
    trap 'rmdir "$lock" 2>/dev/null || true' EXIT
fi

mkdir -p "$logs"
order=$logs/../api-order.txt
status=0

if [[ -z $api_only ]]; then
    rm -rf "$out"
    rm -f "$logs/mdbook.log"
    echo "build-site.sh: mdbook build docs/user -> $out"
    # An absolute destination: mdBook resolves -d against the book's root.
    case $out in
        /*) dest=$out ;;
        *) dest=$PWD/$out ;;
    esac
    mdbook build docs/user -d "$dest" >"$logs/mdbook.log" 2>&1 || {
        status=1
        echo "build-site.sh: mdbook build failed; see $logs/mdbook.log" >&2
    }
    api=$out/api
else
    api=$out
fi

"$PYTHON" scripts/doc_audit.py api-plan --features "$features" ${crates:+--crates "$crates"} >"$order" ||
    die "doc_audit.py api-plan failed"
[[ -s $order ]] || die "no crates to document"

rm -f "$logs"/rustdoc-*.log
rm -rf target/doc
while read -r crate feats; do
    flags=()
    [[ $feats == - ]] || flags=(--features "$feats")
    echo "build-site.sh: cargo doc --no-deps -p $crate ${flags[*]+"${flags[*]}"}"
    RUSTDOCFLAGS="${RUSTDOCFLAGS:+$RUSTDOCFLAGS }-Dwarnings" \
        cargo doc --no-deps -p "$crate" ${flags[@]+"${flags[@]}"} \
        >"$logs/rustdoc-$crate.log" 2>&1 || {
        status=1
        echo "build-site.sh: cargo doc of $crate failed; see $logs/rustdoc-$crate.log" >&2
    }
done <"$order"

rm -rf "$api"
mkdir -p "$(dirname "$api")"
cp -R target/doc "$api"
# The landing page of /api/ (rustdoc's help and settings pages link to it): the crates.
{
    echo '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">'
    echo '<meta name="viewport" content="width=device-width, initial-scale=1">'
    echo '<title>tetraloma API</title></head><body><h1>tetraloma API</h1>'
    if [[ -z $api_only ]]; then
        echo '<p>The <a href="../index.html">user book</a>, and the API of each crate:</p><ul>'
    else
        echo '<p>The API of each crate:</p><ul>'
    fi
    while read -r crate feats; do
        echo "<li><a href=\"${crate//-/_}/index.html\">$crate</a></li>"
    done <"$order"
    echo '</ul></body></html>'
} >"$api/index.html"

if [[ $status -eq 0 ]]; then
    echo "build-site.sh: site at $out"
fi
exit $status
