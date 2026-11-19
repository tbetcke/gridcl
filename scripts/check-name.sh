#!/usr/bin/env bash
# The project's name is tetraloma (ADR-0066, rule 5; E1 acceptance). Four checks:
#   (a) no tracked path and no tracked file content names the placeholder "gridcl" (any case),
#       outside the allowlist below;
#   (b) every workspace package except ptscotch-sys and parmetis-sys is named tetraloma or
#       tetraloma-*, and conformance/ is tetraloma-conformance;
#   (c) [workspace.package] repository is https://github.com/tetracl/tetraloma;
#   (d) no source file (anything but Markdown) reads or sets a GRIDCL_* environment variable.
#
# Allowlist (ADR-0066 rules 1 and 3): recorded data (docs/perf/data/, the bench crate's
# history/), the spack repository namespace in scripts/hpc/spack/ (its directory, `gridcl.llvm`,
# `namespace: 'gridcl'` and the repos key), and the three files that describe the rename: this
# script, ADR-0066 and the E1 plan.
#
# Usage: scripts/check-name.sh [-v]
#   -v  print every hit (default: the first 10 per check)
#
# Exit status: 0 if every check passes, 1 if one fails, 2 on an error.
set -euo pipefail

cd "$(dirname "$0")/.."

limit=10
if [[ "${1:-}" == "-v" ]]; then
    limit=0
elif [[ $# -gt 0 ]]; then
    echo "usage: scripts/check-name.sh [-v]" >&2
    exit 2
fi

# Paths allowed to contain the old name, and files whose content is not checked.
allowed_path_re='^scripts/hpc/spack/repo/spack_repo/gridcl/'
excluded_files=(
    ':(exclude)docs/perf/data/**'
    ':(exclude,glob)*-bench/history/**'
    ':(exclude)scripts/check-name.sh'
    ':(exclude)docs/adr/0066-the-project-is-named-tetraloma.md'
    ':(exclude)docs/plans/E1.md'
)

failed=0

show() { # show <hits>: print the first $limit lines (all with -v)
    if [[ -z "$1" ]]; then return; fi
    if [[ $limit -eq 0 ]]; then
        sed 's/^/    /' <<<"$1"
    else
        head -n "$limit" <<<"$1" | sed 's/^/    /'
        local n
        n=$(wc -l <<<"$1" | tr -d ' ')
        if [[ $n -gt $limit ]]; then echo "    ... $((n - limit)) more (-v lists all)"; fi
    fi
}

report() { # report <check> <count> <what> <hits>
    if [[ "$2" -eq 0 ]]; then
        echo "check-name: ($1) ok"
    else
        echo "check-name: ($1) FAIL: $2 $3"
        show "$4"
        failed=1
    fi
}

# (a) paths and content.
path_hits=$(git ls-files | grep -i 'gridcl' | grep -Ev "$allowed_path_re" || true)
# Lines with the old name; in the spack directory the namespace tokens are removed first. Each
# remaining line is printed with its count of occurrences.
content_hits=$(git grep -I -n -i 'gridcl' -- . "${excluded_files[@]}" | perl -ne '
    my ($file, $line, $text) = /^([^:]*):(\d+):(.*)$/ or next;
    if ($file =~ m{^scripts/hpc/spack/}) {
        $text =~ s{spack_repo/gridcl}{}g;
        $text =~ s{\bgridcl\.llvm\b}{}g;
        $text =~ s{namespace: .gridcl.}{}g;
        $text =~ s{^\s*gridcl(?=: \@ROOT\@)}{};
    }
    my $n = () = $text =~ /gridcl/gi;
    print "$n $file:$line: $text\n" if $n;
' || true)
path_count=$(grep -c . <<<"$path_hits" || true)
occurrences=$(awk '{ s += $1 } END { print s + 0 }' <<<"$content_hits")
files=$(cut -d' ' -f2 <<<"$content_hits" | cut -d: -f1 | grep . | sort -u | wc -l | tr -d ' ')
report "a: paths" "$path_count" "tracked paths name gridcl" "$path_hits"
report "a: content" "$occurrences" "occurrences of gridcl in $files files" \
    "$(cut -d' ' -f2- <<<"$content_hits" | cut -c1-160)"

# (b) package names.
names=$(cargo metadata --no-deps --format-version 1 --offline 2>/dev/null ||
    cargo metadata --no-deps --format-version 1) || exit 2
bad_names=$(python3 -c '
import json, re, sys
for p in json.load(sys.stdin)["packages"]:
    if p["name"] not in ("ptscotch-sys", "parmetis-sys") and not re.fullmatch(r"tetraloma(-.+)?", p["name"]):
        print(p["name"])
' <<<"$names") || exit 2
conformance=$(sed -n 's/^name *= *"\(.*\)"/\1/p' conformance/Cargo.toml | head -n 1)
if [[ "$conformance" != "tetraloma-conformance" ]]; then
    bad_names+=${bad_names:+$'\n'}"conformance/: $conformance"
fi
report "b: package names" "$(grep -c . <<<"$bad_names" || true)" "packages not named tetraloma-*" \
    "$bad_names"

# (c) repository.
repository=$(python3 -c '
import tomllib
with open("Cargo.toml", "rb") as f:
    print(tomllib.load(f).get("workspace", {}).get("package", {}).get("repository", ""))
') || exit 2
if [[ "$repository" == "https://github.com/tetracl/tetraloma" ]]; then
    report "c: repository" 0 "" ""
else
    report "c: repository" 1 "[workspace.package] repository is" "\"$repository\""
fi

# (d) GRIDCL_ variables outside Markdown.
var_hits=$(git grep -I -n -o -E 'GRIDCL_[A-Z0-9_]+' -- . ':(exclude,glob)**/*.md' \
    "${excluded_files[@]}" || true)
report "d: variables" "$(grep -c . <<<"$var_hits" || true)" "uses of GRIDCL_* variables" "$var_hits"

exit "$failed"
