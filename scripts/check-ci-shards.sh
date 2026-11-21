#!/usr/bin/env bash
# usage: scripts/check-ci-shards.sh [cargo test option]...
# Checks that the shards of scripts/ci-tetraloma-shard.sh still run every test of tetraloma once
# (E3), without running the tests:
#   1. each `shard:` matrix of .github/workflows/ci.yml (cpu-tetraloma, vulkan) holds shards 1-12,
#      each once (entries like "11,12" hold two; `other` is not a shard);
#   2. every test of `cargo test -p tetraloma` (lib, integration and doc tests) is selected by
#      exactly one of shards 1-12, h5-h8 (p = 5-8, which hosted CI does not run) or the exclusion
#      below, by the shard script's own selection (TETRALOMA_SHARD_LIST=1 lists instead of running).
#   3. tetraloma's test targets are the library and `integration` (with the doc tests), the ones
#      the shard script and step 2 know; another target would be in no listing.
# It lists the tests of one build, by default debug with default features. The vulkan jobs build
# release with tetraloma-testing/vulkan, which lists the same tests: no test depends on the
# profile or on that feature (only on hdf5, metis and scotch, which run outside the shards; E3).
# An ignored test counts as selected (p = 5-8 in debug builds): the check is about selection.
# The options (for example --release) go to every cargo test. Exit status 1 on any finding.
set -eu
set -o pipefail
export LC_ALL=C
cd "$(dirname "$0")/.."

shards=(1 2 3 4 5 6 7 8 9 10 11 12)
uncovered=(h5 h6 h7 h8)
# In no shard: p = 9 and 10, release only; Metal, CUDA and the CPU runtime run them (the shard
# script's header).
excluded=(
    "integration high_order::open_ended_ranges_launch::p9"
    "integration high_order::open_ended_ranges_launch::p10"
)
fail=0
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

# 1. The CI matrices.
printf '%s\n' "${shards[@]}" | sort >"$tmp/want"
grep -E '^ +shard: \[' .github/workflows/ci.yml >"$tmp/matrices" || true
if [ "$(wc -l <"$tmp/matrices")" -ne 2 ]; then
    echo "FAIL  expected 2 shard matrices in ci.yml (cpu-tetraloma, vulkan):"
    cat "$tmp/matrices"
    fail=1
fi
while IFS= read -r line; do
    echo "$line" | sed -e 's/.*\[//' -e 's/\].*//' | tr -d "\"' " | tr ',' '\n' |
        { grep -vx other || true; } | sort >"$tmp/got"
    if cmp -s "$tmp/want" "$tmp/got"; then
        echo "ok    ci.yml:${line}"
    else
        echo "FAIL  ci.yml:${line}: not shards ${shards[*]}, each once"
        fail=1
    fi
done <"$tmp/matrices"

# 3. The test targets (before step 2, whose listings assume them).
cargo metadata --no-deps --format-version 1 | python3 -I -c '
import json, sys
for p in json.load(sys.stdin)["packages"]:
    if p["name"] == "tetraloma":
        for t in p["targets"]:
            print(" ".join(t["kind"]), t["name"])
' | sort >"$tmp/targets"
printf '%s\n' "lib tetraloma" "test integration" >"$tmp/known"
if cmp -s "$tmp/known" "$tmp/targets"; then
    echo "ok    targets: $(tr '\n' ',' <"$tmp/targets" | sed 's/,$//')"
else
    echo "FAIL  tetraloma's targets are not the library and \`integration\` (update the shard script and this check):"
    sed 's/^/        /' "$tmp/targets"
    fail=1
fi

# 2. The tests. Each listing follows a `+ cargo test` line, which names its target.
tag() {
    awk '
        /^\+ cargo test / {
            t = "integration"
            if ($0 ~ / --lib( |$)/) t = "lib"
            else if ($0 ~ / --doc( |$)/) t = "doc"
            next
        }
        /: test$/ { sub(/: test$/, ""); print t " " $0 }
    '
}
for target in --lib "--test integration" --doc; do
    echo "+ cargo test $target"
    # shellcheck disable=SC2086 # $target is one or two words
    cargo test -p tetraloma "$@" $target -- --list --format terse
done | tag | sort >"$tmp/all"
: >"$tmp/selected"
for k in "${shards[@]}" "${uncovered[@]}"; do
    if ! TETRALOMA_SHARD_LIST=1 scripts/ci-tetraloma-shard.sh "$k" "$@" >"$tmp/shard"; then
        echo "FAIL  shard $k: scripts/ci-tetraloma-shard.sh failed"
        fail=1
        continue
    fi
    tag <"$tmp/shard" | sed "s/^/$k /" >>"$tmp/selected"
    echo "ok    shard $k: $(tag <"$tmp/shard" | wc -l | tr -d ' ') tests"
done
for e in "${excluded[@]}"; do echo "excluded $e"; done >>"$tmp/selected"
cut -d ' ' -f 2- "$tmp/selected" | sort >"$tmp/names"

comm -23 "$tmp/all" "$tmp/names" >"$tmp/missing"
uniq -d "$tmp/names" >"$tmp/dups"
comm -13 "$tmp/all" <(sort -u "$tmp/names") >"$tmp/unknown"
while IFS= read -r t; do echo "FAIL  in no shard: $t"; done <"$tmp/missing"
while IFS= read -r t; do
    where=$(t="$t" awk '{ k = $1; sub(/^[^ ]+ /, ""); if ($0 == ENVIRON["t"]) printf " %s", k }' "$tmp/selected")
    echo "FAIL  in more than one shard ($where): $t"
done <"$tmp/dups"
# A selected name the full listing lacks: an exclusion that no longer exists.
while IFS= read -r t; do echo "FAIL  selected but not in the full listing: $t"; done <"$tmp/unknown"
if [ -s "$tmp/missing" ] || [ -s "$tmp/dups" ] || [ -s "$tmp/unknown" ]; then
    fail=1
fi
if [ "$(wc -l <"$tmp/all")" -eq 0 ]; then
    echo "FAIL  the full listing is empty"
    fail=1
fi

if [ "$fail" -ne 0 ]; then
    exit 1
fi
echo "PASS  $(wc -l <"$tmp/all" | tr -d ' ') tests of tetraloma, each in exactly one of shards" \
    "${shards[*]}, ${uncovered[*]} or the exclusion; both CI matrices hold shards ${shards[*]}"
