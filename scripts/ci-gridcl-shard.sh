#!/usr/bin/env bash
# usage: scripts/ci-gridcl-shard.sh <k>[,<k>...] [cargo test option]...
# Runs shards k (1-6) of `cargo test -p gridcl` in hosted CI, where the whole suite took
# 52-59 min in one job (M8 review). Shards 1-6 together run every test of gridcl once, doc
# tests included. Each shard is one cargo test per test target, so the tests of a binary share
# their process: the CPU runtime's threads and the compiled kernels (nextest's process per
# test made the suite five times slower). io_threads' global_mesh_identical tests, one per
# rank count, are the slowest; shards 1-3 share them out by their times in CI.
#   1-3. io_threads: the global_mesh_identical tests of $gmi;
#   4.   the rest of io_threads, gmsh and parallel_threads;
#   5.   compile (trybuild) and the doc tests;
#   6.   the library and the other test targets.
# The options (for example --release --features gridcl-testing/vulkan) go to every cargo test.
set -eu
set -o pipefail

case ${1:-} in
[1-6] | [1-6],*) shards=${1//,/ } ;;
*)
    sed -n '2p' "$0" | sed 's/^# //' >&2
    exit 2
    ;;
esac
shift

gmi=(
    [1]="global_mesh_identical_subset::ranks_5 global_mesh_identical_subset::ranks_6"
    [2]="global_mesh_identical_subset::ranks_7 global_mesh_identical::ranks_1 global_mesh_identical::ranks_2 global_mesh_identical::ranks_3"
    [3]="global_mesh_identical::ranks_4 global_mesh_identical::ranks_8"
)
# The test targets of shards 4 and 5.
named=" io_threads gmsh parallel_threads compile "
run() {
    echo "+ cargo test -p gridcl $*"
    cargo test -p gridcl "$@"
}
for k in $shards; do
    case $k in
    [1-3])
        # shellcheck disable=SC2086 # one word per test name
        run "$@" --test io_threads -- --exact ${gmi[$k]}
        ;;
    4)
        skip=()
        for t in ${gmi[1]} ${gmi[2]} ${gmi[3]}; do skip+=(--skip "$t"); done
        run "$@" --test io_threads --test gmsh --test parallel_threads -- --exact "${skip[@]}"
        ;;
    5)
        run "$@" --test compile
        run "$@" --doc
        ;;
    6)
        others=()
        while read -r t; do
            case $named in *" $t "*) ;; *) others+=(--test "$t") ;; esac
        done < <(cargo metadata --no-deps --format-version 1 |
            jq -r '.packages[] | select(.name == "gridcl") | .targets[] | select(.kind == ["test"]) | .name')
        run "$@" --lib "${others[@]}"
        ;;
    *)
        sed -n '2p' "$0" | sed 's/^# //' >&2
        exit 2
        ;;
    esac
done
