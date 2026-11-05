#!/usr/bin/env bash
# usage: scripts/ci-gridcl-shard.sh <k>[,<k>...] [cargo test option]...
# Runs shards k (1-6) of `cargo test -p gridcl` in hosted CI, where the whole suite took
# 52-59 min in one job (M8 review). Shards 1-6 together run every test of gridcl once, doc
# tests included. gridcl's integration tests are one executable (tests/main.rs, a module per
# file), so its tests share a process: the CPU runtime's threads and the compiled kernels
# (nextest's process per test made the suite five times slower). Shards select modules by name:
# a libtest filter matches anywhere in a test's path, so `gmsh::` would also match a module
# `gmsh` nested in another; none is. io_threads' global_mesh_identical tests, one per rank
# count, are the slowest; shards 1-3 share them out by their times in CI.
#   1-3. io_threads: the global_mesh_identical tests of $gmi;
#   4.   the rest of io_threads, gmsh and parallel_threads;
#   5.   compile (trybuild) and the doc tests;
#   6.   the library and the other modules.
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
# The modules of shards 4 and 5.
named=(io_threads gmsh parallel_threads compile)
run() {
    echo "+ cargo test -p gridcl $*"
    cargo test -p gridcl "$@"
}
for k in $shards; do
    case $k in
    [1-3])
        exact=()
        for t in ${gmi[$k]}; do exact+=("io_threads::$t"); done
        run "$@" --test integration -- --exact "${exact[@]}"
        ;;
    4)
        run "$@" --test integration -- io_threads:: gmsh:: parallel_threads:: \
            --skip io_threads::global_mesh_identical
        ;;
    5)
        run "$@" --test integration -- compile::
        run "$@" --doc
        ;;
    6)
        skip=()
        for m in "${named[@]}"; do skip+=(--skip "$m::"); done
        run "$@" --lib
        run "$@" --test integration -- "${skip[@]}"
        ;;
    *)
        sed -n '2p' "$0" | sed 's/^# //' >&2
        exit 2
        ;;
    esac
done
