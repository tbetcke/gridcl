#!/usr/bin/env bash
# usage: scripts/ci-gridcl-shard.sh <k>/<n> [cargo test option]...
# Runs shard k of n (n = 2 or 4) of `cargo test -p gridcl` in hosted CI, where the whole suite
# took 52-59 min in one job (M8 review). Together the shards of one n run every test of gridcl
# once, doc tests included. Each shard is one cargo test, so the tests of a binary share their
# process: the CPU runtime's threads and the compiled kernels (nextest's process per test made
# the suite five times slower). io_threads' global_mesh_identical tests, one per rank count,
# are the slowest.
#   n = 4: 1. io_threads global_mesh_identical_subset (5-7 ranks); 2. io_threads
#          global_mesh_identical (1-4 and 8 ranks); 3. the rest of io_threads, gmsh and
#          parallel_threads; 4. the library, the other test targets and the doc tests.
#   n = 2: 1. shards 1 and 2 of n = 4; 2. shards 3 and 4.
# The options (for example --release --features gridcl-testing/vulkan) go to every cargo test.
set -eu
set -o pipefail

case ${1:-} in
1/2 | 2/2 | [1-4]/4) shard=$1 ;;
*)
    sed -n '2p' "$0" | sed 's/^# //' >&2
    exit 2
    ;;
esac
shift

# The test targets of shard 3 of 4.
heavy=" io_threads gmsh parallel_threads "
run() {
    echo "+ cargo test -p gridcl $*"
    cargo test -p gridcl "$@"
}
case $shard in
1/4) run "$@" --test io_threads global_mesh_identical_subset ;;
2/4) run "$@" --test io_threads global_mesh_identical:: ;;
1/2) run "$@" --test io_threads global_mesh_identical ;;
3/4) run "$@" --test io_threads --test gmsh --test parallel_threads -- --skip global_mesh_identical ;;
4/4)
    others=()
    while read -r t; do
        case $heavy in *" $t "*) ;; *) others+=(--test "$t") ;; esac
    done < <(cargo metadata --no-deps --format-version 1 |
        jq -r '.packages[] | select(.name == "gridcl") | .targets[] | select(.kind == ["test"]) | .name')
    run "$@" --lib "${others[@]}"
    run "$@" --doc
    ;;
2/2)
    run "$@" --lib --tests -- --skip global_mesh_identical
    run "$@" --doc
    ;;
esac
