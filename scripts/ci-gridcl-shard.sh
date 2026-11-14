#!/usr/bin/env bash
# usage: scripts/ci-gridcl-shard.sh <k>[,<k>...] [cargo test option]...   (k: 1-10, h5-h8)
# Runs shards k of `cargo test -p gridcl` in hosted CI, where the whole suite took 52-59 min in
# one job (M8 review). Shards 1-10 together run every test of gridcl once, doc tests included;
# in release builds the M9h tests at p = 5-8 are in shards h5-h8 (one degree each), and are
# ignored in debug builds. gridcl's integration tests are one executable (tests/main.rs, a module
# per file), so its tests share a process: the CPU runtime's threads and the compiled kernels
# (nextest's process per test made the suite five times slower). Shards select modules by name:
# a libtest filter matches anywhere in a test's path, so `gmsh::` would also match a module
# `gmsh` nested in another; none is. io_threads' global_mesh_identical tests, one per rank
# count, are the slowest; shards 1-3 share them out by their times in CI. M9's thread tests
# took shard 6 from 9 to 37 min, so they have shards of their own. M9h's tests are one test per
# degree (`<test>::p<p>`, `degree_tests!`); M9h took shards 4 and 6 from 11 to 26-28 min, so
# shards 2, 3, 6, 9 and 10 share out p = 1-4 by their times (docs/plans/M9h.md, T12).
#   1-3. io_threads: the global_mesh_identical tests of $gmi;
#   4.   the rest of io_threads, gmsh and parallel_threads;
#   5.   compile (trybuild) and the doc tests;
#   6.   the library and the other modules;
#   7.   submesh_threads;
#   8.   orient_threads;
#   9.   (M9h tests only);
#   10.  submesh;
#   and in shards 2, 3, 6, 9 and 10 the M9h tests of $ho at p = 1-4; h5-h8 those at p = 5-8.
# The options (for example --release --features gridcl-testing/vulkan) go to every cargo test.
set -eu
set -o pipefail

usage() {
    sed -n '2p' "$0" | sed 's/^# //' >&2
    exit 2
}
shards=${1:-}
shards=${shards//,/ }
[ -n "$shards" ] || usage
for k in $shards; do
    case $k in
    [1-9] | 10 | h[5-8]) ;;
    *) usage ;;
    esac
done
shift

gmi=(
    [1]="global_mesh_identical_subset::ranks_5 global_mesh_identical_subset::ranks_6"
    [2]="global_mesh_identical_subset::ranks_7 global_mesh_identical::ranks_1 global_mesh_identical::ranks_2 global_mesh_identical::ranks_3"
    [3]="global_mesh_identical::ranks_4 global_mesh_identical::ranks_8"
)
# M9h's tests, one per degree (`<test>::p<p>`), and those at p = 1-4 of each shard.
sizes=high_order_invariance::vector_size_invariance
tiles=high_order_invariance::tile_invariance
twice=high_order_invariance::two_runs_are_bitwise_identical
factors=high_order::factors_match_host_reference
pull_back=high_order::pull_back_recovers_reference_points
parallel=parallel_threads::high_order_parallel_invariance
ho_tests=("$sizes" "$tiles" "$twice" "$factors" "$pull_back" "$parallel")
ho=(
    [2]="$sizes::p1 $sizes::p4"
    [3]="$sizes::p2 $sizes::p3 $tiles::p1 $tiles::p2 $twice::p1 $twice::p2"
    [6]="$factors::p1 $factors::p2 $factors::p3 $factors::p4 $pull_back::p1 $pull_back::p2"
    [9]="$parallel::p1 $parallel::p2 $parallel::p3 $parallel::p4 $tiles::p3 $tiles::p4"
    [10]="$twice::p3 $twice::p4 $pull_back::p3 $pull_back::p4"
)
# The modules of shards 4, 5, 7, 8 and 10.
named=(io_threads gmsh parallel_threads compile submesh_threads submesh orient_threads)
run() {
    echo "+ cargo test -p gridcl $*"
    cargo test -p gridcl "$@"
}
for k in $shards; do
    # This shard's M9h tests (bash 3.2: no empty arrays under set -u).
    mine=(${ho[${k#h}]:-})
    case $k in
    [1-3])
        exact=()
        for t in ${gmi[$k]}; do exact+=("io_threads::$t"); done
        run "$@" --test integration -- --exact "${exact[@]}" ${mine[@]+"${mine[@]}"}
        ;;
    4)
        run "$@" --test integration -- io_threads:: gmsh:: parallel_threads:: \
            --skip io_threads::global_mesh_identical --skip "$parallel"
        ;;
    5)
        run "$@" --test integration -- compile::
        run "$@" --doc
        ;;
    6)
        skip=()
        for m in "${named[@]}"; do skip+=(--skip "$m::"); done
        # Every M9h test but this shard's.
        for t in "${ho_tests[@]}"; do
            for p in 1 2 3 4 5 6 7 8; do
                [[ " ${mine[*]} " == *" $t::p$p "* ]] || skip+=(--skip "$t::p$p")
            done
        done
        run "$@" --lib
        run "$@" --test integration -- "${skip[@]}"
        ;;
    7)
        run "$@" --test integration -- submesh_threads::
        ;;
    8)
        run "$@" --test integration -- orient_threads::
        ;;
    9)
        run "$@" --test integration -- --exact ${mine[@]+"${mine[@]}"}
        ;;
    10)
        run "$@" --test integration -- submesh:: ${mine[@]+"${mine[@]}"}
        ;;
    h[5-8])
        names=()
        for t in "${ho_tests[@]}"; do names+=("$t::p${k#h}"); done
        run "$@" --test integration -- --exact "${names[@]}"
        ;;
    esac
done
