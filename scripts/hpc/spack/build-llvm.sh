#!/bin/bash -l
# Job script of the Myriad LLVM build (ADR-0004), rendered by setup.sh into @ROOT@: a CPU-only
# job (no GPU) that builds the Spack environment @ROOT@/env, lays the LLVM out for
# tracel-llvm-bundler and prints its llvm-config. About 20 min of compiling on 36 cores; the
# install into home over myriadfs is slow (2 h 34 min in all on 2026-10-18), hence h_rt=6:00:00.
# Progress: @ROOT@/progress-<job id>.txt (ninja's [n/N] every 5 min).
#$ -S /bin/bash
#$ -N gridcl-llvm
#$ -pe smp 36
#$ -l h_rt=6:00:00
#$ -l mem=4G
#$ -l tmpfs=40G
#$ -wd @ROOT@
#$ -o @ROOT@/build-$JOB_ID.log
#$ -j y
bundle=tracel-llvm-23.1.0-3 # the directory tracel-llvm-bundler 23.1.0-3 reads
if ! type module >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    . "${MODULESHOME:-/shared/ucl/apps/modules/5.3.1}/init/bash"
fi
module purge
module load gcc-libs/10.2.0 compilers/gnu/10.2.0
# shellcheck source=/dev/null
. @SPACK_ROOT@/share/spack/setup-env.sh
echo "host: $(hostname); $(grep -m1 'model name' /proc/cpuinfo); slots: $NSLOTS; TMPDIR: $TMPDIR"
echo "start: $(date -u +%FT%TZ)"
s=$(date +%s)
(
    while sleep 300; do
        f=$(find "$TMPDIR/$USER/spack-stage" -path '*spack-stage-llvm-*' -name spack-build-out.txt 2>/dev/null | head -n 1)
        [ -n "$f" ] && { date +%T; grep -o '\[[0-9]*/[0-9]*\]' "$f" | tail -n 1; } >"@ROOT@/progress-$JOB_ID.txt"
    done
) &
spack -e @ROOT@/env install --fail-fast -j "$NSLOTS"
status=$?
echo "end: $(date -u +%FT%TZ); install status $status; $(($(date +%s) - s)) s"
if [ "$status" = 0 ]; then
    prefix=$(spack -e @ROOT@/env location -i gridcl.llvm)
    x=@ROOT@/xdg/tracel/$bundle
    mkdir -p "$x"
    for d in bin include lib; do ln -sfn "$prefix/$d" "$x/$d"; done
    touch "$x/.tracel-llvm-installed"
    ln -sfn "$prefix/bin/ld.lld" @ROOT@/lld-bin/ld.lld
    echo "prefix: $prefix"
    for o in --version --targets-built --build-mode --shared-mode --system-libs --has-rtti --assertion-mode; do
        echo "$o: $("$x/bin/llvm-config" "$o" | tr '\n' ' ')"
    done
    echo "static libs: LLVM $(find "$prefix/lib" -name 'libLLVM*.a' | wc -l), lld $(find "$prefix/lib" -name 'liblld*.a' | wc -l)"
fi
echo "exit $status"
