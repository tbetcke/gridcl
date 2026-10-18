#!/usr/bin/env bash
# usage: bash scripts/hpc/spack/setup.sh   (on a Myriad login node, in the tree the runner synced)
#
# Prepares the Spack build of CubeCL's LLVM on Myriad (ADR-0004): RHEL 7.9 has glibc 2.17, too
# old for the LLVM bundle tracel-llvm-bundler downloads, so the same LLVM version is built here
# and laid out where the bundler looks for it. In $GRIDCL_LLVM_ROOT (default $HOME/gridcl-llvm)
# this script
#   - installs Spack $GRIDCL_SPACK_VERSION into $GRIDCL_SPACK (default $HOME/spack) if missing,
#   - copies the overlay repo and renders env/spack.yaml and build-llvm.sh,
#   - writes lld-bin/gcc, GCC 10 linking with LLD (RHEL 7's GNU ld 2.27 resolves archives in
#     one pass and misses LLVM symbols that only pliron-llvm uses),
#   - registers GCC 10.2 with Spack, concretizes, and fetches every source (compute nodes have
#     no internet).
# Then submit the build, a CPU-only job: cd "$GRIDCL_LLVM_ROOT" && qsub build-llvm.sh
# The job installs LLVM, creates xdg/tracel/tracel-llvm-<version>/ with the bundler's sentinel
# file, and links lld-bin/ld.lld. targets.env then exports, in LOGIN_MODULES and MODULES:
#   XDG_DATA_HOME=<root>/xdg PATH=<root>/lld-bin:$PATH
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
root=${GRIDCL_LLVM_ROOT:-$HOME/gridcl-llvm}
spack_root=${GRIDCL_SPACK:-$HOME/spack}
spack_version=${GRIDCL_SPACK_VERSION:-v1.2.2}
case $root in
    /*) ;;
    *) echo "setup.sh: GRIDCL_LLVM_ROOT must be an absolute path" >&2; exit 2 ;;
esac

if ! type module >/dev/null 2>&1; then
    # shellcheck source=/dev/null
    . "${MODULESHOME:?setup.sh: no module system}/init/bash"
fi
module purge
module load gcc-libs/10.2.0 compilers/gnu/10.2.0
gcc=$(command -v gcc)
cmake_prefix=$(module show cmake/4.1.2 2>&1 | awk '$1 == "prepend-path" && $2 == "PATH" { print $3; exit }')
cmake_prefix=${cmake_prefix%/bin}
if [ ! -x "$cmake_prefix/bin/cmake" ]; then
    echo "setup.sh: module cmake/4.1.2 not found" >&2
    exit 1
fi

if [ ! -d "$spack_root" ]; then
    git clone -q -c feature.manyFiles=true --depth=2 --branch="$spack_version" \
        https://github.com/spack/spack.git "$spack_root"
fi
# shellcheck source=/dev/null
. "$spack_root/share/spack/setup-env.sh"

mkdir -p "$root/env" "$root/repo" "$root/lld-bin"
cp -R "$here/repo/." "$root/repo/"
render() { sed -e "s|@ROOT@|$root|g" -e "s|@SPACK_ROOT@|$spack_root|g" -e "s|@CMAKE_PREFIX@|$cmake_prefix|g" "$1"; }
render "$here/spack.yaml.in" >"$root/env/spack.yaml"
render "$here/build-llvm.sh" >"$root/build-llvm.sh"
printf '#!/bin/sh\n# GCC 10 linking with LLD (scripts/hpc/spack/setup.sh, ADR-0004).\nexec %s -fuse-ld=lld "$@"\n' \
    "$gcc" >"$root/lld-bin/gcc"
chmod +x "$root/lld-bin/gcc"

spack compiler find "$(dirname "$gcc")" >/dev/null
spack -e "$root/env" concretize -f
spack -e "$root/env" fetch
echo "setup.sh: ready; submit the build with: cd $root && qsub build-llvm.sh"
