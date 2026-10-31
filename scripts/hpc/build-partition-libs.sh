#!/usr/bin/env bash
# usage: scripts/hpc/build-partition-libs.sh [prefix]
# Builds the partitioner libraries of M7 from pinned, checksummed sources with cmake and the
# MPI compiler wrapper mpicc on PATH: Scotch with PT-Scotch (threaded, without
# SCOTCH_PTHREAD_MPI, ADR-0029 item 13) into <prefix>/scotch-<version>, and ParMETIS with the
# METIS of its tarball into <prefix>/parmetis-<version>. Shared libraries (libmetis is static and
# also inside libparmetis); each records its own lib/ (RUNPATH on Linux, install name on macOS). A library whose stamp matches its version,
# checksum and options is skipped; downloads are kept in <prefix>/src.
# The prefix is the argument, else $GRIDCL_PARTITION_LIBS, else, in a job of
# scripts/hpc/remote-test.sh, partition-libs/ in the target's base directory (beside the trees).
# Prints the settings for the sys crates: SCOTCH_DIR, PARMETIS_DIR and METIS_DIR (the same
# prefix as PARMETIS_DIR), and the library paths. Runs on Linux and macOS.
set -eu
set -o pipefail

SCOTCH_VERSION=7.0.14
SCOTCH_SHA256=35e9d27edba28497876be3c338ff2ac5fa2835a3f15ce8196085f3cb669491b1
SCOTCH_URLS="https://gitlab.inria.fr/scotch/scotch/-/archive/v$SCOTCH_VERSION/scotch-v$SCOTCH_VERSION.tar.gz"
# The original tarball of glaros.dtc.umn.edu (now gone); Debian's orig tarball is the same file.
PARMETIS_VERSION=4.0.3
PARMETIS_SHA256=f2d9a231b7cf97f1fee6e8c9663113ebf6c240d407d3c118c55b3633d6be6e5f
PARMETIS_URLS="https://deb.debian.org/debian/pool/non-free/p/parmetis/parmetis_$PARMETIS_VERSION.orig.tar.gz
https://ftp.debian.org/debian/pool/non-free/p/parmetis/parmetis_$PARMETIS_VERSION.orig.tar.gz"

msg() { printf 'build-partition-libs: %s\n' "$*" >&2; }
die() {
    msg "$*"
    exit 1
}

usage() {
    sed -n '2,12p' "$0" | sed 's/^# \{0,1\}//'
}

sha256() {
    if command -v sha256sum >/dev/null 2>&1; then
        sha256sum "$1" | cut -d ' ' -f 1
    else
        shasum -a 256 "$1" | cut -d ' ' -f 1
    fi
}

ncpus() {
    getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4
}

# fetch <file> <sha256> <urls>: <file> with the given checksum, downloaded unless present.
fetch() {
    local file=$1 sum=$2 url
    if [ -f "$file" ] && [ "$(sha256 "$file")" = "$sum" ]; then
        return 0
    fi
    rm -f "$file"
    for url in $3; do
        msg "downloading $url"
        if command -v curl >/dev/null 2>&1; then
            curl -fsSL --retry 2 --connect-timeout 30 -o "$file.part" "$url" || continue
        else
            wget -q -O "$file.part" "$url" || continue
        fi
        if [ "$(sha256 "$file.part")" = "$sum" ]; then
            mv "$file.part" "$file"
            return 0
        fi
        msg "checksum mismatch for $url"
    done
    rm -f "$file.part"
    die "cannot download $(basename "$file") with sha256 $sum"
}

# fresh <stamp file> <stamp>: true when the install recorded this stamp.
fresh() {
    [ -f "$1" ] && [ "$(cat "$1")" = "$2" ]
}

# The options that make a shared library find its neighbours in <lib> once installed.
rpath_options() {
    case $(uname -s) in
        Darwin) printf '%s\n' "-DCMAKE_INSTALL_NAME_DIR=$1" ;;
        *) printf '%s\n' "-DCMAKE_INSTALL_RPATH=$1" ;;
    esac
}

build_scotch() {
    local dest=$PREFIX/scotch-$SCOTCH_VERSION src build stamp tarball
    local -a options
    # INTSIZE stays empty: SCOTCH_Num is then C's int, as in Homebrew's and Debian's builds.
    # INTSIZE=32 types it int32_t, and scotch.h does not include stdint.h for it.
    options=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER="$(command -v mpicc)"
        -DCMAKE_INSTALL_PREFIX="$dest"
        -DCMAKE_INSTALL_LIBDIR=lib
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_PTSCOTCH=ON
        -DBUILD_FORTRAN=OFF
        -DBUILD_LIBESMUMPS=OFF
        -DBUILD_LIBSCOTCHMETIS=OFF
        -DINSTALL_METIS_HEADERS=OFF
        -DTHREADS=ON
        -DMPI_THREAD_MULTIPLE=OFF
        -DUSE_ZLIB=OFF
        -DUSE_LZMA=OFF
        -DUSE_BZ2=OFF
        -DENABLE_TESTS=OFF
        "$(rpath_options "$dest/lib")"
    )
    stamp="scotch $SCOTCH_VERSION $SCOTCH_SHA256 ${options[*]}"
    if fresh "$dest/.gridcl-stamp" "$stamp"; then
        msg "scotch $SCOTCH_VERSION: up to date in $dest"
        return 0
    fi
    tarball=$PREFIX/src/scotch-v$SCOTCH_VERSION.tar.gz
    fetch "$tarball" "$SCOTCH_SHA256" "$SCOTCH_URLS"
    build=$PREFIX/build/scotch-$SCOTCH_VERSION
    rm -rf "$build" "$dest"
    mkdir -p "$build"
    tar -xzf "$tarball" -C "$build"
    src=$build/scotch-v$SCOTCH_VERSION
    msg "scotch $SCOTCH_VERSION: configuring"
    cmake -S "$src" -B "$build/build" "${options[@]}" >"$build/configure.log" 2>&1 ||
        die "scotch: cmake failed; see $build/configure.log"
    # ADR-0029 item 13: a PT-Scotch with SCOTCH_PTHREAD_MPI needs MPI_THREAD_MULTIPLE.
    if grep -rq SCOTCH_PTHREAD_MPI "$build/build" --include=flags.make; then
        die "scotch: the build defines SCOTCH_PTHREAD_MPI"
    fi
    msg "scotch $SCOTCH_VERSION: building"
    # parser_ly.h is an output of parser_yy_c's rule and an input of parser_ll_c's, so a
    # parallel make runs that rule in both targets at once and they race (seen on Kathleen
    # with bison 3.7.4, which takes Scotch's fallback parser files). Build it first.
    { cmake --build "$build/build" --target parser_yy_c &&
        cmake --build "$build/build" --parallel "$(ncpus)"; } >"$build/build.log" 2>&1 ||
        die "scotch: build failed; see $build/build.log"
    cmake --install "$build/build" >"$build/install.log" 2>&1 ||
        die "scotch: install failed; see $build/install.log"
    [ -f "$dest/include/ptscotch.h" ] || die "scotch: no include/ptscotch.h in $dest"
    printf '%s\n' "$stamp" >"$dest/.gridcl-stamp"
    rm -rf "$build"
}

build_parmetis() {
    local dest=$PREFIX/parmetis-$PARMETIS_VERSION src build stamp tarball
    local -a options
    # ParMETIS builds a static libmetis from its tarball's metis/ and links it into
    # libparmetis (position-independent code); METIS_INSTALL installs libmetis.a as well.
    # CMAKE_POLICY_VERSION_MINIMUM: the CMakeLists ask for 2.8, which cmake 4 rejects
    # otherwise (cmake 3 ignores the variable).
    options=(
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER="$(command -v mpicc)"
        -DCMAKE_INSTALL_PREFIX="$dest"
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
        -DCMAKE_POSITION_INDEPENDENT_CODE=ON
        -DSHARED=1
        -DMETIS_INSTALL=ON
        "$(rpath_options "$dest/lib")"
    )
    stamp="parmetis $PARMETIS_VERSION $PARMETIS_SHA256 ${options[*]}"
    if fresh "$dest/.gridcl-stamp" "$stamp"; then
        msg "parmetis $PARMETIS_VERSION: up to date in $dest"
        return 0
    fi
    tarball=$PREFIX/src/parmetis-$PARMETIS_VERSION.tar.gz
    fetch "$tarball" "$PARMETIS_SHA256" "$PARMETIS_URLS"
    build=$PREFIX/build/parmetis-$PARMETIS_VERSION
    rm -rf "$build" "$dest"
    mkdir -p "$build"
    tar -xzf "$tarball" -C "$build"
    src=$build/parmetis-$PARMETIS_VERSION
    msg "parmetis $PARMETIS_VERSION: configuring"
    # Configured from the build directory: the CMakeLists create a symlink `metis` in cmake's
    # working directory.
    mkdir -p "$build/build"
    (cd "$build/build" && cmake -S "$src" -B . "${options[@]}" \
        -DGKLIB_PATH="$src/metis/GKlib" -DMETIS_PATH="$src/metis") >"$build/configure.log" 2>&1 ||
        die "parmetis: cmake failed; see $build/configure.log"
    msg "parmetis $PARMETIS_VERSION: building"
    cmake --build "$build/build" --parallel "$(ncpus)" >"$build/build.log" 2>&1 ||
        die "parmetis: build failed; see $build/build.log"
    cmake --install "$build/build" >"$build/install.log" 2>&1 ||
        die "parmetis: install failed; see $build/install.log"
    # ParMETIS installs libmetis but not its header; parmetis.h includes metis.h.
    mkdir -p "$dest/include"
    cp "$src/metis/include/metis.h" "$dest/include/metis.h"
    [ -f "$dest/include/parmetis.h" ] || die "parmetis: no include/parmetis.h in $dest"
    printf '%s\n' "$stamp" >"$dest/.gridcl-stamp"
    rm -rf "$build"
}

main() {
    case ${1:-} in
        -h | --help)
            usage
            return 0
            ;;
    esac
    [ $# -le 1 ] || {
        usage >&2
        return 2
    }
    PREFIX=${1:-${GRIDCL_PARTITION_LIBS:-}}
    if [ -z "$PREFIX" ] && [ -n "${GRIDCL_HPC_TREE:-}" ]; then
        PREFIX=$(dirname "$GRIDCL_HPC_TREE")/partition-libs
    fi
    [ -n "$PREFIX" ] || {
        msg "no prefix: give one, or set GRIDCL_PARTITION_LIBS"
        usage >&2
        return 2
    }
    for tool in cmake mpicc tar; do
        command -v "$tool" >/dev/null 2>&1 || die "$tool not found on PATH"
    done
    command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 ||
        die "neither curl nor wget found on PATH"
    mkdir -p "$PREFIX/src" "$PREFIX/build"
    PREFIX=$(cd "$PREFIX" && pwd)
    msg "prefix $PREFIX; $(cmake --version | head -n 1); mpicc: $(command -v mpicc)"
    build_scotch
    build_parmetis
    rmdir "$PREFIX/build" 2>/dev/null || true
    local scotch=$PREFIX/scotch-$SCOTCH_VERSION parmetis=$PREFIX/parmetis-$PARMETIS_VERSION
    cat <<EOF
# Settings for the sys crates (ptscotch-sys, parmetis-sys) and, through CPATH and
# LIBRARY_PATH, scotch-sys (the scotch feature); the library paths for the test executables.
export SCOTCH_DIR=$scotch
export PARMETIS_DIR=$parmetis
export METIS_DIR=$parmetis
export CPATH=$scotch/include\${CPATH:+:\$CPATH}
export LIBRARY_PATH=$scotch/lib:$parmetis/lib\${LIBRARY_PATH:+:\$LIBRARY_PATH}
export LD_LIBRARY_PATH=$scotch/lib:$parmetis/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
EOF
}

main "$@"
