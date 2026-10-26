#!/usr/bin/env bash
# Check and print the local tool versions of the M3 Max (docs/design.md, E0 acceptance):
# vtk >= 9.2, meshio and the pinned gmsh wheel (4.15.2, the M4 fixture corpus) importable in
# ${GRIDCL_PYTHON:-.venv/bin/python}, gmsh,
# claude >= 2.1.286, rsync >= 3 and mpirun. Prints one line per tool; exits 1 if any
# check fails.
#
# usage: scripts/check-local-tools.sh
set -u

repo=$(cd "$(dirname "$0")/.." && pwd)
python=${GRIDCL_PYTHON:-$repo/.venv/bin/python}
failed=0

# version_ge <have> <want>: dotted numeric comparison, ignoring non-numeric suffixes.
version_ge() {
    local have=$1 want=$2 h w i
    for i in 1 2 3 4; do
        h=$(echo "$have" | cut -d. -f"$i" | sed 's/[^0-9].*//')
        w=$(echo "$want" | cut -d. -f"$i" | sed 's/[^0-9].*//')
        h=${h:-0}
        w=${w:-0}
        if [ "$h" -gt "$w" ]; then return 0; fi
        if [ "$h" -lt "$w" ]; then return 1; fi
    done
    return 0
}

# report <tool> <version or empty> [minimum]
report() {
    local tool=$1 version=$2 minimum=${3:-}
    if [ -z "$version" ]; then
        printf '%-8s FAIL: not found\n' "$tool"
        failed=1
    elif [ -n "$minimum" ] && ! version_ge "$version" "$minimum"; then
        printf '%-8s %s FAIL: need >= %s\n' "$tool" "$version" "$minimum"
        failed=1
    elif [ -n "$minimum" ]; then
        printf '%-8s %s (>= %s)\n' "$tool" "$version" "$minimum"
    else
        printf '%-8s %s\n' "$tool" "$version"
    fi
}

# first_version <text>: the first dotted version number in the text.
first_version() {
    echo "$1" | grep -Eo '[0-9]+(\.[0-9]+)+[^ ]*' | head -n 1
}

if [ -x "$python" ]; then
    report python "$("$python" -c 'import platform; print(platform.python_version())' 2>/dev/null)"
    report vtk "$("$python" -c 'import vtk; print(vtk.vtkVersion.GetVTKVersion())' 2>/dev/null)" 9.2
    report meshio "$("$python" -c 'import meshio; print(meshio.__version__)' 2>/dev/null)"
    # The Gmsh wheel that writes the M4 fixture corpus: exactly this version (docs/plans/M4.md, T1).
    gmsh_pin=4.15.2
    gmsh_py=$("$python" -c 'import gmsh; print(gmsh.__version__)' 2>/dev/null)
    if [ "$gmsh_py" = "$gmsh_pin" ]; then
        printf '%-8s %s (pinned)\n' gmsh-py "$gmsh_py"
    else
        printf '%-8s %s FAIL: need %s (.venv/bin/pip install gmsh==%s)\n' gmsh-py "${gmsh_py:-none}" \
            "$gmsh_pin" "$gmsh_pin"
        failed=1
    fi
else
    printf '%-8s FAIL: no interpreter at %s (python3 -m venv .venv && .venv/bin/pip install vtk meshio gmsh==4.15.2)\n' \
        python "$python"
    failed=1
fi

report gmsh "$(gmsh --version 2>&1 | head -n 1 | grep -Eo '^[0-9][^ ]*')"
report claude "$(first_version "$(claude --version 2>/dev/null)")" 2.1.286
report rsync "$(first_version "$(rsync --version 2>/dev/null | head -n 1)")" 3
mpi=$(mpirun --version 2>/dev/null | head -n 1)
report mpirun "$(first_version "$mpi")"
[ -z "$mpi" ] || printf '%-8s %s\n' "" "$mpi"

exit "$failed"
