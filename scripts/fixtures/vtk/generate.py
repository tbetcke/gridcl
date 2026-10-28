#!/usr/bin/env python3
"""Generates the VTK-written reference cells of M5 (docs/plans/M5.md, T1; ADR-0008, ADR-0024).

Writes gridcl-io/tests/fixtures/vtk/<cell>_p<p>.vtu and manifest.json with the pinned VTK Python
wheel (PyPI vtk==9.7.1, installed in .venv): one cell of every VTK Lagrange type 68-73 at every
degree p = 1-6. The points of each cell are VTK's own parametric coordinates of its nodes
(GetParametricCoords of the vtkLagrange* cell with that point count), in VTK's node order, and
the connectivity is 0, 1, ..., n-1. The files are written in ASCII by
vtkXMLUnstructuredGridWriter with its default settings otherwise (VTK chooses the file version:
2.3 for hexahedra, whose numbering changed in VTK 9, 0.1 for the others).

Every file is read back with vtkXMLUnstructuredGridReader before it is kept: one cell of the
type, the identity connectivity and the parametric coordinates bitwise.

Usage:
    .venv/bin/python scripts/fixtures/vtk/generate.py          # (re)write the fixtures
    .venv/bin/python scripts/fixtures/vtk/generate.py --check  # regenerate into a temporary
                                                               # directory and compare bytes
"""

import argparse
import filecmp
import json
import sys
import tempfile
from pathlib import Path

try:
    from vtkmodules.vtkCommonCore import vtkIdList, vtkPoints, vtkVersion
    from vtkmodules.vtkCommonDataModel import (
        vtkLagrangeCurve,
        vtkLagrangeHexahedron,
        vtkLagrangeQuadrilateral,
        vtkLagrangeTetra,
        vtkLagrangeTriangle,
        vtkLagrangeWedge,
        vtkUnstructuredGrid,
    )
    from vtkmodules.vtkIOXML import vtkXMLUnstructuredGridReader, vtkXMLUnstructuredGridWriter
    from vtkmodules.util.numpy_support import vtk_to_numpy
except ImportError as e:
    sys.exit(f"generate.py: {e}; install the pin: .venv/bin/pip install vtk==9.7.1")

PINNED = "9.7.1"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "gridcl-io" / "tests" / "fixtures" / "vtk"
DEGREES = range(1, 7)

# VTK type: (file name stem, cell class, number of nodes at degree p).
CELLS = {
    68: ("curve", vtkLagrangeCurve, lambda p: p + 1),
    69: ("triangle", vtkLagrangeTriangle, lambda p: (p + 1) * (p + 2) // 2),
    70: ("quadrilateral", vtkLagrangeQuadrilateral, lambda p: (p + 1) ** 2),
    71: ("tetrahedron", vtkLagrangeTetra, lambda p: (p + 1) * (p + 2) * (p + 3) // 6),
    72: ("hexahedron", vtkLagrangeHexahedron, lambda p: (p + 1) ** 3),
    73: ("wedge", vtkLagrangeWedge, lambda p: (p + 1) ** 2 * (p + 2) // 2),
}


def parametric_coords(vtk_type, n):
    """VTK's parametric coordinates of the n nodes of a Lagrange cell, [node][3]."""
    _, cls, _ = CELLS[vtk_type]
    cell = cls()
    cell.GetPointIds().SetNumberOfIds(n)
    cell.GetPoints().SetNumberOfPoints(n)
    if hasattr(cell, "SetUniformOrderFromNumPoints"):  # quadrilateral, hexahedron, wedge
        cell.SetUniformOrderFromNumPoints(n)
    cell.Initialize()
    pc = cell.GetParametricCoords()
    assert pc is not None and len(pc) == 3 * n, f"type {vtk_type}, {n} nodes: {pc!r}"
    return [tuple(pc[3 * k:3 * k + 3]) for k in range(n)]


def write_cell(path, vtk_type, coords):
    grid = vtkUnstructuredGrid()
    points = vtkPoints()
    points.SetDataTypeToDouble()
    ids = vtkIdList()
    for k, x in enumerate(coords):
        points.InsertNextPoint(*x)
        ids.InsertNextId(k)
    grid.SetPoints(points)
    grid.InsertNextCell(vtk_type, ids)
    writer = vtkXMLUnstructuredGridWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(grid)
    writer.SetDataModeToAscii()
    if writer.Write() != 1:
        sys.exit(f"generate.py: {path}: vtkXMLUnstructuredGridWriter failed")


def check_read_back(path, vtk_type, coords):
    reader = vtkXMLUnstructuredGridReader()
    reader.SetFileName(str(path))
    reader.Update()
    grid = reader.GetOutput()
    n = len(coords)
    assert grid.GetNumberOfCells() == 1, path
    assert grid.GetCellType(0) == vtk_type, path
    assert grid.GetNumberOfPoints() == n, path
    connectivity = vtk_to_numpy(grid.GetCells().GetConnectivityArray()).tolist()
    assert connectivity == list(range(n)), f"{path}: VTK renumbered the nodes: {connectivity}"
    read = [tuple(grid.GetPoint(k)) for k in range(n)]
    assert read == coords, f"{path}: points do not round-trip"


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"vtk_version": vtkVersion.GetVTKVersion(), "cells": []}
    for vtk_type, (stem, _, count) in CELLS.items():
        for p in DEGREES:
            n = count(p)
            coords = parametric_coords(vtk_type, n)
            name = f"{stem}_p{p}.vtu"
            write_cell(out / name, vtk_type, coords)
            check_read_back(out / name, vtk_type, coords)
            manifest["cells"].append(
                {"file": name, "vtk_type": vtk_type, "degree": p, "num_points": n})
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temporary directory and compare bytes")
    args = parser.parse_args()
    version = vtkVersion.GetVTKVersion()
    if version != PINNED:
        sys.exit(f"generate.py: VTK {version}, need {PINNED}: .venv/bin/pip install vtk=={PINNED}")
    if not args.check:
        generate(OUT)
        print(f"wrote {len(CELLS) * len(DEGREES)} cells and manifest.json to {OUT}")
        return
    with tempfile.TemporaryDirectory() as tmp:
        fresh = Path(tmp)
        generate(fresh)
        want = sorted(p.name for p in fresh.iterdir())
        have = sorted(p.name for p in OUT.iterdir()) if OUT.is_dir() else []
        bad = sorted(set(want) ^ set(have))
        bad += [n for n in want if n in have and not filecmp.cmp(fresh / n, OUT / n, shallow=False)]
        if bad:
            print("differ from a fresh run (rerun generate.py and commit):", file=sys.stderr)
            for n in bad:
                print(f"  {n}", file=sys.stderr)
            sys.exit(1)
        print(f"{len(want)} files identical to a fresh run")


if __name__ == "__main__":
    main()
