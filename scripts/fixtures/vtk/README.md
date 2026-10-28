# VTK-written reference cells (M5)

`generate.py` writes the committed reference cells in `gridcl-io/tests/fixtures/vtk/` with the
pinned VTK Python wheel, `vtk==9.7.1` from PyPI (docs/plans/M5.md, T1; ADR-0008, ADR-0024 item
12). It refuses to run with any other VTK version.

```sh
python3 -m venv .venv && .venv/bin/pip install vtk==9.7.1 meshio==5.3.5
.venv/bin/python scripts/fixtures/vtk/generate.py          # rewrite the fixtures
.venv/bin/python scripts/fixtures/vtk/generate.py --check  # regenerate into a temporary
                                                           # directory, compare bytes
```

## Files

- `<cell>_p<p>.vtu`: one cell of each VTK Lagrange type, at every degree p = 1–6 (36 files):
  `curve` (68), `triangle` (69), `quadrilateral` (70), `tetrahedron` (71), `hexahedron` (72)
  and `wedge` (73). The wedges at p = 3, 5 and 6 are the ones ADR-0008 asks for: their
  triangular faces are lexicographic, and differ from the recursive order from p = 5 on.
- The points of each cell are VTK's own parametric coordinates of its nodes, in VTK's node
  order: `GetParametricCoords()` of the `vtkLagrange*` cell with that many points (quadrilaterals,
  hexahedra and wedges after `SetUniformOrderFromNumPoints`). The connectivity is 0, 1, …, n−1.
  So the k-th point of a file is VTK's k-th node of the reference cell, and gridcl-io's
  `vtk_to_internal` must map it onto `lagrange_points(cell, p)` (the T4 test
  `vtk_written_cells_match_vtk_to_internal`).
- Written in ASCII by `vtkXMLUnstructuredGridWriter`, otherwise with its default settings. VTK
  chooses the file version: 2.3 for hexahedra (VTK 9 renumbered higher-order hexahedra, and VTK
  renumbers files below version 2.1 on reading) and 0.1 for the other types. The ASCII writer
  prints shortest round-trip doubles, so the coordinates are exact.
- Before a file is kept, `generate.py` reads it back with `vtkXMLUnstructuredGridReader` and
  checks the cell type, the identity connectivity and the points bitwise.

## Manifest

`manifest.json` is `{"vtk_version": "9.7.1", "cells": [{"file", "vtk_type", "degree",
"num_points"}]}`, one entry per file. `gridcl/tests/vtk.rs` (`vtk_reads_every_fixture`) reads it
and asserts exactly the 36 (type, degree) pairs.
