#!/usr/bin/env python3
"""The Python oracles of the M5 VTK tests (docs/plans/M5.md, T1; ADR-0024, item 11; ADR-0025).

Runs VTK's own XML readers and meshio on VTU/PVTU files and prints JSON on stdout. The consumer
is gridcl/tests/common/python.rs. The pinned versions are asserted: any other vtk or meshio
version is refused with a non-zero exit and the install command on stderr.

Usage:
    vtk_oracle.py read FILE...          VTK's reader: counts, cell types, bounds, arrays,
                                        errors and warnings
    vtk_oracle.py integrate FILE...     vtkIntegrateAttributes: length, area, volume
    vtk_oracle.py evaluate REQUEST.json vtkCell::EvaluateLocation at given parametric points
    vtk_oracle.py meshio FILE...        the raw arrays as meshio reads them
    vtk_oracle.py versions              the interpreter's vtk, meshio and Python versions

Exit status: 0 when every file was processed (VTK errors in a file are reported in its "errors"
list), 1 when meshio fails on a file, 2 on a usage error or a missing or wrong package version.

read: one object per file, in argument order:
    {"file", "num_points", "num_cells", "types": [VTK type of each cell],
     "bounds": [xmin, xmax, ymin, ymax, zmin, zmax], "point_data": [{"name", "components",
     "dtype"}], "cell_data": [...], "errors": [str], "warnings": [str]}
  .pvtu files are read with vtkXMLPUnstructuredGridReader, others with
  vtkXMLUnstructuredGridReader. Errors and warnings come from an observer on the reader and
  from VTK's global output window (messages of the XML parser, the pipeline and piece readers).
  "dtype" is the numpy name of the array's type (float32, float64, int32, uint32, int64, uint8).

integrate: {"file", "length", "area", "volume", "errors", "warnings"}: the sums
  vtkIntegrateAttributes puts in its output cell data ("Length", "Area", "Volume"; 0.0 when
  absent).

evaluate: REQUEST is a JSON list [{"file": path, "points": {"<VTK type>": [r, s, t, ...]}}]. For
  every cell of the file, in order, EvaluateLocation at each parametric point of its type gives
  {"file", "cells": [[x0, y0, z0, x1, y1, z1, ...] per cell], "errors", "warnings"}.

meshio: {"file", "version", "points": Arr, "connectivity": [int], "offsets": [int],
         "types": [int], "point_data": {name: Arr}, "cell_data": {name: Arr}}
  meshio 5.3.5 accepts only VTU versions 0.1 and 1.0, so it reads a temporary copy whose
  <VTKFile> version attribute alone reads 1.0; "version" is the file's own value (ADR-0025).
  Connectivity, offsets and types are those of the file: the blocks in order, with meshio's own
  node permutation undone (meshio_to_vtk_order; it reorders only linear wedges). Cell data is
  concatenated over the blocks. Arr = {"dtype": numpy name, "shape": [...], "hex": the
  little-endian bytes}, taken from meshio's arrays without type conversion; points have shape
  [n, 3], a 1-component array [n], k components [n, k].
"""

import contextlib
import json
import os
import re
import sys
import tempfile

PINS = {"vtk": "9.7.1", "meshio": "5.3.5"}
INSTALL = ("python3 -m venv .venv && .venv/bin/pip install vtk==9.7.1 meshio==5.3.5 "
           "(or set GRIDCL_PYTHON to an interpreter that has them)")


def die(message, status=2):
    print(f"vtk_oracle.py: {message}", file=sys.stderr)
    sys.exit(status)


def check_versions():
    """The pinned packages, or exit 2 naming the install command."""
    try:
        from vtkmodules.vtkCommonCore import vtkVersion
        import meshio
    except ImportError as e:
        die(f"{e}; install the pinned oracles: {INSTALL}")
    found = {"vtk": vtkVersion.GetVTKVersion(), "meshio": meshio.__version__}
    wrong = [f"{k} {found[k]} (need {v})" for k, v in PINS.items() if found[k] != v]
    if wrong:
        die(f"wrong versions: {', '.join(wrong)}; install the pinned oracles: {INSTALL}")
    return found


# --------------------------------------------------------------------------------------------
# VTK


class Messages:
    """Errors and warnings of VTK objects: observers on chosen objects, and the global output
    window for every other object (XML parser, executive, the readers of PVTU pieces)."""

    def __init__(self):
        from vtkmodules.vtkCommonCore import vtkLogger, vtkOutputWindow, vtkStringOutputWindow

        vtkLogger.SetStderrVerbosity(vtkLogger.VERBOSITY_OFF)
        self.window = vtkStringOutputWindow()
        vtkOutputWindow.SetInstance(self.window)
        self.seen = 0
        self.errors = []
        self.warnings = []

    def observe(self, obj):
        from vtkmodules.util.misc import calldata_type
        from vtkmodules.util.vtkConstants import VTK_STRING

        @calldata_type(VTK_STRING)
        def on_error(_obj, _event, message):
            self.errors.append(str(message).strip())

        @calldata_type(VTK_STRING)
        def on_warning(_obj, _event, message):
            self.warnings.append(str(message).strip())

        obj.AddObserver("ErrorEvent", on_error)
        obj.AddObserver("WarningEvent", on_warning)
        return obj

    def take(self):
        """The messages since the last call, as (errors, warnings)."""
        # The window only accumulates (Python cannot clear it): take what is new.
        full = self.window.GetOutput() or ""
        text, self.seen = full[self.seen:], len(full)
        for message in re.split(r"\n(?=ERROR:|Warning:|Generic Warning:|Debug:)", text):
            message = message.strip()
            if not message:
                continue
            if message.startswith("ERROR"):
                self.errors.append(message)
            elif message.startswith("Debug"):
                continue
            else:
                self.warnings.append(message)
        errors, warnings = self.errors, self.warnings
        self.errors, self.warnings = [], []
        return errors, warnings


def read_grid(path, messages):
    """VTK's reader of `path` (observed), after Update()."""
    from vtkmodules.vtkIOXML import vtkXMLPUnstructuredGridReader, vtkXMLUnstructuredGridReader

    if path.lower().endswith(".pvtu"):
        reader = vtkXMLPUnstructuredGridReader()
    else:
        reader = vtkXMLUnstructuredGridReader()
    messages.observe(reader)
    reader.SetFileName(path)
    reader.Update()
    return reader


def numpy_dtype(array):
    from vtkmodules.vtkCommonCore import vtkDataArray

    if isinstance(array, vtkDataArray):
        import numpy as np
        from vtkmodules.util.numpy_support import get_numpy_array_type

        return np.dtype(get_numpy_array_type(array.GetDataType())).name
    return array.GetDataTypeAsString()


def arrays(data):
    out = []
    for i in range(data.GetNumberOfArrays()):
        a = data.GetAbstractArray(i)
        out.append({"name": a.GetName(), "components": a.GetNumberOfComponents(),
                    "dtype": numpy_dtype(a)})
    return out


def cmd_read(files):
    messages = Messages()
    out = []
    for path in files:
        grid = read_grid(path, messages).GetOutput()
        result = {"file": path, "num_points": 0, "num_cells": 0, "types": [],
                  "bounds": [0.0] * 6, "point_data": [], "cell_data": []}
        if grid is not None:
            result.update(
                num_points=grid.GetNumberOfPoints(),
                num_cells=grid.GetNumberOfCells(),
                types=[grid.GetCellType(i) for i in range(grid.GetNumberOfCells())],
                bounds=list(grid.GetBounds()),
                point_data=arrays(grid.GetPointData()),
                cell_data=arrays(grid.GetCellData()),
            )
        result["errors"], result["warnings"] = messages.take()
        out.append(result)
    return out


def cmd_integrate(files):
    from vtkmodules.vtkFiltersParallel import vtkIntegrateAttributes

    messages = Messages()
    out = []
    for path in files:
        reader = read_grid(path, messages)
        integrator = messages.observe(vtkIntegrateAttributes())
        integrator.SetInputConnection(reader.GetOutputPort())
        integrator.Update()
        cell_data = integrator.GetOutput().GetCellData()
        result = {"file": path}
        for key, name in [("length", "Length"), ("area", "Area"), ("volume", "Volume")]:
            a = cell_data.GetArray(name)
            result[key] = float(a.GetTuple1(0)) if a is not None and a.GetNumberOfTuples() else 0.0
        result["errors"], result["warnings"] = messages.take()
        out.append(result)
    return out


def cmd_evaluate(args):
    from vtkmodules.vtkCommonCore import reference

    if len(args) != 1:
        die("evaluate takes one REQUEST.json")
    try:
        with open(args[0]) as f:
            request = json.load(f)
    except (OSError, ValueError) as e:
        die(f"{args[0]}: {e}")
    if not isinstance(request, list):
        die(f"{args[0]}: expected a JSON list")
    messages = Messages()
    out = []
    for item in request:
        path = str(item["file"])
        points = {int(k): [float(x) for x in v] for k, v in item["points"].items()}
        for t, p in points.items():
            if len(p) % 3:
                die(f"{args[0]}: {path}: type {t}: {len(p)} coordinates, not a multiple of 3")
        grid = read_grid(path, messages).GetOutput()
        cells = []
        missing = set()
        for i in range(grid.GetNumberOfCells() if grid is not None else 0):
            cell = grid.GetCell(i)
            t = cell.GetCellType()
            pcoords = points.get(t)
            if pcoords is None:
                missing.add(t)
                cells.append([])
                continue
            weights = [0.0] * cell.GetNumberOfPoints()
            xs = []
            for q in range(0, len(pcoords), 3):
                x = [0.0, 0.0, 0.0]
                cell.EvaluateLocation(reference(0), pcoords[q:q + 3], x, weights)
                xs.extend(x)
            cells.append(xs)
        errors, warnings = messages.take()
        errors.extend(f"no parametric points for VTK type {t}" for t in sorted(missing))
        out.append({"file": path, "cells": cells, "errors": errors, "warnings": warnings})
    return out


# --------------------------------------------------------------------------------------------
# meshio

VTKFILE_TAG = re.compile(rb"<VTKFile\b[^>]*>")
VERSION_ATTR = re.compile(rb"""(\sversion\s*=\s*)(["'])([^"']*)\2""")


def relabelled(raw):
    """(the file's version or None, the bytes with the <VTKFile> version attribute set to 1.0)."""
    tag = VTKFILE_TAG.search(raw)
    if tag is None:
        return None, raw
    m = VERSION_ATTR.search(raw, tag.start(), tag.end())
    if m is None:
        return None, raw
    version = m.group(3).decode("ascii", "replace")
    new = m.group(1) + m.group(2) + b"1.0" + m.group(2)
    return version, raw[:m.start()] + new + raw[m.end():]


def arr(a):
    """{"dtype", "shape", "hex"} of a numpy array, without type conversion."""
    import numpy as np

    a = np.ascontiguousarray(a)
    if a.ndim == 2 and a.shape[1] == 1:
        a = a.reshape(a.shape[0])
    little = a.astype(a.dtype.newbyteorder("<"), copy=False)
    return {"dtype": a.dtype.name, "shape": list(a.shape), "hex": little.tobytes().hex()}


def cmd_meshio(files):
    import numpy as np
    import meshio
    from meshio._vtk_common import meshio_to_vtk_order, meshio_to_vtk_type

    out = []
    with tempfile.TemporaryDirectory(prefix="vtk_oracle_") as tmp:
        for path in files:
            try:
                with open(path, "rb") as f:
                    raw = f.read()
            except OSError as e:
                die(f"{path}: {e}", 1)
            version, copy_bytes = relabelled(raw)
            copy = os.path.join(tmp, os.path.basename(path))
            with open(copy, "wb") as f:
                f.write(copy_bytes)
            try:
                # meshio prints some errors to stdout; keep stdout for the JSON.
                with contextlib.redirect_stdout(sys.stderr):
                    mesh = meshio.vtu.read(copy)
            except (Exception, SystemExit) as e:  # noqa: BLE001 - any meshio failure
                die(f"{path}: meshio {meshio.__version__}: {type(e).__name__}: {e}", 1)
            connectivity, offsets, types = [], [], []
            end = 0
            for block in mesh.cells:
                data = np.asarray(block.data)
                order = meshio_to_vtk_order(block.type)
                if order is not None:
                    data = data[:, order]
                vtk_type = meshio_to_vtk_type[block.type]
                for cell in data:
                    connectivity.extend(int(n) for n in cell)
                    end += len(cell)
                    offsets.append(end)
                    types.append(vtk_type)
            out.append({
                "file": path,
                "version": version,
                "points": arr(mesh.points),
                "connectivity": connectivity,
                "offsets": offsets,
                "types": types,
                "point_data": {k: arr(v) for k, v in mesh.point_data.items()},
                "cell_data": {k: arr(np.concatenate([np.asarray(b) for b in v]))
                              for k, v in mesh.cell_data.items()},
            })
    return out


# --------------------------------------------------------------------------------------------


def main(argv):
    if len(argv) < 2 or argv[1] in ("-h", "--help"):
        print(__doc__, file=sys.stderr)
        sys.exit(0 if len(argv) >= 2 else 2)
    command, args = argv[1], argv[2:]
    commands = {"read": cmd_read, "integrate": cmd_integrate, "evaluate": cmd_evaluate,
                "meshio": cmd_meshio}
    if command != "versions" and command not in commands:
        die(f"unknown command {command!r} (read, integrate, evaluate, meshio, versions)")
    found = check_versions()
    if command == "versions":
        import platform

        result = dict(found, python=platform.python_version())
    else:
        if command != "evaluate" and not args:
            die(f"{command} needs at least one FILE")
        result = commands[command](args)
    json.dump(result, sys.stdout)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main(sys.argv)
