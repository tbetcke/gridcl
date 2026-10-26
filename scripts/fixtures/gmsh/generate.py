#!/usr/bin/env python3
"""Generates the Gmsh fixture corpus of M4 (docs/plans/M4.md, T1; ADR-0022).

Writes gridcl-io/tests/fixtures/gmsh/*.msh and manifest.json with the pinned Gmsh Python wheel
(PyPI gmsh==4.15.2, installed in .venv). The manifest records Gmsh's own view of every file, as
Gmsh reads it back: the node count, the distinct nodes of the D_TOP elements, the element counts
per type, the expected (D_TOP, D_GEO), and per physical group its size and the sorted
corner-node tags of every element.

Usage:
    .venv/bin/python scripts/fixtures/gmsh/generate.py          # (re)write the corpus
    .venv/bin/python scripts/fixtures/gmsh/generate.py --check  # regenerate into a temporary
                                                                # directory and compare bytes
"""

import argparse
import filecmp
import json
import re
import sys
import tempfile
from pathlib import Path

import gmsh

PINNED = "4.15.2"
REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "gridcl-io" / "tests" / "fixtures" / "gmsh"

# (dimension, corners, nodes) of the Gmsh types the corpus uses (A.7). A table, since
# gmsh.model.mesh.getElementProperties fails for some high-order types (40-node prisms).
TYPES = {
    15: (0, 1, 1),
    1: (1, 2, 2), 8: (1, 2, 3), 26: (1, 2, 4), 27: (1, 2, 5),
    2: (2, 3, 3), 9: (2, 3, 6), 21: (2, 3, 10), 23: (2, 3, 15),
    3: (2, 4, 4), 10: (2, 4, 9), 36: (2, 4, 16), 37: (2, 4, 25), 16: (2, 4, 8),
    4: (3, 4, 4), 11: (3, 4, 10), 29: (3, 4, 20), 30: (3, 4, 35),
    5: (3, 8, 8), 12: (3, 8, 27), 92: (3, 8, 64), 93: (3, 8, 125), 17: (3, 8, 20),
    6: (3, 6, 6), 13: (3, 6, 18), 90: (3, 6, 40), 91: (3, 6, 75), 18: (3, 6, 15),
    7: (3, 5, 5), 14: (3, 5, 14), 118: (3, 5, 30), 19: (3, 5, 13),
}


def start(name):
    """A fresh Gmsh session with deterministic settings."""
    gmsh.initialize(["-noenv"], readConfigFiles=False)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.NumThreads", 1)
    gmsh.option.setNumber("Mesh.SaveAll", 1)
    gmsh.model.add(name)


def finish():
    gmsh.finalize()


def physical(dim, entities, tag, name=None):
    gmsh.model.addPhysicalGroup(dim, entities, tag)
    if name is not None:
        gmsh.model.setPhysicalName(dim, tag, name)


def mesh_size(h):
    """Unstructured meshes of element size h."""
    gmsh.option.setNumber("Mesh.MeshSizeMin", h)
    gmsh.option.setNumber("Mesh.MeshSizeMax", h)


def entity_at(dim, point, tol=1e-9):
    """The entity of dimension `dim` whose bounding box centre is `point`."""
    found = []
    for _, tag in gmsh.model.getEntities(dim):
        box = gmsh.model.getBoundingBox(dim, tag)
        centre = [(box[i] + box[i + 3]) / 2 for i in range(3)]
        if all(abs(c - p) <= tol for c, p in zip(centre, point)):
            found.append(tag)
    assert len(found) == 1, f"dimension {dim} entity at {point}: {found}"
    return found[0]


# --------------------------------------------------------------------------------------------
# Models. Each builds the geometry, the physical groups and the linear mesh.


def line(points):
    """A polyline through `points`: two curves of two elements each."""
    geo = gmsh.model.geo
    p = [geo.addPoint(*x) for x in points]
    curves = [geo.addLine(p[0], p[1]), geo.addLine(p[1], p[2])]
    geo.synchronize()
    for c in curves:
        gmsh.model.mesh.setTransfiniteCurve(c, 3)
    physical(1, curves, 1, "domain")
    physical(1, [curves[1]], 2)  # curve 2 in two groups, unnamed
    physical(0, [p[0], p[2]], 3, "ends")
    gmsh.model.mesh.generate(1)


def rectangle(z=0.0, quads=False):
    """[0, 2] x [0, 1] at height z: unstructured triangles, or 4 x 2 quadrilaterals."""
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, z) for x, y in [(0, 0), (2, 0), (2, 1), (0, 1)]]
    c = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = geo.addPlaneSurface([geo.addCurveLoop(c)])
    geo.synchronize()
    if quads:
        for i, n in zip(c, [5, 3, 5, 3]):
            gmsh.model.mesh.setTransfiniteCurve(i, n)
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    else:
        mesh_size(0.6)
    physical(2, [s], 1, "domain")
    physical(1, [c[3]], 11, "left")
    physical(1, [c[0], c[2]], 12, "walls")
    physical(1, [c[0]], 13)  # the bottom is in two groups
    physical(0, p, 21, "corners")
    gmsh.model.mesh.generate(2)


def extruded(quads):
    """[0, 2] x [0, 1] x [0, 1]: a 2D mesh extruded in two layers (hexahedra or prisms)."""
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0) for x, y in [(0, 0), (2, 0), (2, 1), (0, 1)]]
    c = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = geo.addPlaneSurface([geo.addCurveLoop(c)])
    out = geo.extrude([(2, s)], 0, 0, 1, numElements=[2], recombine=True)
    geo.synchronize()
    # After the last synchronize: it resets mesh constraints set on the model.
    if quads:
        for i, n in zip(c, [4, 3, 4, 3]):
            gmsh.model.mesh.setTransfiniteCurve(i, n)
        gmsh.model.mesh.setTransfiniteSurface(s)
        gmsh.model.mesh.setRecombine(2, s)
    else:
        mesh_size(0.8)
    top = out[0][1]
    vol = out[1][1]
    physical(3, [vol], 1, "domain")
    physical(2, [s], 11, "bottom")
    physical(2, [top], 12, "top")
    physical(1, [c[0]], 13)
    physical(0, [p[0]], 21, "origin")
    gmsh.model.mesh.generate(3)


def tetrahedra():
    """[0, 1]^3 in unstructured tetrahedra."""
    occ = gmsh.model.occ
    vol = occ.addBox(0, 0, 0, 1, 1, 1)
    occ.synchronize()
    mesh_size(0.5)
    left = entity_at(2, (0, 0.5, 0.5))
    right = entity_at(2, (1, 0.5, 0.5))
    physical(3, [vol], 1, "domain")
    physical(2, [left], 11, "left")
    physical(2, [left, right], 12, "sides")
    physical(1, [entity_at(1, (0.5, 0, 0))], 13)
    physical(0, [entity_at(0, (0, 0, 0))], 21, "origin")
    gmsh.model.mesh.generate(3)


def hybrid():
    """[0, 2] x [0, 1]^2: one hexahedron in [0, 1]^3, tetrahedra beside it, pyramids between."""
    occ = gmsh.model.occ
    a = occ.addBox(0, 0, 0, 1, 1, 1)
    b = occ.addBox(1, 0, 0, 1, 1, 1)
    occ.fragment([(3, a)], [(3, b)])
    occ.synchronize()
    left = entity_at(3, (0.5, 0.5, 0.5))
    right = entity_at(3, (1.5, 0.5, 0.5))
    for dim, tag in gmsh.model.getBoundary([(3, left)], combined=False, oriented=False,
                                           recursive=True):
        if dim == 1:
            gmsh.model.mesh.setTransfiniteCurve(tag, 2)
    for _, tag in gmsh.model.getBoundary([(3, left)], combined=False, oriented=False):
        gmsh.model.mesh.setTransfiniteSurface(tag)
        gmsh.model.mesh.setRecombine(2, tag)
    gmsh.model.mesh.setTransfiniteVolume(left)
    gmsh.model.mesh.setRecombine(3, left)
    mesh_size(1.0)
    physical(3, [left], 1, "hex")
    physical(3, [right], 2, "tet")
    physical(3, [left, right], 3, "domain")
    physical(2, [entity_at(2, (1, 0.5, 0.5))], 11, "interface")
    gmsh.model.mesh.generate(3)


def box_surface():
    """The surface of [0, 1]^3 in triangles: a closed surface in 3D."""
    occ = gmsh.model.occ
    occ.addBox(0, 0, 0, 1, 1, 1)
    occ.synchronize()
    mesh_size(0.6)
    faces = [t for _, t in gmsh.model.getEntities(2)]
    physical(2, faces, 1, "surface")
    physical(2, [entity_at(2, (0, 0.5, 0.5))], 2, "left")
    physical(1, [entity_at(1, (0.5, 0, 0))], 11)
    gmsh.model.mesh.generate(2)


def two_regions():
    """[0, 2] x [0, 1] as two surfaces: triangles in [0, 1]^2, quadrilaterals beside them.

    Physical groups: one entity in two groups (both surfaces in "all"), a name in two
    dimensions ("boundary" for curves and points), unnamed groups.
    """
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0) for x, y in [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]]
    bottom_l = geo.addLine(p[0], p[1])
    bottom_r = geo.addLine(p[1], p[2])
    right = geo.addLine(p[2], p[3])
    top_r = geo.addLine(p[3], p[4])
    top_l = geo.addLine(p[4], p[5])
    left = geo.addLine(p[5], p[0])
    mid = geo.addLine(p[1], p[4])
    s_l = geo.addPlaneSurface([geo.addCurveLoop([bottom_l, mid, top_l, left])])
    s_r = geo.addPlaneSurface([geo.addCurveLoop([bottom_r, right, top_r, -mid])])
    geo.synchronize()
    for c in [bottom_r, top_r]:
        gmsh.model.mesh.setTransfiniteCurve(c, 4)
    for c in [right, mid]:
        gmsh.model.mesh.setTransfiniteCurve(c, 4)
    gmsh.model.mesh.setTransfiniteSurface(s_r)
    gmsh.model.mesh.setRecombine(2, s_r)
    mesh_size(0.4)
    physical(2, [s_l], 1, "triangles")
    physical(2, [s_r], 2, "quadrilaterals")
    physical(2, [s_l, s_r], 3, "all")
    physical(1, [bottom_l, bottom_r, right, top_r, top_l, left], 10, "boundary")
    physical(1, [mid], 11, "interface")
    physical(1, [bottom_l, bottom_r], 12)
    physical(0, [p[0], p[2], p[3], p[5]], 20, "boundary")
    physical(0, [p[1]], 21)
    gmsh.model.mesh.generate(2)


def periodic():
    """[0, 1]^2 in triangles, the right side periodic with the left."""
    geo = gmsh.model.geo
    p = [geo.addPoint(x, y, 0) for x, y in [(0, 0), (1, 0), (1, 1), (0, 1)]]
    c = [geo.addLine(p[i], p[(i + 1) % 4]) for i in range(4)]
    s = geo.addPlaneSurface([geo.addCurveLoop(c)])
    geo.synchronize()
    mesh_size(0.3)
    translate = [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    gmsh.model.mesh.setPeriodic(1, [c[1]], [c[3]], translate)
    physical(2, [s], 1, "domain")
    physical(1, [c[1], c[3]], 11, "periodic")
    gmsh.model.mesh.generate(2)


# --------------------------------------------------------------------------------------------
# Fixtures: (file stem, model, (D_TOP, D_GEO), order, serendipity, binary copy, partitions).


def fixtures():
    out = []

    def add(stem, model, dims, order=1, serendipity=False, binary=False, partitions=0,
            twin=None):
        out.append(dict(stem=stem, model=model, dims=dims, order=order,
                        serendipity=serendipity, binary=binary, partitions=partitions,
                        twin=twin))

    straight_1d = lambda: line([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
    for p in range(1, 5):
        add(f"interval_p{p}", straight_1d, (1, 1), p, binary=p <= 2)
        add(f"triangle_p{p}", rectangle, (2, 2), p, binary=p <= 2)
        add(f"quadrilateral_p{p}", lambda: rectangle(quads=True), (2, 2), p, binary=p <= 2)
        add(f"tetrahedron_p{p}", tetrahedra, (3, 3), p, binary=p <= 2)
        add(f"hexahedron_p{p}", lambda: extruded(True), (3, 3), p, binary=p <= 2)
        add(f"prism_p{p}", lambda: extruded(False), (3, 3), p, binary=p <= 2)
        if p <= 3:
            add(f"hybrid_p{p}", hybrid, (3, 3), p, binary=p <= 2)
    add("quadrilateral_s2", lambda: rectangle(quads=True), (2, 2), 2, True, True,
        twin="quadrilateral_p2")
    add("hexahedron_s2", lambda: extruded(True), (3, 3), 2, True, True, twin="hexahedron_p2")
    add("prism_s2", lambda: extruded(False), (3, 3), 2, True, True, twin="prism_p2")
    add("hybrid_s2", hybrid, (3, 3), 2, True, True, twin="hybrid_p2")
    add("curve_2d_p2", lambda: line([(0, 0, 0), (1, 0.5, 0), (2, 1, 0)]), (1, 2), 2)
    add("curve_3d_p1", lambda: line([(0, 0, 0), (1, 0.5, 0.25), (2, 1, 0.5)]), (1, 3), 1)
    add("square_z1_p1", lambda: rectangle(z=1.0), (2, 3), 1)
    add("box_surface_p2", box_surface, (2, 3), 2, binary=True)
    add("two_regions_p1", two_regions, (2, 2), 1, binary=True)
    add("partitioned_p1", two_regions, (2, 2), 1, partitions=2)
    add("periodic_p1", periodic, (2, 2), 1)
    return out


# --------------------------------------------------------------------------------------------
# Writing and the manifest.


def write(spec, path, binary):
    start(spec["stem"])
    spec["model"]()
    if spec["serendipity"]:
        gmsh.option.setNumber("Mesh.SecondOrderIncomplete", 1)
    if spec["order"] > 1:
        gmsh.model.mesh.setOrder(spec["order"])
    if spec["partitions"]:
        gmsh.model.mesh.partition(spec["partitions"])
    gmsh.option.setNumber("Mesh.Binary", 1 if binary else 0)
    gmsh.write(str(path))
    finish()


def describe(path, dims):
    """Gmsh's view of the file at `path`, read back by Gmsh."""
    start("read")
    gmsh.open(str(path))
    mesh = gmsh.model.mesh
    node_tags, _, _ = mesh.getNodes()
    counts = {}
    mesh_nodes = set()
    tdim = 0
    types, _, nodes = mesh.getElements()
    for ty, tags in zip(types, nodes):
        dim, _, num = TYPES[int(ty)]
        counts[str(int(ty))] = counts.get(str(int(ty)), 0) + len(tags) // num
        tdim = max(tdim, dim)
    for ty, tags in zip(types, nodes):
        if TYPES[int(ty)][0] == tdim:
            mesh_nodes.update(int(t) for t in tags)
    assert tdim == dims[0], f"{path.name}: D_TOP {tdim}, expected {dims[0]}"
    groups = []
    for dim, tag in gmsh.model.getPhysicalGroups():
        elements = []
        for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
            etypes, _, enodes = mesh.getElements(dim, entity)
            for ty, tags in zip(etypes, enodes):
                _, k, num = TYPES[int(ty)]
                for i in range(0, len(tags), num):
                    elements.append(sorted(int(t) for t in tags[i:i + k]))
        elements.sort()
        name = gmsh.model.getPhysicalName(dim, tag)
        groups.append(dict(dim=dim, tag=tag, name=name or None, size=len(elements),
                           elements=elements))
    entry = dict(
        file=path.name,
        binary=gmsh.option.getNumber("Mesh.Binary") == 1 or is_binary(path),
        num_nodes=len(node_tags),
        num_mesh_nodes=len(mesh_nodes),
        element_counts=dict(sorted(counts.items(), key=lambda kv: int(kv[0]))),
        tdim=dims[0],
        gdim=dims[1],
        physical_groups=groups,
        twin=None,
        partitioned=gmsh.model.getNumberOfPartitions() > 0,
        periodic=False,
    )
    finish()
    entry["binary"] = is_binary(path)
    entry["periodic"] = has_section(path, b"$Periodic")
    return entry


def is_binary(path):
    with open(path, "rb") as f:
        f.readline()
        return f.readline().split()[1] == b"1"


def has_section(path, name):
    return name + b"\n" in path.read_bytes()


def duplicate_nodes(src, dst):
    """A copy of the ASCII file `src` in which the last node of the last node block takes the
    tag of the first node of that block, and elements refer to the first node instead: one node
    tag is then repeated, and nothing else is wrong. Returns the repeated tag."""
    lines = src.read_text().split("\n")
    start_ = lines.index("$Nodes")
    i = start_ + 2
    last = None
    while lines[i] != "$EndNodes":
        n = int(lines[i].split()[3])
        if n >= 2:
            last = i
        i += 1 + 2 * n
    n = int(lines[last].split()[3])
    first_tag, last_tag = lines[last + 1], lines[last + n]
    lines[last + n] = first_tag
    e = lines.index("$Elements") + 2
    while lines[e] != "$EndElements":
        header = lines[e].split()
        count = int(header[3])
        for j in range(e + 1, e + 1 + count):
            fields = lines[j].split(" ")
            lines[j] = " ".join([fields[0]] + [first_tag if f == last_tag else f
                                               for f in fields[1:]])
        e += 1 + count
    dst.write_text("\n".join(lines))
    return int(first_tag)


def duplicate_elements(src, dst):
    """A copy of the ASCII file `src` in which the last element of the last element block takes
    the tag of the first element of that block. Returns the repeated tag."""
    lines = src.read_text().split("\n")
    e = lines.index("$Elements") + 2
    last = None
    while lines[e] != "$EndElements":
        count = int(lines[e].split()[3])
        if count >= 2:
            last = e
        e += 1 + count
    count = int(lines[last].split()[3])
    first = lines[last + 1].split(" ")[0]
    fields = lines[last + count].split(" ")
    lines[last + count] = " ".join([first] + fields[1:])
    dst.write_text("\n".join(lines))
    return int(first)


def generate(out):
    out.mkdir(parents=True, exist_ok=True)
    entries = []
    by_stem = {}
    for spec in fixtures():
        encodings = [False, True] if spec["binary"] else [False]
        for binary in encodings:
            name = spec["stem"] + ("-binary" if binary else "") + ".msh"
            path = out / name
            write(spec, path, binary)
            entry = describe(path, spec["dims"])
            if spec["twin"]:
                twin = spec["twin"] + ("-binary" if binary else "") + ".msh"
                entry["twin"] = twin
                entry["num_mesh_nodes"] = by_stem[twin]["num_mesh_nodes"]
            by_stem[name] = entry
            entries.append(entry)
            print(f"{name}: {entry['num_nodes']} nodes, {entry['element_counts']}")
    source = out / "triangle_p1.msh"
    for stem, make, of in [("duplicate_node_tags", duplicate_nodes, "nodes"),
                           ("duplicate_element_tags", duplicate_elements, "elements")]:
        path = out / f"{stem}.msh"
        tag = make(source, path)
        entry = dict(by_stem["triangle_p1.msh"], file=path.name, physical_groups=[],
                     expect_error="DuplicateOriginId", duplicate_id=tag, duplicate_of=of)
        entries.append(entry)
        print(f"{path.name}: tag {tag} repeated")
    manifest = dict(gmsh_version=gmsh.__version__, files=entries)
    text = json.dumps(manifest, indent=1)
    # One line per innermost list of numbers (an element's corner tags).
    text = re.sub(r"\[\s*(-?\d+(?:,\s*-?\d+)*)\s*\]",
                  lambda m: "[" + ",".join(m.group(1).split()).replace(",,", ",") + "]", text)
    (out / "manifest.json").write_text(text + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true",
                        help="regenerate into a temporary directory and compare bytes")
    args = parser.parse_args()
    if gmsh.__version__ != PINNED:
        sys.exit(f"gmsh {gmsh.__version__} found; the corpus is pinned to {PINNED} "
                 f"(.venv/bin/pip install gmsh=={PINNED})")
    if not args.check:
        generate(OUT)
        return
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        generate(tmp)
        names = sorted(p.name for p in tmp.iterdir())
        committed = sorted(p.name for p in OUT.iterdir())
        if names != committed:
            sys.exit(f"file lists differ:\n  generated {names}\n  committed {committed}")
        _, mismatch, errors = filecmp.cmpfiles(tmp, OUT, names, shallow=False)
        if mismatch or errors:
            sys.exit(f"files differ from the committed corpus: {mismatch + errors}")
        print(f"check: {len(names)} files byte-identical")


if __name__ == "__main__":
    main()
