#!/usr/bin/env python3
"""The checks of scripts/check-user-docs.sh (ADR-0067, rule 7), on a built site. Standard library
only.

Usage:
    scripts/doc_audit.py check --site DIR --work DIR [--checks abcdefgh] [--crate CRATE]
                               [--api DIR] [--verbose]
    scripts/doc_audit.py api-plan --features FEATURES [--crates CRATE,...]

`check` runs the checks named by `--checks` (default all) and prints a count per check and crate:

    (a) the site builds without warnings: the build logs in WORK/logs (mdbook.log for mdBook's
        `WARN` and `ERROR` lines, rustdoc-*.log for rustdoc's `warning` and `error` lines);
    (b) links and anchors resolve inside the site, including links into /api/ and absolute links to
        the site's base URL (mapped to the local build); links from the book into /api/ target item
        pages, not anchors on them;
    (c) no developer citation (a docs/ path, an ADR number, a plan question, a milestone or task
        id, a design listing or appendix) in a rendered doc block of the audited crates;
    (d) every item of rule 6.2 links to a book section with an absolute URL under the site's base
        URL;
    (e) every function and method of rule 6.3 has an `# Example` heading and a doctest that runs
        (listed in WORK/doctests.txt and not in WORK/doctests-ignored.txt), or an entry in
        docs/user/doc-audit-allow.toml;
    (f) the book has the outline of rule 4 (the files, titles and fixed headings of OUTLINE below),
        and no heading has fewer than MIN_WORDS words of prose under it;
    (g) every fence has a language tag, no code block is indented, no `rust` block is `no_run`,
        every `ignore` block is only an `{{#include}}` of a program that scripts/run-examples.sh
        runs, no other block includes a file, and the doctest harness of tetraloma-examples
        includes every chapter;
    (h) while docs/plans/D1.md is not done, the sources of the nine library crates differ from D1's
        base commit only in comments (scripts/comments_only.py).

`--crate` limits (c)-(e) to one crate (its items, wherever rustdoc renders them); `--api` audits
another rustdoc output directory (the `parmetis` build) instead of SITE/api. Exit status 1 if a
check fails, 2 on a usage error or if the audit finds no items (a changed rustdoc markup).

`api-plan` prints the order in which the library crates are documented (dependencies first, so
that rustdoc links to the docs of the crates already built) and each crate's own features when the
facade is built with FEATURES (`tetraloma/mpi,...`): one line `CRATE FEATURES`, FEATURES `-` if
none.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import posixpath
import re
import subprocess
import sys
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parent.parent

# The site (ADR-0067, rule 2).
BASE_URL = "https://tetracl.github.io/tetraloma/"
BASE_PATH = "/tetraloma/"
LIBRARY_CRATES = [
    "tetraloma",
    "tetraloma-dims",
    "tetraloma-refcell",
    "tetraloma-device",
    "tetraloma-parallel",
    "tetraloma-topology",
    "tetraloma-geometry",
    "tetraloma-partition",
    "tetraloma-io",
]
BOOK_DIR = ROOT / "docs/user"
ALLOWLIST = BOOK_DIR / "doc-audit-allow.toml"
HARNESS = ROOT / "tetraloma-examples/src/book.rs"
RUN_EXAMPLES = ROOT / "scripts/run-examples.sh"
D1_PLAN = ROOT / "docs/plans/D1.md"

# A heading with fewer words of prose under it (its subsections included) is a stub (check f).
MIN_WORDS = 20

# The book's outline (ADR-0067, rule 4; docs/plans/D1.md, "Book outline"): file, title and the
# fixed top-level headings, which the API docs link to. Chapters may add headings, not rename these.
OUTLINE = [
    ("introduction.md", "Introduction",
     ["What tetraloma is", "Features", "Status", "How this documentation is organised"]),
    ("getting-started/installation.md", "Installation",
     ["Toolchain", "Adding tetraloma to a project", "Runtime features of CubeCL",
      "System libraries"]),
    ("getting-started/first-program.md", "A first program",
     ["The program", "Opening a device", "Building a mesh", "Geometric data",
      "Writing a VTU file", "Running it"]),
    ("concepts/meshes.md", "Meshes and topology",
     ["Cells and entities", "Connectivity", "Cell blocks", "Ownership and ghosts", "Origin ids"]),
    ("concepts/reference-cells.md", "Reference cells and maps",
     ["Reference cells", "Sub-entities", "The reference-to-physical map"]),
    ("concepts/geometry.md", "Lagrange geometry",
     ["Coordinate elements", "High-order geometry", "GLL and equispaced nodes"]),
    ("concepts/geometric-factors.md", "Geometric factors",
     ["The Jacobian", "The determinant", "The pseudo-inverse", "Measure", "Normals",
      "Facet normals"]),
    ("concepts/orientation.md", "Orientation", ["Manifold orientation", "Permutation bits"]),
    ("concepts/quadrature.md", "Quadrature", ["Quadrature rules", "Facet rules", "Tabulation"]),
    ("concepts/pull-back.md", "The pull-back",
     ["The problem", "The affine start", "The iteration", "Convergence and failure"]),
    ("concepts/typed-dimensions.md", "Typed dimensions",
     ["The mesh type", "Codimension", "Methods by dimension", "AnyMesh"]),
    ("concepts/precision.md", "Precision", ["f32 and f64", "Tolerances", "Determinism"]),
    ("elements/reference-cells.md", "Reference cell definitions",
     ["Interval", "Triangle", "Quadrilateral", "Tetrahedron", "Hexahedron", "Prism", "Pyramid"]),
    ("elements/lagrange.md", "Lagrange nodes", ["Node counts", "Node ordering", "Node families"]),
    ("elements/orientation-bits.md", "Orientation bits", ["Edge bits", "Face bits", "Storage"]),
    ("elements/file-formats.md", "File formats", ["Gmsh", "VTK"]),
    ("elements/basix-dolfinx.md", "Basix and DOLFINx",
     ["Cell and vertex numbering", "Node orderings", "Converting a mesh"]),
    ("tutorials/build-mesh.md", "Building a mesh",
     ["From arrays", "Several cell types", "Higher-degree geometry", "Errors"]),
    ("tutorials/read-write.md", "Reading and writing files",
     ["Reading Gmsh", "Fixing the dimensions", "Writing Gmsh", "Writing VTU", "Writing PVTU",
      "Writing VTKHDF"]),
    ("tutorials/iterate.md", "Iterating over a mesh",
     ["Blocks", "Cells", "Entities", "Connectivity", "Reading data back to the host"]),
    ("tutorials/geometric-data.md", "Geometric data",
     ["Points", "Jacobians", "Determinants", "Pseudo-inverses", "Normals", "Measures"]),
    ("tutorials/quadrature.md", "Quadrature on a mesh",
     ["Rules", "Integrating over a mesh", "High-order geometry"]),
    ("tutorials/pull-back.md", "Pulling back points",
     ["Points to reference coordinates", "Checking the result", "Points outside a cell"]),
    ("tutorials/tags-submeshes.md", "Tags and submeshes",
     ["Physical groups", "Regions", "Boundary meshes", "Surface meshes", "Orientation"]),
    ("tutorials/distributed.md", "Distributed meshes",
     ["Building on a communicator", "Partitioners", "Ghost modes", "Cell ordering"]),
    ("tutorials/halo-exchange.md", "Halo exchange",
     ["Index maps", "Scatterers", "Forward", "Reverse"]),
    ("tutorials/parallel-io.md", "Parallel I/O",
     ["Reading Gmsh in parallel", "PVTU output", "VTKHDF output"]),
    ("tutorials/kernels.md", "Writing kernels",
     ["DeviceGrid", "GeometryBlock", "BlockFields", "A kernel that integrates",
      "Generic algorithms", "with_mesh!"]),
    ("devices.md", "Compute devices",
     ["The CPU runtime", "Metal", "CUDA", "Vulkan", "HIP and wgpu",
      "Choosing a device at run time", "f64 support", "One GPU per MPI rank",
      "Host–device transfers", "Performance settings"]),
    ("features.md", "Cargo features",
     ["Runtime features", "MPI", "HDF5", "Partitioners", "Reserved features"]),
    ("troubleshooting.md", "Troubleshooting",
     ["Build problems", "Device problems", "MPI problems", "Numerical problems"]),
    ("glossary.md", "Glossary", []),
]

# Developer citations (check c), searched in the prose of rendered doc blocks (code excluded),
# except DOC_PATH, which is searched in code too.
DOC_PATH = re.compile(r"\bdocs/[\w./-]+|\b(?:IMPLEMENTATION|CLAUDE)\.md\b|\bdesign\.md\b")
PROSE_CITATIONS = [
    ("ADR", re.compile(r"\bADR[- ]?\d+\b")),
    ("plan question", re.compile(r"\b(?:plan|plans)\b[^.]{0,40}\bquestion\s+\d+|\bquestion\s+\d+\b", re.I)),
    ("milestone", re.compile(r"\b(?:M\d{1,2}[a-z]?|E\d)\b")),
    ("task", re.compile(r"\bT\d{1,2}\b")),
    ("design listing", re.compile(r"\bListings?\s+[A-Z]\d+[a-z]?\b")),
    ("design appendix", re.compile(r"\bAppendix\s+[A-Z]\b")),
    ("roadmap", re.compile(r"\b[Rr]oadmap\b")),
]

# Rustdoc's markup at the pinned toolchain (1.99), in one place. The audit fails if it finds no
# items, so a changed markup cannot pass silently.
ITEM_PAGE = re.compile(
    r"^(struct|enum|union|trait|type|constant|static|fn|macro|derive|attr|traitalias)\.(.+)\.html$"
)
MODULE_PAGE = "index.html"
MAIN_CONTENT_ID = "main-content"
BLANKET_IMPLS_ID = "blanket-implementations-list"
TOP_DOC_CLASS = "top-doc"
DOCBLOCK_CLASS = "docblock"
MAIN_HEADING_CLASS = "main-heading"
SRC_CLASS = "src"
IMPLEMENTATIONS_ID = "implementations-list"
TRAIT_METHODS_CLASS = "methods"
METHOD_SECTION_CLASS = "method"
METHOD_ID = re.compile(r"^(?:ty)?method\.([A-Za-z_][A-Za-z0-9_]*)$")
ITEM_TABLE_CLASS = "item-table"
SRC_HREF = re.compile(r"(?:^|/)src/([^/]+)/(.+)\.html#(\d+)(?:-(\d+))?$")
DOCTEST_LINE = re.compile(r"^(?P<file>\S+\.rs) - (?P<name>.+?) \(line (?P<line>\d+)\)(?P<kind>.*): test$")

VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
        "track", "wbr"}


# ---------------------------------------------------------------------------------------------
# A minimal DOM


@dataclass(eq=False)
class Node:
    tag: str
    attrs: dict
    parent: "Node | None" = None
    children: list = field(default_factory=list)

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def has_class(self, c: str) -> bool:
        return c in self.classes()

    def iter(self):
        yield self
        for ch in self.children:
            if isinstance(ch, Node):
                yield from ch.iter()

    def find_all(self, pred):
        return [n for n in self.iter() if pred(n)]

    def find(self, pred):
        return next((n for n in self.iter() if pred(n)), None)

    def ancestors(self):
        n = self.parent
        while n is not None:
            yield n
            n = n.parent

    def text(self, skip=lambda n: False) -> str:
        out = []

        def walk(n):
            for ch in n.children:
                if isinstance(ch, str):
                    out.append(ch)
                elif not skip(ch):
                    walk(ch)

        walk(self)
        return "".join(out)


class _Builder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        n = Node(tag, {k: (v or "") for k, v in attrs}, self.cur)
        self.cur.children.append(n)
        if tag not in VOID:
            self.cur = n

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, {k: (v or "") for k, v in attrs}, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        if self.cur.tag not in ("script", "style"):
            self.cur.children.append(data)


def parse_html(text: str) -> Node:
    b = _Builder()
    b.feed(text)
    b.close()
    return b.root


# ---------------------------------------------------------------------------------------------
# Results


class Results:
    """Failures per check and crate (or book chapter), with one line of detail each."""

    def __init__(self):
        self.failures: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        self.ran: list[str] = []
        self.notes: list[str] = []

    def fail(self, check: str, scope: str, detail: str):
        self.failures[check][scope].append(detail)

    def total(self) -> int:
        return sum(len(v) for c in self.failures.values() for v in c.values())


CHECK_NAMES = {
    "a": "site builds without warnings",
    "b": "links and anchors resolve",
    "c": "no developer citations",
    "d": "items link to the book",
    "e": "functions have running examples",
    "f": "outline complete, no stubs",
    "g": "code blocks",
    "h": "library sources: comments only",
}


# ---------------------------------------------------------------------------------------------
# (a) build logs


def check_logs(work: Path, res: Results):
    logs = work / "logs"
    mdlog = logs / "mdbook.log"
    if not mdlog.exists():
        res.fail("a", "book", f"{mdlog} missing (build the site first)")
    else:
        for line in mdlog.read_text(errors="replace").splitlines():
            plain = re.sub(r"\x1b\[[0-9;]*m", "", line)
            if re.search(r"\b(WARN|ERROR)\b", plain):
                res.fail("a", "book", plain.strip())
    rlogs = sorted(logs.glob("rustdoc-*.log"))
    if not rlogs:
        res.fail("a", "api", f"no rustdoc logs in {logs} (build the site first)")
    for log in rlogs:
        crate = log.stem.removeprefix("rustdoc-")
        for line in log.read_text(errors="replace").splitlines():
            if re.match(r"^\s*(warning|error)(\[\w+\])?:", line):
                res.fail("a", crate, line.strip())


# ---------------------------------------------------------------------------------------------
# (b) links

HREF = re.compile(r"""<(a|link|img|script|iframe|source)\b[^>]*?\s(href|src)\s*=\s*("([^"]*)"|'([^']*)')""", re.I)
IDS = re.compile(r"""\s(?:id|name)\s*=\s*("([^"]*)"|'([^']*)')""", re.I)
BASE_TAG = re.compile(r"""<base\s[^>]*href\s*=\s*"([^"]*)\"""", re.I)


REDIRECT = re.compile(r"""<meta http-equiv="refresh" content="0;URL=([^"]+)">""")


def redirect_of(p: Path) -> Path | None:
    """The page that rustdoc's redirect stub `p` forwards to, or None if `p` is no stub."""
    if p.suffix != ".html":
        return None
    with p.open(errors="replace") as fh:
        head = fh.read(600)
    m = REDIRECT.search(head)
    return (p.parent / m.group(1)).resolve() if m else None


def check_links(site: Path, res: Results):
    site = site.resolve()
    ids_cache: dict[Path, set[str]] = {}

    def ids_of(p: Path) -> set[str]:
        if p not in ids_cache:
            t = p.read_text(errors="replace")
            ids_cache[p] = {html.unescape(m.group(2) or m.group(3) or "") for m in IDS.finditer(t)}
        return ids_cache[p]

    pages = sorted(site.rglob("*.html"))
    if not pages:
        res.fail("b", "site", f"no HTML pages in {site}")
        return
    for page in pages:
        rel = page.relative_to(site).as_posix()
        if rel.startswith("api/src/"):
            scope = "api-src"
        elif rel.startswith("api/"):
            scope = rel.split("/")[1] if rel.count("/") >= 2 else "api"
        else:
            scope = "book"
        text = page.read_text(errors="replace")
        # Docs of blanket impls are copied from foreign crates (std, tracing, ...), with their
        # relative links; they are not ours to fix.
        cut = text.find(f'id="{BLANKET_IMPLS_ID}"')
        if cut >= 0:
            text = text[:cut]
        base_m = BASE_TAG.search(text)
        base_dir = posixpath.dirname(rel)
        if base_m:
            b = base_m.group(1)
            if b.startswith(BASE_PATH):
                base_dir = b[len(BASE_PATH):].rstrip("/")
        for m in HREF.finditer(text):
            raw = html.unescape(m.group(4) if m.group(4) is not None else (m.group(5) or ""))
            target = raw.strip()
            if not target or target.startswith(("mailto:", "javascript:", "data:", "tel:")):
                continue
            from_book = not rel.startswith("api/")
            if target.startswith(BASE_URL):
                local = target[len(BASE_URL):]
                origin = "absolute"
            elif target.startswith(BASE_PATH):
                local = target[len(BASE_PATH):]
                origin = "absolute"
            elif re.match(r"^[a-z][a-z0-9+.-]*:", target, re.I) or target.startswith("//"):
                continue  # external
            else:
                local = target
                origin = "relative"
            parts = urlsplit(local)
            path, frag, raw_frag = unquote(parts.path), unquote(parts.fragment), parts.fragment
            if origin == "relative":
                if path == "":
                    tpath = rel
                else:
                    tpath = posixpath.normpath(posixpath.join(base_dir, path))
            else:
                tpath = posixpath.normpath(path) if path else ""
            if tpath in ("", "."):
                tpath = "index.html"
            if tpath.startswith("../"):
                res.fail("b", scope, f"{rel}: {raw} leaves the site")
                continue
            fs = site / tpath
            if fs.is_dir() or local.endswith("/") and path:
                fs = fs / "index.html"
                tpath = posixpath.join(tpath, "index.html")
            if not fs.exists():
                res.fail("b", scope, f"{rel}: {raw} -> {tpath} does not exist")
                continue
            redirected = redirect_of(fs)
            if redirected is not None:
                fs = redirected
                tpath = fs.relative_to(site).as_posix()
                if not fs.exists():
                    res.fail("b", scope, f"{rel}: {raw} redirects to {tpath}, which does not exist")
                    continue
            if from_book and tpath.startswith("api/") and frag:
                res.fail("b", scope, f"{rel}: {raw} links to an anchor on an API page (link the item page)")
                continue
            if not frag or fs.suffix != ".html":
                continue
            if tpath.startswith("api/src/") and re.fullmatch(r"\d+(-\d+)?", frag):
                continue  # line anchors of source pages (rustdoc's script)
            ids = ids_of(fs)
            if frag not in ids and raw_frag not in ids:
                res.fail("b", scope, f"{rel}: {raw} -> #{frag} not in {tpath}")


# ---------------------------------------------------------------------------------------------
# API pages: items, (c), (d), (e)


@dataclass
class Item:
    kind: str  # module, struct, ..., fn, method
    path: str  # public path as rendered (crate::module::Name[::method])
    name: str
    owner: str | None  # type or trait of a method
    page: str  # page relative to the API root
    crate: str  # crate whose sources define it (crate name, underscores)
    src: tuple | None  # (crate name, file under src/, first line, last line)
    doc: Node | None  # the item's own doc block

    def key(self):
        # Inlined copies give the first line only (`#46`), the defining page a range (`#46-48`).
        return (self.src[:3] if self.src else None, self.kind, self.name, self.owner)


def _src_of(node: Node | None):
    if node is None:
        return None
    a = node.find(lambda n: n.tag == "a" and n.has_class(SRC_CLASS))
    if a is None:
        return None
    m = SRC_HREF.search(a.attrs.get("href", ""))
    if not m:
        return None
    return (m.group(1), m.group(2), int(m.group(3)), int(m.group(4) or m.group(3)))


def _doc_after_section(section: Node) -> Node | None:
    """The doc block of a method whose `<section>` is in a `<summary>` of a toggle."""
    summ = section.parent
    if summ is None or summ.tag != "summary" or summ.parent is None:
        return None
    for ch in summ.parent.children:
        if isinstance(ch, Node) and ch.tag == "div" and ch.has_class(DOCBLOCK_CLASS):
            return ch
    return None


def parse_api_page(api: Path, page: Path):
    """The items of one rustdoc page and all of its doc blocks (for check c), or None for a
    redirect or a page that is not an item page."""
    rel = page.relative_to(api).as_posix()
    m = ITEM_PAGE.match(page.name)
    if page.name != MODULE_PAGE and not m:
        return None
    text = page.read_text(errors="replace")
    if 'http-equiv="refresh"' in text[:2000]:
        return None
    root = parse_html(text)
    main = root.find(lambda n: n.attrs.get("id") == MAIN_CONTENT_ID)
    if main is None:
        return None
    dirs = rel.split("/")[:-1]
    page_crate = dirs[0]
    if m:
        kind, name = m.group(1), m.group(2)
        path = "::".join(dirs + [name])
    else:
        kind, name = ("crate" if len(dirs) == 1 else "module"), dirs[-1]
        path = "::".join(dirs)
    heading = main.find(lambda n: n.has_class(MAIN_HEADING_CLASS))
    src = _src_of(heading)
    top = main.find(lambda n: n.tag == "details" and n.has_class(TOP_DOC_CLASS))
    doc = None
    if top is not None:
        doc = next((c for c in top.children if isinstance(c, Node) and c.has_class(DOCBLOCK_CLASS)), None)
    items = [Item(kind, path, name, None, rel, src[0] if src else page_crate, src, doc)]

    regions = []
    if kind == "trait":
        regions = main.find_all(lambda n: n.tag == "div" and n.has_class(TRAIT_METHODS_CLASS))
    else:
        impls = main.find(lambda n: n.attrs.get("id") == IMPLEMENTATIONS_ID)
        if impls is not None:
            regions = [impls]
    for region in regions:
        for sec in region.find_all(lambda n: n.tag == "section" and n.has_class(METHOD_SECTION_CLASS)):
            mm = METHOD_ID.match(sec.attrs.get("id", ""))
            if not mm or sec.has_class("trait-impl"):
                continue
            msrc = _src_of(sec)
            items.append(Item("method", f"{path}::{mm.group(1)}", mm.group(1), name, rel,
                              msrc[0] if msrc else page_crate, msrc, _doc_after_section(sec)))

    blocks = main.find_all(lambda n: (n.tag == "div" and n.has_class(DOCBLOCK_CLASS))
                           or (n.tag == "dd" and n.parent is not None and n.parent.has_class(ITEM_TABLE_CLASS)))
    # Nested doc blocks are searched with their parents.
    blocks = [b for b in blocks if not any(a in blocks for a in b.ancestors())]
    return page_crate, items, blocks


def book_links(doc: Node | None) -> list[str]:
    if doc is None:
        return []
    out = []
    for a in doc.find_all(lambda n: n.tag == "a"):
        h = a.attrs.get("href", "")
        if h.startswith(BASE_URL):
            rest = h[len(BASE_URL):]
            if not rest.startswith("api/") and re.match(r"^[\w./-]+\.html(#[\w.-]*)?$", rest):
                out.append(h)
    return out


def has_example_heading(doc: Node | None) -> bool:
    if doc is None:
        return False
    for h in doc.find_all(lambda n: re.fullmatch(r"h[1-6]", n.tag) is not None):
        if h.text().strip().lstrip("§").strip() in ("Example", "Examples"):
            return True
    return False


def code_free_text(node: Node) -> str:
    return node.text(skip=lambda n: n.tag in ("code", "pre"))


def strip_generics(s: str) -> str:
    out, depth = [], 0
    for ch in s:
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def load_doctests(work: Path):
    """Running doctests: {(crate dir, file under src/): [(line, last name, owner name)]}."""
    listed = (work / "doctests.txt").read_text(errors="replace").splitlines()
    ignored_path = work / "doctests-ignored.txt"
    ignored = set(ignored_path.read_text(errors="replace").splitlines()) if ignored_path.exists() else set()
    tests = defaultdict(list)
    sources: dict[str, list[str]] = {}
    for line in listed:
        line = line.strip()
        m = DOCTEST_LINE.match(line)
        if not m or line in ignored or m.group("kind").strip():
            continue  # not a doctest, ignored, or compiled only
        f = m.group("file")
        if "/src/" not in f:
            continue
        # `--list` does not mark `no_run`: read the attributes of the opening fence, the listed
        # line of the source.
        if f not in sources:
            p = ROOT / f
            sources[f] = p.read_text(errors="replace").splitlines() if p.exists() else []
        src_lines = sources[f]
        n = int(m.group("line"))
        fence = src_lines[n - 1] if 0 < n <= len(src_lines) else ""
        fm = re.search(r"(```+|~~~+)\s*(.*)$", fence)
        attrs = set(re.split(r"[,\s]+", fm.group(2))) if fm else set()
        if attrs & {"no_run", "ignore", "compile_fail"}:
            continue
        crate_dir, sub = f.split("/src/", 1)
        segs = strip_generics(m.group("name")).replace(" ", "").split("::")
        tests[(crate_dir.split("/")[-1], sub)].append(
            (int(m.group("line")), segs[-1], segs[-2] if len(segs) > 1 else None))
    return tests


def load_allowlist(res: Results):
    if not ALLOWLIST.exists():
        res.fail("e", "allowlist", f"{ALLOWLIST} missing")
        return {}
    data = tomllib.loads(ALLOWLIST.read_text())
    allow = {}
    for entry in data.get("allow", []):
        item, reason = entry.get("item"), entry.get("reason", "").strip()
        if not item or not reason:
            res.fail("e", "allowlist", f"entry without item or reason: {entry}")
            continue
        allow[item] = reason
    return allow


def check_api(api: Path, work: Path, checks: str, only: str | None, res: Results):
    crates_us = [c.replace("-", "_") for c in LIBRARY_CRATES]
    # Dependencies first, so that an item inlined into the facade is seen where it is defined.
    order = [c for c in DOC_ORDER_US if (api / c).is_dir()] or [c for c in crates_us if (api / c).is_dir()]
    if only:
        only = only.replace("-", "_")
        if not (api / only).is_dir():
            res.fail("d", only, f"no rustdoc output at {api / only}")
            return
    items: dict = {}
    seen_blocks: set[str] = set()
    n_pages = 0
    for crate in order:
        for page in sorted((api / crate).rglob("*.html")):
            parsed = parse_api_page(api, page)
            if parsed is None:
                continue
            n_pages += 1
            page_crate, page_items, blocks = parsed
            for it in page_items:
                if it.crate not in crates_us:
                    continue  # inlined from a crate outside the nine
                k = it.key()
                if k in items:
                    items[k][1].append(it.path)
                else:
                    items[k] = (it, [it.path])
            if "c" in checks:
                for b in blocks:
                    prose = code_free_text(b)
                    full = b.text()
                    h = hashlib.sha1(full.encode()).hexdigest()
                    if h in seen_blocks:
                        continue
                    seen_blocks.add(h)
                    if only and page_crate != only:
                        continue
                    found = [m.group(0) for m in DOC_PATH.finditer(full)]
                    for label, rx in PROSE_CITATIONS:
                        found += [f"{label} '{m.group(0)}'" for m in rx.finditer(prose)]
                    if found:
                        res.fail("c", page_crate, f"{page.relative_to(api)}: {'; '.join(sorted(set(found)))}")
    if n_pages == 0 or not items:
        print(f"doc_audit: no items found under {api} (rustdoc markup changed?)", file=sys.stderr)
        sys.exit(2)
    if not any(it.kind == "method" for it, _ in items.values()):
        print("doc_audit: no methods found (rustdoc markup changed?)", file=sys.stderr)
        sys.exit(2)

    audited = [(it, paths) for it, paths in items.values() if not only or it.crate == only]
    if "d" in checks:
        for it, _ in audited:
            if not book_links(it.doc):
                res.fail("d", it.crate, f"{it.path} ({it.kind}, {it.page})")

    if "e" in checks:
        tests = load_doctests(work)
        allow = load_allowlist(res)
        used_allow = set()
        # Every item's start line per file and name, to give each doctest to the nearest item after
        # it with its name.
        by_file = defaultdict(list)
        for it, _ in items.values():
            if it.src:
                by_file[(it.src[0], it.src[1])].append(it)
        credited = set()
        for (crate_dir, sub), lst in tests.items():
            key = (crate_dir.replace("-", "_"), sub.removesuffix(".html"))
            cands = by_file.get(key, [])
            for line, name, owner in lst:
                best = None
                for it in cands:
                    if it.name != name or it.src[2] <= line:
                        continue
                    if it.kind == "method" and owner is not None and it.owner != owner:
                        continue
                    if best is None or it.src[2] < best.src[2]:
                        best = it
                if best is not None:
                    credited.add(best.key())
        for it, paths in audited:
            if it.kind not in ("fn", "method"):
                continue
            allowed = next((p for p in paths if p in allow), None)
            ok = it.key() in credited and has_example_heading(it.doc)
            if allowed:
                used_allow.add(allowed)
                if ok:
                    res.fail("e", "allowlist", f"{allowed}: has a running example; remove its entry")
                continue
            if not ok:
                why = "no running doctest" if it.key() not in credited else "no `# Example` heading"
                res.fail("e", it.crate, f"{it.path} ({why}; {it.page})")
        if not only:
            for p in sorted(set(allow) - used_allow):
                res.fail("e", "allowlist", f"{p}: names no audited function or method")
    n_fn = sum(1 for it, _ in audited if it.kind in ("fn", "method"))
    res.notes.append(f"API audit: {len(audited)} items ({n_fn} functions and methods) in "
                     f"{only or 'the nine crates'}, from {n_pages} pages")


# ---------------------------------------------------------------------------------------------
# The book: (f), (g)

FENCE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")
LIST_ITEM = re.compile(r"^(\s*)([-*+]|\d{1,9}[.)])(\s+|$)")
HEADING = re.compile(r"^ {0,3}(#{1,6})\s+(.*?)\s*#*\s*$")
INCLUDE = re.compile(r"^\{\{#include\s+([^}\s:]+)(?::[^}\s]*)?\s*\}\}$")


def summary_chapters(src: Path) -> list[str]:
    text = (src / "SUMMARY.md").read_text()
    return [m.group(1) for m in re.finditer(r"\]\(([^)\s]+\.md)\)", text)]


def scan_markdown(text: str):
    """(headings [(line, level, title)], fences [(line, info, body lines, indent)],
    indented code lines, prose lines [(line, text)])"""
    lines = text.split("\n")
    headings, fences, indented, prose = [], [], [], []
    i, n = 0, len(lines)
    list_indents: list[int] = []
    prev_blank = True
    in_comment = False
    in_math = False
    while i < n:
        line = lines[i]
        if in_comment:
            if "-->" in line:
                in_comment = False
            i += 1
            continue
        if line.strip().startswith("<!--") and "-->" not in line:
            in_comment = True
            i += 1
            continue
        fm = FENCE.match(line)
        if fm:
            marker, info = fm.group(2), fm.group(3).strip()
            start = i
            body = []
            i += 1
            while i < n and not re.match(r"^ {0,3}" + re.escape(marker[0]) + "{" + str(len(marker)) + r",}\s*$", lines[i]):
                body.append(lines[i])
                i += 1
            fences.append((start + 1, info, body, len(fm.group(1))))
            i += 1
            prev_blank = False
            continue
        if line.strip() == "$$":
            in_math = not in_math
            i += 1
            continue
        if in_math:
            i += 1
            continue
        if not line.strip():
            prev_blank = True
            i += 1
            continue
        indent = len(line) - len(line.lstrip(" "))
        if "\t" in line[: indent + 1]:
            indented.append(i + 1)
        lm = LIST_ITEM.match(line)
        while list_indents and indent < list_indents[-1] and prev_blank:
            list_indents.pop()
        if lm:
            while list_indents and indent < list_indents[-1]:
                list_indents.pop()
            list_indents.append(len(lm.group(1)) + len(lm.group(2)) + max(1, len(lm.group(3))))
        else:
            base = list_indents[-1] if list_indents else 0
            if prev_blank and indent >= base + 4:
                indented.append(i + 1)
            elif not list_indents and prev_blank and indent >= 4:
                indented.append(i + 1)
        hm = HEADING.match(line)
        if hm:
            headings.append((i + 1, len(hm.group(1)), hm.group(2).strip()))
        else:
            prose.append((i + 1, line))
        prev_blank = False
        i += 1
    return headings, fences, indented, prose


def count_words(s: str) -> int:
    s = re.sub(r"\$[^$]*\$", " ", s)  # inline maths
    s = re.sub(r"\]\([^)]*\)", "]", s)  # link targets
    s = re.sub(r"<[^>]+>", " ", s)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’_-]*", s))


def run_examples_programs() -> set[str]:
    text = RUN_EXAMPLES.read_text()
    progs = set()
    for m in re.finditer(r'^(serial_examples|mpi_examples)="([^"]*)"', text, re.M):
        progs |= set(m.group(2).split())
    return progs


def check_book(checks: str, res: Results):
    src = BOOK_DIR / "src"
    chapters = summary_chapters(src)
    if "f" in checks:
        for f, title, heads in OUTLINE:
            if f not in chapters:
                res.fail("f", f, "not in SUMMARY.md")
            p = src / f
            if not p.exists():
                res.fail("f", f, "missing")
                continue
            hs, _, _, _ = scan_markdown(p.read_text())
            h1 = [t for _, lvl, t in hs if lvl == 1]
            if h1[:1] != [title]:
                res.fail("f", f, f"title {h1[:1]}, expected '# {title}'")
            h2 = [t for _, lvl, t in hs if lvl == 2]
            missing = [h for h in heads if h not in h2]
            if missing:
                res.fail("f", f, f"fixed headings missing or renamed: {missing}")
            elif [h for h in h2 if h in heads] != heads:
                res.fail("f", f, "fixed headings out of order")
        for f in chapters:
            p = src / f
            if not p.exists():
                res.fail("f", f, "in SUMMARY.md but missing")
                continue
            hs, _, _, prose = scan_markdown(p.read_text())
            for k, (line, lvl, t) in enumerate(hs):
                end = next((l for l, lv, _ in hs[k + 1:] if lv <= lvl), 10**9)
                words = sum(count_words(s) for l, s in prose if line < l < end)
                if words < MIN_WORDS:
                    res.fail("f", f, f"line {line}: '{'#' * lvl} {t}' has {words} words (stub; at least {MIN_WORDS})")
    if "g" in checks:
        progs = run_examples_programs()
        if not progs:
            res.fail("g", "run-examples.sh", "no program lists (serial_examples, mpi_examples) found")
        harness = HARNESS.read_text() if HARNESS.exists() else ""
        for f in chapters:
            p = src / f
            if not p.exists():
                continue
            if f'"../../docs/user/src/{f}"' not in harness:
                res.fail("g", f, f"not included by the doctest harness {HARNESS.relative_to(ROOT)}")
            _, fences, indented, _ = scan_markdown(p.read_text())
            for line in indented:
                res.fail("g", f, f"line {line}: indented code block")
            for line, info, body, _ in fences:
                words = [w for w in re.split(r"[,\s]+", info) if w]
                if not words:
                    res.fail("g", f, f"line {line}: fence without a language tag")
                    continue
                includes = [l for l in body if "{{#" in l]
                if "ignore" in words:
                    nonblank = [l.strip() for l in body if l.strip()]
                    if not nonblank:
                        res.fail("g", f, f"line {line}: empty ignore block")
                    for l in nonblank:
                        im = INCLUDE.match(l)
                        if not im:
                            res.fail("g", f, f"line {line}: an ignore block may only include a program: {l!r}")
                            continue
                        target = (p.parent / im.group(1)).resolve()
                        try:
                            relt = target.relative_to(ROOT / "tetraloma-examples")
                        except ValueError:
                            res.fail("g", f, f"line {line}: includes {im.group(1)}, not a tetraloma-examples program")
                            continue
                        stem = target.stem
                        if not target.exists():
                            res.fail("g", f, f"line {line}: {im.group(1)} does not exist")
                        elif relt.parts[0] not in ("src", "examples") or stem not in progs:
                            res.fail("g", f, f"line {line}: {relt} is not a program that run-examples.sh runs")
                elif includes:
                    res.fail("g", f, f"line {line}: an include outside an ignore block")
                if words[0] == "rust" and "no_run" in words:
                    res.fail("g", f, f"line {line}: no_run block (book code runs or is an included program)")


# ---------------------------------------------------------------------------------------------
# (h)


def d1_open() -> bool:
    if not D1_PLAN.exists():
        return False
    m = re.search(r"^\- \*\*Status:\*\* (\w+)", D1_PLAN.read_text(), re.M)
    return not (m and m.group(1) == "done")


def d1_base() -> str | None:
    m = re.search(r"^\- \*\*Base commit:\*\* ([0-9a-f]{7,40})", D1_PLAN.read_text(), re.M)
    return m.group(1) if m else None


def check_comments_only(res: Results):
    if not d1_open():
        res.notes.append("(h) skipped: D1 is done")
        return
    base = d1_base()
    if not base:
        res.fail("h", "plan", f"no base commit in {D1_PLAN}")
        return
    sys.path.insert(0, str(ROOT / "scripts"))
    import comments_only  # noqa: E402

    # The tokenizer first proves itself on the planted cases.
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()) as out:
        wrong = comments_only.self_test()
    if wrong:
        res.fail("h", "comments_only.py", f"{wrong} planted case(s) judged wrongly:\n{out.getvalue()}")
    cwd = os.getcwd()
    os.chdir(ROOT)
    try:
        problems = comments_only.compare(base, LIBRARY_CRATES)
    finally:
        os.chdir(cwd)
    for p in problems:
        res.fail("h", p.split("/", 1)[0], p)


# ---------------------------------------------------------------------------------------------
# api-plan

DOC_ORDER_US: list[str] = []


def api_plan(features: str, crates: list[str] | None):
    meta = json.loads(subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"], cwd=ROOT,
        capture_output=True, text=True, check=True).stdout)
    deps = {p["name"]: {d["name"] for d in p["dependencies"] if d.get("kind") is None}
            for p in meta["packages"]}
    order, done = [], set()

    def visit(c):
        if c in done:
            return
        done.add(c)
        for d in sorted(deps.get(c, ())):
            if d in LIBRARY_CRATES:
                visit(d)
        order.append(c)

    for c in LIBRARY_CRATES:
        visit(c)
    tree = subprocess.run(
        ["cargo", "tree", "-e", "features,normal", "-p", "tetraloma", "--features", features,
         "--prefix", "none", "-f", "{p} [{f}]"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    feats = {}
    for line in tree.splitlines():
        m = re.match(r"^(\S+) v\S+(?: \([^)]*\))? \[([^\]]*)\]", line)
        if m and m.group(1) in LIBRARY_CRATES:
            feats[m.group(1)] = m.group(2)
    for c in order:
        if crates and c not in crates:
            continue
        print(c, feats.get(c) or "-")


# ---------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--site", type=Path, required=True)
    c.add_argument("--work", type=Path, required=True)
    c.add_argument("--checks", default="abcdefgh")
    c.add_argument("--crate")
    c.add_argument("--api", type=Path)
    c.add_argument("--verbose", action="store_true")
    p = sub.add_parser("api-plan")
    p.add_argument("--features", required=True)
    p.add_argument("--crates")
    args = ap.parse_args()

    if args.cmd == "api-plan":
        api_plan(args.features, args.crates.split(",") if args.crates else None)
        return 0

    global DOC_ORDER_US
    order_file = args.work / "api-order.txt"
    if order_file.exists():
        DOC_ORDER_US = [l.split()[0].replace("-", "_") for l in order_file.read_text().splitlines() if l.strip()]
    res = Results()
    checks = args.checks
    if "a" in checks:
        check_logs(args.work, res)
    if "b" in checks:
        check_links(args.site, res)
    if any(x in checks for x in "cde"):
        check_api(args.api or args.site / "api", args.work, checks, args.crate, res)
    if any(x in checks for x in "fg"):
        check_book(checks, res)
    if "h" in checks:
        check_comments_only(res)

    report = args.work / "report.txt"
    lines = []
    for ch in sorted(checks):
        per = res.failures.get(ch, {})
        n = sum(len(v) for v in per.values())
        lines.append(f"({ch}) {CHECK_NAMES[ch]}: {'pass' if n == 0 else f'FAIL, {n}'}")
        for scope in sorted(per):
            lines.append(f"      {scope}: {len(per[scope])}")
    details = []
    for ch in sorted(res.failures):
        for scope in sorted(res.failures[ch]):
            for d in res.failures[ch][scope]:
                details.append(f"({ch}) {scope}: {d}")
    args.work.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines + [""] + res.notes + [""] + details) + "\n")
    print("\n".join(lines))
    for note in res.notes:
        print(note)
    if args.verbose:
        print("\n".join(details))
    elif details:
        print(f"details: {report}")
    return 1 if res.total() else 0


if __name__ == "__main__":
    sys.exit(main())
