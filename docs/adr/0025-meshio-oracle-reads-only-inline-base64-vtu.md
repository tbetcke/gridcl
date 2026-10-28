# ADR-0025: The meshio oracle reads only inline-base64 VTU files

- **Status:** Accepted
- **Date:** 2026-10-27
- **Raised in:** M5 T1, the risk check of docs/plans/M5.md ("check first that meshio 5.3.5 reads appended raw data and Lagrange types 68–73")
- **Design sections affected:** "Import and export / VTK" (the "Test oracles" bullet); "Implementation roadmap / M5 VTK output" (the third acceptance bullet)

## Context

M5's third acceptance bullet reads:

> - meshio reads identical raw arrays;

and "Import and export / VTK" says:

> **VTU** (XML with `version="2.2"`, appended raw or base64 binary, optional zlib). […]
> The version matters: VTK renumbers higher-order hexahedra in files with a version below 2.1
>
> **Test oracles:** […] meshio checks the raw arrays; it writes version 0.1 and applies no
> conversion. Pinned at `vtk==9.7.1` and `meshio==5.3.5`.

ADR-0024 (item 1) fixes the writer's header: `version="2.2"`, `header_type="UInt64"`, appended
raw data by default, inline base64 with `.base64(true)`, zlib with `.zlib(true)`. The T0 test
`meshio_reads_identical_raw_arrays` writes its data meshes in all four encodings (appended raw,
appended raw + zlib, base64, base64 + zlib) and expects meshio to read every file.

The plan's T1 risk note says: if meshio 5.3.5 cannot read one of the encodings, stop and raise
an ADR row instead of dropping the encoding. meshio 5.3.5 is the newest release on PyPI
(checked 2026-10-27).

## Problem

T1 wrote small files with VTK 9.7.1's own `vtkXMLUnstructuredGridWriter`: one cell of every
Lagrange type 68–73 at p = 2 and 3, a Float32 point array and an Int32 cell array, in ASCII,
inline binary (base64), appended raw and appended base64, each with and without zlib, and with
UInt32 and UInt64 headers. A second sweep wrote 40 files per encoding (2 to 41 points, P2 curves
and a line). meshio 5.3.5 ran under CPython 3.14.7 (the `.venv`) and its arrays were compared
with VTK's.

1. **meshio rejects every file whose version is not 0.1 or 1.0.** `meshio/vtu/_vtu.py`:

   ```python
   if version not in ["0.1", "1.0"]:
       raise ReadError(f"Unknown VTU file version '{version}'.")
   ```

   VTK 9.7.1 writes `version="2.3"`, gridcl will write `2.2`; meshio reads neither, in any
   encoding. gridcl cannot write 1.0 instead: VTK renumbers the hexahedra of older files. A
   VTK-written P2 and P3 file relabelled `version="1.0"` reads back in VTK with the nodes of
   hexahedron edges 10 and 11 swapped (P2: connectivity entries 46 and 47 exchanged; P3: 78–79
   with 80–81), without an error or warning. Relabelled `2.2`, it reads back identically.
2. **With the version relabelled to 1.0 (the only byte changed), meshio's results by encoding:**

   | Encoding | Lagrange cells, p = 2, 3 (UInt32 / UInt64 header) | Size sweep, 40 files each (UInt32 / UInt64) |
   | --- | --- | --- |
   | ASCII | reads, arrays equal | – |
   | inline base64 | reads, arrays equal / reads, arrays equal | 40 / 40 |
   | inline base64 + zlib | reads, arrays equal / reads, arrays equal | 40 / 40 |
   | appended raw | reads / `binascii.Error: Incorrect padding` | 0 / 0 |
   | appended raw + zlib | reads / `Incorrect padding` | 0 / 0 |
   | appended base64 | reads / `Incorrect padding` | 0 / 0 |
   | appended base64 + zlib | `Invalid base64-encoded string` / `Incorrect padding` | 3 / 2 |

   Lagrange types 68–73 map to meshio's `VTK_LAGRANGE_*` cell types and back; the points, the
   connectivity of every block and both data arrays are bitwise equal to VTK's whenever meshio
   reads the file.
3. **Why appended data fails.** meshio decodes an appended array by calling
   `base64.b64decode(appended[offset:])` on everything from the array's offset to the end of the
   section, which holds several separately padded base64 strings (for raw data, meshio first
   re-encodes each array). Its code assumes that decoding stops at the first padding. CPython
   3.13.15 and 3.14.7 (both checked) ignore padding in the middle in non-strict mode and then
   fail on the total length (`base64.b64decode(b'AAAAAAAAAAA=QUJD')` raises `Incorrect
   padding`). The result depends on the sizes of all later arrays, so it is not a property of
   the header type. With a decoder that stops at the first padding substituted for
   `base64.b64decode` inside meshio, 504 of the 506 files read. The two failures (4-point
   appended raw files) come from a second defect: meshio rewrites the `offset` attributes of the
   raw arrays one by one, and when a rewritten offset equals a raw offset it has not yet
   processed, it looks up the wrong `DataArray` (`len(offsets) != num_cells`).

So meshio 5.3.5 cannot read gridcl's files as specified (problem 1). Even past the version
check, it cannot read appended data (gridcl's default encoding) reliably (problems 2 and 3), and
this does not depend on gridcl's writer.

## Options

1. **Keep the design as written.** `meshio_reads_identical_raw_arrays` fails on every file
   (version), and on most appended files after any version workaround. Not possible with the
   pinned meshio.
2. **Write `version="1.0"`.** meshio then accepts the version, but VTK and ParaView renumber
   gridcl's Lagrange hexahedra (problem 1). The appended-data failures remain. Not chosen.
3. **Relabel the version for meshio only, and check appended files through their base64 twins**
   (proposed).
   - The oracle's `meshio` subcommand hands meshio a temporary copy of each file in which only
     the `version` attribute of the `<VTKFile>` start tag reads `1.0`. meshio has no other
     version-dependent code and applies no conversion, so the arrays it returns are those of the
     file. The VTK oracle always reads the unmodified file.
   - The meshio bullet covers the inline-base64 files, with and without zlib.
   - Every appended file of the meshio test (raw, with and without zlib) has a base64 twin with
     the same mesh and data. The test asserts that `parse_vtu` gives bitwise equal `VtuFile`s for
     both. meshio checks the twin, so the appended file's arrays are checked bitwise through the
     twin. VTK's reader reads every appended file in `vtk_reads_every_fixture`, as before.
   - No writer, API or pin changes.
4. **Patch meshio in the oracle.** Substitute a stop-at-padding decoder for meshio's
   `base64.b64decode` and skip its version check. This covers most appended files, but not all
   of them (the offset-collision defect, problem 3), and the oracle is then a modified meshio
   whose result depends on the patch. Not chosen.
5. **Pin an older Python for the `.venv`.** meshio's appended path may work on CPython 3.12 or
   older (not checked here). It still needs the version workaround and fails on the offset
   collision. It also ties the oracle to a Python that Homebrew and CI images will drop. Not
   chosen.
6. **Replace meshio with another raw-array oracle**, for example VTK's own arrays through
   `vtk.util.numpy_support`. VTK is the reader the first bullet already uses, so the second,
   independent reader the design asks for would be lost. Not chosen.

## Proposed decision

Option 3. The writer stays as ADR-0024 fixes it: `version="2.2"` is needed by VTK, and appended
raw stays the default for ParaView users. The two meshio limitations are meshio's own: its
version whitelist predates VTK's version-2 files, and its appended decoder relies on a CPython behaviour that
has since changed. Relabelling the version of a copy is a one-attribute change that meshio does
not otherwise act on, and the oracle reports it. Restricting meshio to inline base64 keeps it an
unmodified, independent reader of every byte it checks. The base64-twin equality extends the
bitwise check to appended data without trusting meshio's appended decoder. If a later meshio
release reads version 2.2 and appended data, the pin can move and the restriction can go.

## Consequences

- Text to change in docs/design.md:
  - "Import and export / VTK", "Test oracles": replace "meshio checks the raw arrays; it writes
    version 0.1 and applies no conversion." with "meshio checks the raw arrays of the inline-base64
    files (with and without zlib); it writes version 0.1 and applies no conversion. meshio 5.3.5
    accepts only versions 0.1 and 1.0, so the oracle hands it a copy whose version attribute reads
    1.0. Its appended-data decoder fails on VTK's own appended files under CPython ≥ 3.13, so
    appended files are checked through base64 twins (ADR-0025)."
  - Roadmap M5, the third acceptance bullet: "meshio reads identical raw arrays from the
    inline-base64 files (with and without zlib), and every appended file parses to the same arrays
    as its base64 twin (ADR-0025);".
- Public API changes: none.
- Acceptance tests affected:
  - `gridcl::vtk::meshio_reads_identical_raw_arrays`:
    - meshio reads the base64 and base64 + zlib files only;
    - each appended file (raw, raw + zlib) is written beside a base64 twin of the same case;
    - the test asserts `parse_vtu(appended) == parse_vtu(twin)` bitwise.

    `gridcl/tests/common/vtk.rs` (`data_cases`, the meshio corpus cases) and `gridcl/tests/vtk.rs`
    change accordingly.
  - The T1 oracle contract: `meshio FILE...` relabels the version of a temporary copy and adds
    `"version": "<original>"` to each result. No other test changes.
- Later milestones affected: M8 (VTKHDF) does not use meshio. If a later meshio release lifts
  both limitations, the restriction is removed when the pin moves.

## Decision record

Accepted by the maintainer on 2026-10-27, as proposed (option 3), without conditions. Applied to docs/design.md in `adr: 0025 apply`.
