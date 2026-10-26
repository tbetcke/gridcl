# Gmsh fixture corpus (M4)

`generate.py` writes the committed corpus in `gridcl-io/tests/fixtures/gmsh/` with the pinned
Gmsh Python wheel, `gmsh==4.15.2` from PyPI (docs/plans/M4.md, T1 and question 13). It refuses
to run with any other Gmsh version.

```sh
python3 -m venv .venv && .venv/bin/pip install gmsh==4.15.2
.venv/bin/python scripts/fixtures/gmsh/generate.py          # rewrite the corpus
.venv/bin/python scripts/fixtures/gmsh/generate.py --check  # regenerate into a temporary
                                                            # directory, compare bytes
```

## Files

- `<cell>_p<p>.msh`: small axis-aligned boxes of one cell type and degree, every type of A.7 up
  to order 4 (pyramids up to order 3). Pyramids come from `hybrid_p<p>.msh`: hexahedra in
  [0, 1]³ beside tetrahedra in [1, 2] × [0, 1]², with pyramids between them. Every cell is an
  affine image of its reference cell. Degrees 1 and 2 also have a `-binary` copy.
- `<cell>_s2.msh`: Gmsh's serendipity elements (types 16–19); `twin` in the manifest names the
  same mesh as complete P2.
- `curve_2d_p2`, `curve_3d_p1`, `square_z1_p1`, `box_surface_p2`: D_GEO above D_TOP (a plane
  at z = 1 is 3D).
- `two_regions_p1`: triangles and quadrilaterals, physical groups in every dimension, named and
  unnamed, one entity in two groups and one name ("boundary") in two dimensions.
- `partitioned_p1` (the same mesh in two partitions, with `$PartitionedEntities`) and
  `periodic_p1` (with `$Periodic`).
- `duplicate_node_tags`, `duplicate_element_tags`: post-processed copies of `triangle_p1` with
  one repeated node tag or element tag; reading them must fail with `DuplicateOriginId`.

Every file is written with `Mesh.SaveAll = 1`, so elements outside physical groups are kept.

## Manifest

`manifest.json` records Gmsh's view of every file, as Gmsh reads it back
(`gridcl-io/tests/common/mod.rs` loads it):

- `num_nodes`: the file's node count;
- `num_mesh_nodes`: the distinct nodes of the D_TOP elements (for serendipity files, those of
  the complete-P2 twin);
- `element_counts`: per Gmsh type id;
- `tdim`, `gdim`: the expected (D_TOP, D_GEO), set by construction;
- `physical_groups`: per group its dimension, tag, name, size, and the sorted corner node tags
  of every element, sorted;
- `twin`, `partitioned`, `periodic`, and for the duplicate-tag files `expect_error`,
  `duplicate_id` and `duplicate_of`.
