//! A host mesh description to the file model.

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;

use super::cell_type::{cell_name, written_linear};
use super::{DataValues, VtkMesh, VtuFile, vtk_cell_type};

/// The number of Lagrange nodes of `cell` at `degree` (A.4), or `None` if it overflows.
///
/// The formulas of `gridcl_refcell::num_lagrange_nodes`, which panics on overflow instead.
fn checked_num_nodes(cell: CellType, degree: u32) -> Option<usize> {
    let p = u64::from(degree);
    let a = p.checked_add(1)?;
    let b = p.checked_add(2)?;
    let n = match cell {
        CellType::Point => 1,
        CellType::Interval => a,
        CellType::Triangle => a.checked_mul(b)? / 2,
        CellType::Quadrilateral => a.checked_mul(a)?,
        CellType::Tetrahedron => a.checked_mul(b)?.checked_mul(p.checked_add(3)?)? / 6,
        CellType::Hexahedron => a.checked_mul(a)?.checked_mul(a)?,
        CellType::Prism => a.checked_mul(a)?.checked_mul(b)? / 2,
        CellType::Pyramid => {
            let c = p.checked_mul(2)?.checked_add(3)?;
            a.checked_mul(b)?.checked_mul(c)? / 6
        }
    };
    usize::try_from(n).ok()
}

/// The VTK node order of a cell: entry k is the internal index of the cell's k-th VTK node
/// (A.9), at `degree` for a Lagrange cell and at degree 1 (the corners) for a linear one.
fn vtk_order(cell: CellType, degree: u32, linear: bool) -> Result<Vec<usize>> {
    let p = if linear { 1 } else { degree };
    // Unreachable after the checks in `to_vtu` (degree ≥ 1, no Lagrange pyramids).
    gridcl_refcell::vtk_to_internal(cell, p).map_err(|_| GridError::UnsupportedDegree {
        cell: cell_name(cell),
        degree,
    })
}

/// The file of `mesh`: every node as a point (in f32 if `f32_points`), and every cell with its
/// type from [`vtk_cell_type`](super::vtk_cell_type) and its nodes in VTK order (the inverse
/// of `gridcl_refcell::vtk_to_internal`). Linear cells use their corner nodes. Also returns
/// the number of pyramids of degree ≥ 2 written as linear cells with `lagrange`.
///
/// - **Points:** all nodes in node order, padded to three coordinates with zeros: Float64, or
///   Float32 (rounded to nearest) if `f32_points`. Nodes no cell references are written too,
///   such as the non-corner nodes of linear output (ADR-0024, item 4).
/// - **Cells:** in block order, each block's cells in order. VTK node k of a Lagrange cell is
///   its internal node `vtk_to_internal(cell, p)[k]`. A cell written linear (p = 1,
///   `lagrange == false`, or a pyramid of degree ≥ 2) uses its corners, which come first in
///   the internal order at every degree, in VTK's order `vtk_to_internal(cell, 1)`.
/// - **Offsets** are the cumulative ends of the cells in `connectivity`; there is no point or
///   cell data.
/// - **The count** is the number of pyramid cells of degree ≥ 2 if `lagrange`, else 0: the
///   cells the caller warns about (ADR-0024, item 5).
///
/// # Errors
///
/// The mesh is checked completely, so a malformed one gives an error, never a panic:
///
/// - `DimensionMismatch` if `gdim` is outside 1–3; `expected` holds the largest topological
///   dimension of the blocks and the nearest valid `gdim`;
/// - `LengthMismatch` if the length of `nodes` is not a multiple of `gdim`, or that of a
///   block's node list is not a multiple of `num_lagrange_nodes(cell, degree)`; `expected` is
///   the largest such multiple not above the length found;
/// - `UnsupportedDegree` for a block of degree 0;
/// - `IndexOutOfRange` for a node index past the last node;
/// - `TooLarge` if a node count or an offset overflows.
///
/// # Example
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_io::vtk::{VtkCellBlock, VtkMesh, to_vtu};
///
/// // The reference P2 quadrilateral, nodes in internal (A.5) order: tensor-ordered vertices,
/// // the edges (0,1), (0,2), (1,3), (2,3), then the centre.
/// let mesh = VtkMesh {
///     gdim: 2,
///     nodes: vec![
///         0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, // vertices
///         0.5, 0.0, 0.0, 0.5, 1.0, 0.5, 0.5, 1.0, // edges
///         0.5, 0.5, // interior
///     ],
///     blocks: vec![VtkCellBlock {
///         cell: CellType::Quadrilateral,
///         degree: 2,
///         nodes: (0..9).collect(),
///     }],
/// };
///
/// // VTK runs counter-clockwise round the vertices and the edges (A.9).
/// let (file, pyramids) = to_vtu(&mesh, true, false)?;
/// assert_eq!(file.types, [70]);
/// assert_eq!(file.connectivity, [0, 1, 3, 2, 4, 6, 7, 5, 8]);
/// assert_eq!((file.offsets[0], pyramids), (9, 0));
///
/// // Linear output writes the corners and leaves the other points unreferenced.
/// let (file, _) = to_vtu(&mesh, false, false)?;
/// assert_eq!((file.types[0], &file.connectivity[..]), (9, &[0, 1, 3, 2][..]));
///
/// // Malformed meshes give errors.
/// let mut bad = mesh.clone();
/// bad.blocks[0].nodes[8] = 9;
/// assert!(to_vtu(&bad, true, false).is_err());
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn to_vtu(mesh: &VtkMesh, lagrange: bool, f32_points: bool) -> Result<(VtuFile, usize)> {
    let gd = mesh.gdim;
    if !(1..=3).contains(&gd) {
        let tdim = mesh
            .blocks
            .iter()
            .map(|b| gridcl_refcell::tdim(b.cell))
            .max()
            .unwrap_or(0);
        return Err(GridError::DimensionMismatch {
            expected: (tdim, gd.clamp(tdim.max(1), 3)),
            found: (tdim, gd),
        });
    }
    let found = mesh.nodes.len();
    let num_nodes = found / gd;
    if !found.is_multiple_of(gd) {
        return Err(GridError::LengthMismatch {
            expected: found - found % gd,
            found,
        });
    }

    // Shapes and indices first, so that nothing below goes out of range.
    let mut num_cells = 0usize;
    let mut conn_len = 0usize;
    let mut sizes = Vec::with_capacity(mesh.blocks.len());
    for b in &mesh.blocks {
        if b.degree == 0 {
            return Err(GridError::UnsupportedDegree {
                cell: cell_name(b.cell),
                degree: 0,
            });
        }
        let k = checked_num_nodes(b.cell, b.degree).ok_or(GridError::TooLarge)?;
        let found = b.nodes.len();
        if !found.is_multiple_of(k) {
            return Err(GridError::LengthMismatch {
                expected: found - found % k,
                found,
            });
        }
        if let Some(&i) = b.nodes.iter().find(|&&i| i as usize >= num_nodes) {
            return Err(GridError::IndexOutOfRange {
                index: i as usize,
                len: num_nodes,
            });
        }
        let cells = found / k;
        let per_cell = if written_linear(b.cell, b.degree, lagrange) {
            gridcl_refcell::vertices(b.cell).len()
        } else {
            k
        };
        num_cells = num_cells.checked_add(cells).ok_or(GridError::TooLarge)?;
        conn_len = cells
            .checked_mul(per_cell)
            .and_then(|n| conn_len.checked_add(n))
            .ok_or(GridError::TooLarge)?;
        sizes.push((k, cells));
    }
    // Offsets are Int64.
    i64::try_from(conn_len).map_err(|_| GridError::TooLarge)?;

    let mut connectivity = Vec::with_capacity(conn_len);
    let mut offsets = Vec::with_capacity(num_cells);
    let mut types = Vec::with_capacity(num_cells);
    let mut pyramids = 0usize;
    for (b, &(k, cells)) in mesh.blocks.iter().zip(&sizes) {
        if cells == 0 {
            // No node order is generated for an empty block, whatever its degree.
            continue;
        }
        let linear = written_linear(b.cell, b.degree, lagrange);
        let order = vtk_order(b.cell, b.degree, linear)?;
        let ty = vtk_cell_type(b.cell, b.degree, lagrange);
        if lagrange && b.cell == CellType::Pyramid && b.degree >= 2 {
            pyramids = pyramids.checked_add(cells).ok_or(GridError::TooLarge)?;
        }
        for row in b.nodes.chunks_exact(k) {
            connectivity.extend(order.iter().map(|&m| i64::from(row[m])));
            // Cannot fail: the total was checked above.
            offsets.push(i64::try_from(connectivity.len()).map_err(|_| GridError::TooLarge)?);
            types.push(ty);
        }
    }

    let len = num_nodes.checked_mul(3).ok_or(GridError::TooLarge)?;
    let padded = mesh.nodes.chunks_exact(gd).flat_map(|x| {
        let mut c = [0.0; 3];
        c[..gd].copy_from_slice(x);
        c
    });
    let points = if f32_points {
        let mut v = Vec::with_capacity(len);
        v.extend(padded.map(|x| x as f32));
        DataValues::F32(v)
    } else {
        let mut v = Vec::with_capacity(len);
        v.extend(padded);
        DataValues::F64(v)
    };

    Ok((
        VtuFile {
            points,
            connectivity,
            offsets,
            types,
            point_data: Vec::new(),
            cell_data: Vec::new(),
        },
        pyramids,
    ))
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::vtk::VtkCellBlock;
    use gridcl_refcell::{NodeFamily, lagrange_points, num_lagrange_nodes, tdim, vertices};

    const CELLS: [CellType; 8] = [
        CellType::Point,
        CellType::Interval,
        CellType::Triangle,
        CellType::Quadrilateral,
        CellType::Tetrahedron,
        CellType::Hexahedron,
        CellType::Prism,
        CellType::Pyramid,
    ];

    /// One cell of `cell` and `degree` on its reference nodes (A.5), node i at index i.
    fn reference_cell(cell: CellType, degree: u32) -> VtkMesh {
        let td = tdim(cell);
        let nodes = if td == 0 {
            vec![0.0]
        } else {
            lagrange_points(cell, degree, NodeFamily::Equispaced).unwrap()
        };
        let k = num_lagrange_nodes(cell, degree) as u32;
        VtkMesh {
            gdim: td.max(1),
            nodes,
            blocks: vec![VtkCellBlock {
                cell,
                degree,
                nodes: (0..k).collect(),
            }],
        }
    }

    fn f64_points(file: &VtuFile) -> &[f64] {
        match &file.points {
            DataValues::F64(v) => v,
            other => panic!("Float64 points expected, found {other:?}"),
        }
    }

    #[test]
    fn convert_checked_num_nodes_agrees_with_refcell() {
        for cell in CELLS {
            for p in 0..=12 {
                assert_eq!(
                    checked_num_nodes(cell, p),
                    Some(num_lagrange_nodes(cell, p)),
                    "{cell:?} p = {p}"
                );
            }
            if cell != CellType::Point && cell != CellType::Interval {
                assert_eq!(checked_num_nodes(cell, u32::MAX), None, "{cell:?}");
            }
        }
    }

    /// A.9's "VTK Lagrange → internal" rows at p = 2. With node i of the cell at index i,
    /// VTK node k is point `map[k]`, so the connectivity is the row itself.
    #[test]
    fn convert_p2_cells_give_a9_rows() {
        let rows: [(CellType, &[i64]); 7] = [
            (CellType::Interval, &[0, 1, 2]),
            (CellType::Triangle, &[0, 1, 2, 3, 5, 4]),
            (CellType::Quadrilateral, &[0, 1, 3, 2, 4, 6, 7, 5, 8]),
            (CellType::Tetrahedron, &[0, 1, 2, 3, 4, 7, 5, 6, 8, 9]),
            (
                CellType::Hexahedron,
                &[
                    0, 1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 9, 16, 18, 19, 17, 10, 12, 15, 14, 22, 23,
                    21, 24, 20, 25, 26,
                ],
            ),
            (
                CellType::Prism,
                &[0, 1, 2, 3, 4, 5, 6, 9, 7, 12, 14, 13, 8, 10, 11, 15, 17, 16],
            ),
            (CellType::Pyramid, &[0, 1, 3, 2, 4]),
        ];
        for (cell, row) in rows {
            let mesh = reference_cell(cell, 2);
            let (file, pyramids) = to_vtu(&mesh, true, false).unwrap();
            assert_eq!(file.connectivity, row, "{cell:?}");
            assert_eq!(file.offsets, [row.len() as i64], "{cell:?}");
            assert_eq!(file.types, [vtk_cell_type(cell, 2, true)], "{cell:?}");
            assert_eq!(pyramids, usize::from(cell == CellType::Pyramid), "{cell:?}");
        }
    }

    /// The A.9 rows at p = 1 and VTK's own vertex coordinates: counter-clockwise round the
    /// quadrilateral and each hexahedron face of constant z.
    #[test]
    fn convert_linear_cells_follow_vtk_vertex_order() {
        let mesh = reference_cell(CellType::Hexahedron, 1);
        let (file, _) = to_vtu(&mesh, true, false).unwrap();
        assert_eq!(file.connectivity, [0, 1, 3, 2, 4, 5, 7, 6]);
        let x = f64_points(&file);
        let vtk_hex = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        for (k, &i) in file.connectivity.iter().enumerate() {
            let i = i as usize;
            assert_eq!(x[3 * i..3 * i + 3], vtk_hex[k], "VTK node {k}");
        }
        for (cell, row) in [
            (CellType::Point, &[0][..]),
            (CellType::Interval, &[0, 1]),
            (CellType::Triangle, &[0, 1, 2]),
            (CellType::Quadrilateral, &[0, 1, 3, 2]),
            (CellType::Tetrahedron, &[0, 1, 2, 3]),
            (CellType::Prism, &[0, 1, 2, 3, 4, 5]),
            (CellType::Pyramid, &[0, 1, 3, 2, 4]),
        ] {
            let (file, _) = to_vtu(&reference_cell(cell, 1), true, false).unwrap();
            assert_eq!(file.connectivity, row, "{cell:?}");
            assert_eq!(file.types, [vtk_cell_type(cell, 1, true)], "{cell:?}");
        }
    }

    /// Two cells of each Lagrange type at p = 3 whose node lists are not the identity: each
    /// cell's VTK nodes are its own nodes in `vtk_to_internal` order.
    #[test]
    fn convert_lagrange_connectivity_indexes_each_row() {
        for cell in CELLS.into_iter().filter(|&c| c != CellType::Pyramid) {
            let k = num_lagrange_nodes(cell, 3);
            let n = 2 * k as u32;
            // Cell 0 takes the nodes in reverse, cell 1 the odd ones then the even ones.
            let nodes: Vec<u32> = (0..k as u32)
                .rev()
                .chain((1..n).step_by(2))
                .chain((0..n).step_by(2))
                .take(2 * k)
                .collect();
            let mesh = VtkMesh {
                gdim: 3,
                nodes: vec![0.0; 3 * n as usize],
                blocks: vec![VtkCellBlock {
                    cell,
                    degree: 3,
                    nodes: nodes.clone(),
                }],
            };
            let (file, _) = to_vtu(&mesh, true, false).unwrap();
            let map = gridcl_refcell::vtk_to_internal(cell, 3).unwrap();
            let expected: Vec<i64> = nodes
                .chunks_exact(k)
                .flat_map(|row| map.iter().map(|&m| i64::from(row[m])))
                .collect();
            assert_eq!(file.connectivity, expected, "{cell:?}");
            assert_eq!(file.offsets, [k as i64, 2 * k as i64], "{cell:?}");
        }
    }

    /// Blocks are written in order, each with its own type and node count.
    #[test]
    fn convert_mixed_blocks() {
        // A P2 triangle (nodes 0-5) and a linear quadrilateral (nodes 6-9), in 2D.
        let mesh = VtkMesh {
            gdim: 2,
            nodes: (0..20).map(f64::from).collect(),
            blocks: vec![
                VtkCellBlock {
                    cell: CellType::Triangle,
                    degree: 2,
                    nodes: vec![0, 1, 2, 3, 4, 5],
                },
                VtkCellBlock {
                    cell: CellType::Quadrilateral,
                    degree: 1,
                    nodes: vec![6, 7, 8, 9, 9, 8, 7, 6],
                },
                VtkCellBlock {
                    cell: CellType::Hexahedron,
                    degree: 4,
                    nodes: vec![],
                },
            ],
        };
        let (file, pyramids) = to_vtu(&mesh, true, false).unwrap();
        assert_eq!(
            file.connectivity,
            [0, 1, 2, 3, 5, 4, 6, 7, 9, 8, 9, 8, 6, 7]
        );
        assert_eq!(file.offsets, [6, 10, 14]);
        assert_eq!(file.types, [69, 9, 9]);
        assert_eq!(pyramids, 0);
        // Points: x, y, 0 for each of the ten nodes.
        let x = f64_points(&file);
        assert_eq!(x.len(), 30);
        assert_eq!(x[..6], [0.0, 1.0, 0.0, 2.0, 3.0, 0.0]);
        assert_eq!(x[27..], [18.0, 19.0, 0.0]);

        let (file, _) = to_vtu(&mesh, false, false).unwrap();
        assert_eq!(file.connectivity, [0, 1, 2, 6, 7, 9, 8, 9, 8, 6, 7]);
        assert_eq!(file.offsets, [3, 7, 11]);
        assert_eq!(file.types, [5, 9, 9]);
    }

    /// With `lagrange == false` every cell is written from its corners, in VTK's vertex order.
    #[test]
    fn convert_linear_output_picks_corners() {
        for cell in CELLS {
            for p in 1..=3 {
                let k = num_lagrange_nodes(cell, p);
                let nodes: Vec<u32> = (0..k as u32).map(|i| 100 + i).collect();
                let mesh = VtkMesh {
                    gdim: 3,
                    nodes: vec![1.5; 3 * (100 + k)],
                    blocks: vec![VtkCellBlock {
                        cell,
                        degree: p,
                        nodes,
                    }],
                };
                let (file, pyramids) = to_vtu(&mesh, false, false).unwrap();
                let corners: Vec<i64> = gridcl_refcell::vtk_to_internal(cell, 1)
                    .unwrap()
                    .iter()
                    .map(|&m| 100 + m as i64)
                    .collect();
                assert_eq!(corners.len(), vertices(cell).len());
                assert_eq!(file.connectivity, corners, "{cell:?} p = {p}");
                assert_eq!(
                    file.types,
                    [vtk_cell_type(cell, 1, true)],
                    "{cell:?} p = {p}"
                );
                assert_eq!(pyramids, 0, "{cell:?} p = {p}");
                // Every node is a point, referenced or not.
                assert_eq!(f64_points(&file).len(), 3 * (100 + k));
            }
        }
    }

    /// Pyramids of degree ≥ 2 are written as type 14 from their vertices and counted, but only
    /// with `lagrange`.
    #[test]
    fn convert_pyramid_fallback() {
        for p in 2..=4 {
            let k = num_lagrange_nodes(CellType::Pyramid, p);
            let mesh = VtkMesh {
                gdim: 3,
                nodes: vec![0.0; 3 * 3 * k],
                blocks: vec![
                    VtkCellBlock {
                        cell: CellType::Tetrahedron,
                        degree: 1,
                        nodes: vec![0, 1, 2, 3],
                    },
                    VtkCellBlock {
                        cell: CellType::Pyramid,
                        degree: p,
                        nodes: (0..3 * k as u32).collect(),
                    },
                ],
            };
            for lagrange in [true, false] {
                let (file, pyramids) = to_vtu(&mesh, lagrange, false).unwrap();
                assert_eq!(file.types, [10, 14, 14, 14], "p = {p}");
                let mut expected = vec![0, 1, 2, 3];
                for c in 0..3 {
                    let base = (c * k) as i64;
                    expected.extend([0, 1, 3, 2, 4].map(|m| base + m));
                }
                assert_eq!(file.connectivity, expected, "p = {p}");
                assert_eq!(file.offsets, [4, 9, 14, 19], "p = {p}");
                assert_eq!(pyramids, if lagrange { 3 } else { 0 }, "p = {p}");
            }
        }
        // Linear pyramids are not counted.
        let (_, pyramids) = to_vtu(&reference_cell(CellType::Pyramid, 1), true, false).unwrap();
        assert_eq!(pyramids, 0);
    }

    #[test]
    fn convert_points_pad_and_narrow() {
        let nodes = vec![0.1, -2.0, 1e300, 3.0];
        for (gdim, padded) in [
            (
                1,
                vec![
                    0.1, 0.0, 0.0, -2.0, 0.0, 0.0, 1e300, 0.0, 0.0, 3.0, 0.0, 0.0,
                ],
            ),
            (2, vec![0.1, -2.0, 0.0, 1e300, 3.0, 0.0]),
        ] {
            let mesh = VtkMesh {
                gdim,
                nodes: nodes.clone(),
                blocks: vec![],
            };
            let (file, _) = to_vtu(&mesh, true, false).unwrap();
            assert_eq!(file.points, DataValues::F64(padded.clone()));
            assert!(file.connectivity.is_empty() && file.offsets.is_empty());
            assert!(file.types.is_empty());
            let (file, _) = to_vtu(&mesh, true, true).unwrap();
            let narrowed: Vec<f32> = padded.iter().map(|&x| x as f32).collect();
            assert_eq!(file.points, DataValues::F32(narrowed));
        }
        let (file, _) = to_vtu(
            &VtkMesh {
                gdim: 3,
                nodes: vec![],
                blocks: vec![],
            },
            true,
            true,
        )
        .unwrap();
        assert_eq!(file.points, DataValues::F32(vec![]));
    }

    /// Malformed meshes give errors, not panics.
    #[test]
    fn convert_rejects_malformed_meshes() {
        let good = VtkMesh {
            gdim: 2,
            nodes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 0.5, 0.5],
            blocks: vec![VtkCellBlock {
                cell: CellType::Triangle,
                degree: 2,
                nodes: vec![0, 1, 2, 3, 4, 5],
            }],
        };
        assert!(to_vtu(&good, true, false).is_ok());
        type Edit = (&'static str, fn(&mut VtkMesh), fn(&GridError) -> bool);
        let edits: [Edit; 8] = [
            (
                "gdim 0",
                |m| m.gdim = 0,
                |e| {
                    matches!(
                        e,
                        GridError::DimensionMismatch {
                            expected: (2, 2),
                            found: (2, 0)
                        }
                    )
                },
            ),
            (
                "gdim 4",
                |m| m.gdim = 4,
                |e| {
                    matches!(
                        e,
                        GridError::DimensionMismatch {
                            expected: (2, 3),
                            found: (2, 4)
                        }
                    )
                },
            ),
            (
                "node coordinates",
                |m| m.nodes.push(1.0),
                |e| {
                    matches!(
                        e,
                        GridError::LengthMismatch {
                            expected: 12,
                            found: 13
                        }
                    )
                },
            ),
            (
                "cell nodes",
                |m| m.blocks[0].nodes.push(0),
                |e| {
                    matches!(
                        e,
                        GridError::LengthMismatch {
                            expected: 6,
                            found: 7
                        }
                    )
                },
            ),
            (
                "degree",
                |m| m.blocks[0].degree = 3,
                |e| {
                    matches!(
                        e,
                        GridError::LengthMismatch {
                            expected: 0,
                            found: 6
                        }
                    )
                },
            ),
            (
                "cell node",
                |m| m.blocks[0].nodes[5] = 6,
                |e| matches!(e, GridError::IndexOutOfRange { index: 6, len: 6 }),
            ),
            (
                "degree 0",
                |m| m.blocks[0].degree = 0,
                |e| {
                    matches!(
                        e,
                        GridError::UnsupportedDegree {
                            cell: "Triangle",
                            degree: 0
                        }
                    )
                },
            ),
            (
                "huge degree",
                |m| m.blocks[0].degree = u32::MAX,
                |e| matches!(e, GridError::TooLarge),
            ),
        ];
        for (what, edit, expected) in edits {
            let mut bad = good.clone();
            edit(&mut bad);
            for lagrange in [true, false] {
                match to_vtu(&bad, lagrange, false) {
                    Err(e) if expected(&e) => {}
                    r => panic!("{what}: {:?}", r.map(|_| ())),
                }
            }
        }
        // An empty block is checked too.
        let mut bad = good.clone();
        bad.blocks.push(VtkCellBlock {
            cell: CellType::Prism,
            degree: 0,
            nodes: vec![],
        });
        assert!(matches!(
            to_vtu(&bad, true, false),
            Err(GridError::UnsupportedDegree { cell: "Prism", .. })
        ));
    }

    /// A random block: cell, degree 1-3, and node indices (possibly out of range).
    fn block(num_nodes: u32) -> impl Strategy<Value = VtkCellBlock> {
        (0..CELLS.len(), 1u32..=3, 0usize..4).prop_flat_map(move |(c, p, cells)| {
            let cell = CELLS[c];
            let k = num_lagrange_nodes(cell, p);
            proptest::collection::vec(0..num_nodes, cells * k).prop_map(move |nodes| VtkCellBlock {
                cell,
                degree: p,
                nodes,
            })
        })
    }

    fn mesh() -> impl Strategy<Value = VtkMesh> {
        (1usize..=3, 1u32..40).prop_flat_map(|(gdim, n)| {
            (
                proptest::collection::vec(-1e3f64..1e3, gdim * n as usize),
                proptest::collection::vec(block(n), 0..4),
            )
                .prop_map(move |(nodes, blocks)| VtkMesh {
                    gdim,
                    nodes,
                    blocks,
                })
        })
    }

    proptest! {
        /// On random small meshes: one type and offset per cell, offsets increasing to the end
        /// of the connectivity, every cell's points those of its row (all nodes, or the corners
        /// if linear), padded points and the pyramid count.
        #[test]
        fn convert_random_meshes(mesh in mesh(), lagrange: bool, f32_points: bool) {
            let (file, pyramids) = to_vtu(&mesh, lagrange, f32_points).unwrap();
            let mut cell = 0;
            let mut start = 0usize;
            let mut expected_pyramids = 0;
            for b in &mesh.blocks {
                let k = num_lagrange_nodes(b.cell, b.degree);
                let linear = !lagrange || b.degree == 1 || b.cell == CellType::Pyramid;
                for row in b.nodes.chunks_exact(k) {
                    let end = file.offsets[cell] as usize;
                    let mut written: Vec<i64> = file.connectivity[start..end].to_vec();
                    let used = if linear { &row[..vertices(b.cell).len()] } else { row };
                    let mut expected: Vec<i64> = used.iter().map(|&i| i64::from(i)).collect();
                    written.sort_unstable();
                    expected.sort_unstable();
                    prop_assert_eq!(written, expected);
                    prop_assert_eq!(file.types[cell], vtk_cell_type(b.cell, b.degree, lagrange));
                    if lagrange && b.cell == CellType::Pyramid && b.degree >= 2 {
                        expected_pyramids += 1;
                    }
                    start = end;
                    cell += 1;
                }
            }
            prop_assert_eq!(file.offsets.len(), cell);
            prop_assert_eq!(file.types.len(), cell);
            prop_assert_eq!(start, file.connectivity.len());
            prop_assert_eq!(pyramids, expected_pyramids);
            let points: Vec<f64> = match &file.points {
                DataValues::F32(v) if f32_points => v.iter().map(|&x| f64::from(x)).collect(),
                DataValues::F64(v) if !f32_points => v.clone(),
                other => panic!("points {other:?}"),
            };
            let gd = mesh.gdim;
            prop_assert_eq!(points.len(), 3 * (mesh.nodes.len() / gd));
            for (x, p) in mesh.nodes.chunks_exact(gd).zip(points.as_chunks::<3>().0) {
                for d in 0..3 {
                    let want = if d < gd { x[d] } else { 0.0 };
                    let want = if f32_points { f64::from(want as f32) } else { want };
                    prop_assert_eq!(p[d].to_bits(), want.to_bits());
                }
            }
            prop_assert!(file.point_data.is_empty() && file.cell_data.is_empty());
        }

        /// Random edits of node indices and lengths never panic.
        #[test]
        fn convert_malformed_never_panics(
            mut mesh in mesh(),
            extra_coord: bool,
            bump in proptest::option::of((0usize..4, 0usize..64, 0u32..100)),
            lagrange: bool,
        ) {
            if extra_coord {
                mesh.nodes.push(0.0);
            }
            if let Some((b, i, value)) = bump
                && let Some(block) = mesh.blocks.get_mut(b)
            {
                if let Some(slot) = block.nodes.get_mut(i) {
                    *slot = value;
                } else {
                    block.nodes.push(value);
                }
            }
            let _ = to_vtu(&mesh, lagrange, false);
        }
    }
}
