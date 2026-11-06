//! Closure node maps (ADR-0039, item 8; docs/plans/M9.md, finding 9): the parent cell node at
//! each node of a sub-entity, in the sub-entity's reference order.
//!
//! The child's Lagrange points of the same degree and family, mapped into the parent cell by
//! `facet_points_to_cell` (A.5's affine rule over the sub-entity's vertex list), are matched
//! against the parent's Lagrange points. Both sides use the same one-dimensional point sets,
//! so the match is exact up to rounding.

use gridcl_device::{GridError, Result};
use gridcl_dims::CellType;
use gridcl_geometry::NodeFamily;

use crate::builder::cell_name;

/// The distance below which two reference points are the same node.
const MATCH_TOL: f64 = 1e-10;

/// The name of `family` in errors.
fn family_name(family: NodeFamily) -> &'static str {
    match family {
        NodeFamily::Equispaced => "equispaced",
        NodeFamily::Gll => "GLL",
    }
}

/// The parent node of each node of facet `facet` of `parent` (degree `degree`, family
/// `family`), in the facet's reference node order.
///
/// # Errors
///
/// `UnsupportedDegree` or `UnsupportedNodeFamily` if the facet type has no such element.
///
/// # Panics
///
/// If a facet node has no parent node, which would be a bug in the reference tables.
pub(crate) fn facet_closure(
    parent: CellType,
    facet: usize,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<u32>> {
    let tdim = gridcl_refcell::tdim(parent);
    let child = gridcl_refcell::sub_entity_types(parent, tdim - 1)[facet];
    let child_points = gridcl_refcell::lagrange_points(child, degree, family).map_err(|_| {
        if degree == 0 {
            GridError::UnsupportedDegree {
                cell: cell_name(child),
                degree,
            }
        } else {
            GridError::UnsupportedNodeFamily {
                cell: cell_name(child),
                family: family_name(family),
            }
        }
    })?;
    let parent_points = gridcl_refcell::lagrange_points(parent, degree, family).map_err(|_| {
        GridError::UnsupportedNodeFamily {
            cell: cell_name(parent),
            family: family_name(family),
        }
    })?;
    let mapped = gridcl_refcell::facet_points_to_cell(parent, facet, &child_points);
    Ok(match_points(
        parent,
        &parent_points,
        &mapped,
        "facet",
        facet,
    ))
}

/// The parent node at each of the reference points `mapped` (tdim coordinates each) of
/// `parent`, whose Lagrange points are `parent_points`; `what` and `index` name the
/// sub-entity in the panic.
///
/// # Panics
///
/// If a point has no parent node.
fn match_points(
    parent: CellType,
    parent_points: &[f64],
    mapped: &[f64],
    what: &str,
    index: usize,
) -> Vec<u32> {
    let tdim = gridcl_refcell::tdim(parent);
    mapped
        .chunks_exact(tdim)
        .map(|x| {
            parent_points
                .chunks_exact(tdim)
                .position(|y| {
                    x.iter()
                        .zip(y)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                        < MATCH_TOL
                })
                .unwrap_or_else(|| panic!("{parent:?} {what} {index}: no parent node at {x:?}"))
                as u32
        })
        .collect()
}

/// The parent node of each node of edge `edge` of `parent` (degree `degree`, family `family`),
/// in the reference order of the interval from the edge's first local vertex to its second, or
/// from the second to the first if `reversed`.
///
/// # Errors
///
/// `UnsupportedDegree` for degree 0; `UnsupportedNodeFamily` if `parent` has no such element.
///
/// # Panics
///
/// If an edge node has no parent node, which would be a bug in the reference tables.
pub(crate) fn edge_closure(
    parent: CellType,
    edge: usize,
    reversed: bool,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<u32>> {
    let points =
        gridcl_refcell::lagrange_points(CellType::Interval, degree, family).map_err(|_| {
            GridError::UnsupportedDegree {
                cell: cell_name(CellType::Interval),
                degree,
            }
        })?;
    let parent_points = gridcl_refcell::lagrange_points(parent, degree, family).map_err(|_| {
        GridError::UnsupportedNodeFamily {
            cell: cell_name(parent),
            family: family_name(family),
        }
    })?;
    let refs = gridcl_refcell::vertices(parent);
    let verts = &gridcl_refcell::sub_entities(parent, 1)[edge];
    let (a, b) = if reversed {
        (&refs[verts[1]], &refs[verts[0]])
    } else {
        (&refs[verts[0]], &refs[verts[1]])
    };
    let mapped: Vec<f64> = points
        .iter()
        .flat_map(|&t| a.iter().zip(b.iter()).map(move |(a, b)| a + t * (b - a)))
        .collect();
    Ok(match_points(parent, &parent_points, &mapped, "edge", edge))
}

/// The edge closure maps of `parent`, k entries per code d · ne + e (direction d, 1 for
/// reversed; local edge e of ne), the layout `restrict_rows` reads.
///
/// # Errors
///
/// As [`edge_closure`].
pub(crate) fn edge_table(parent: CellType, degree: u32, family: NodeFamily) -> Result<Vec<u32>> {
    let ne = gridcl_refcell::num_sub_entities(parent, 1);
    let mut table = Vec::new();
    for reversed in [false, true] {
        for e in 0..ne {
            table.extend(edge_closure(parent, e, reversed, degree, family)?);
        }
    }
    Ok(table)
}

/// The closure maps of every facet of `parent` whose type is `child`, `k` entries each, in
/// facet order; facets of another type hold `k` zeros. The layout of the vertex tables of the
/// facet rows, so one kernel serves both.
///
/// # Errors
///
/// As [`facet_closure`].
pub(crate) fn closure_table(
    parent: CellType,
    child: CellType,
    degree: u32,
    family: NodeFamily,
) -> Result<Vec<u32>> {
    let tdim = gridcl_refcell::tdim(parent);
    let k = gridcl_refcell::num_lagrange_nodes(child, degree);
    let mut table = Vec::new();
    for (f, &t) in gridcl_refcell::sub_entity_types(parent, tdim - 1)
        .iter()
        .enumerate()
    {
        if t == child {
            table.extend(facet_closure(parent, f, degree, family)?);
        } else {
            table.extend(std::iter::repeat_n(0u32, k));
        }
    }
    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The closure maps of degree ≤ 4 (pyramids ≤ 3) are permutations of distinct parent
    /// nodes whose first entries are the facet's vertices, and each facet node lies on the
    /// facet: the parent's entity nodes of the facet's closure.
    #[test]
    fn closure_maps_are_facet_nodes() {
        use CellType::*;
        for parent in [
            Triangle,
            Quadrilateral,
            Tetrahedron,
            Hexahedron,
            Prism,
            Pyramid,
        ] {
            let max = if parent == Pyramid { 3 } else { 4 };
            let tdim = gridcl_refcell::tdim(parent);
            for degree in 1..=max {
                let families: &[NodeFamily] = match parent {
                    Quadrilateral | Hexahedron => &[NodeFamily::Equispaced, NodeFamily::Gll],
                    _ => &[NodeFamily::Equispaced],
                };
                for &family in families {
                    for (f, verts) in gridcl_refcell::sub_entities(parent, tdim - 1)
                        .iter()
                        .enumerate()
                    {
                        let map = facet_closure(parent, f, degree, family).expect("closure");
                        let child = gridcl_refcell::sub_entity_types(parent, tdim - 1)[f];
                        assert_eq!(map.len(), gridcl_refcell::num_lagrange_nodes(child, degree));
                        let v: Vec<u32> = verts.iter().map(|&v| v as u32).collect();
                        assert_eq!(&map[..v.len()], &v[..], "{parent:?} p={degree} facet {f}");
                        let mut sorted = map.clone();
                        sorted.sort_unstable();
                        sorted.dedup();
                        assert_eq!(sorted.len(), map.len(), "{parent:?} p={degree} facet {f}");
                        // Every node of the facet's own entities is in the map.
                        let own =
                            gridcl_refcell::lagrange_entity_nodes(parent, degree, tdim - 1, f);
                        assert!(own.into_iter().all(|n| map.contains(&(n as u32))));
                    }
                }
            }
        }
    }

    /// The edge closure maps of degree ≤ 4 (pyramids ≤ 3) start at the edge's vertices in the
    /// chosen direction, and the reversed map is the forward one read backwards along the
    /// edge: vertices swapped, interior nodes reversed.
    #[test]
    fn edge_closures_follow_the_direction() {
        use CellType::*;
        for parent in [Tetrahedron, Hexahedron, Prism, Pyramid] {
            let max = if parent == Pyramid { 3 } else { 4 };
            for degree in 1..=max {
                let families: &[NodeFamily] = match parent {
                    Hexahedron => &[NodeFamily::Equispaced, NodeFamily::Gll],
                    _ => &[NodeFamily::Equispaced],
                };
                for &family in families {
                    let table = edge_table(parent, degree, family).expect("edge table");
                    let ne = gridcl_refcell::num_sub_entities(parent, 1);
                    let k = degree as usize + 1;
                    assert_eq!(table.len(), 2 * ne * k);
                    for (e, verts) in gridcl_refcell::sub_entities(parent, 1).iter().enumerate() {
                        let fwd = &table[e * k..(e + 1) * k];
                        let rev = &table[(ne + e) * k..(ne + e + 1) * k];
                        assert_eq!(fwd[..2], [verts[0] as u32, verts[1] as u32]);
                        assert_eq!(rev[..2], [verts[1] as u32, verts[0] as u32]);
                        let back: Vec<u32> = fwd[2..].iter().rev().copied().collect();
                        assert_eq!(rev[2..], back[..], "{parent:?} p={degree} edge {e}");
                        let own = gridcl_refcell::lagrange_entity_nodes(parent, degree, 1, e);
                        assert!(own.into_iter().all(|n| fwd.contains(&(n as u32))));
                    }
                }
            }
        }
    }
}
