//! Orientation and permutation bits, A.6 (docs/design.md, Appendix A).

use gridcl_dims::CellType;

use crate::{sub_entities, sub_entity_types, tdim, vertices};

/// The permutation info of a cell (A.6): one bit per edge, then three bits per face, with
/// ndelement's layout (`compute_orientation`).
///
/// `vertex_ids` are the origin ids of the cell's vertices in local order; never local or
/// post-partition global ids. Only their order matters. From the least significant bit:
///
/// - one bit per edge in edge order, set when the edge's first listed vertex has the higher id;
/// - three bits per face in face order (cells of dimension 3 only): the low two hold the
///   rotation count, the cyclic position of the face's lowest vertex in the face cycle
///   (triangles: list order; quadrilaterals: tensor order, cycle 0–1–3–2); the third is the
///   reflection, set when that vertex's successor on the cycle has a higher id than its
///   predecessor.
///
/// Cells of dimension below 2 have no bits, and bits 30 and 31 are always zero.
///
/// # Example
///
/// The worked example of A.6: a hexahedron whose face 0 carries origin ids 17, 4, 9, 12 in its
/// tensor slots has rotation 1 and no reflection on that face (bits 12–14 are `0b001`).
///
/// ```
/// use gridcl_dims::CellType;
/// use gridcl_refcell::orientation_bits;
///
/// let ids = [17, 4, 9, 12, 100, 101, 102, 103];
/// let bits = orientation_bits(CellType::Hexahedron, &ids);
/// assert_eq!((bits >> 12) & 0b111, 0b001);
/// // Edge 0 runs from id 17 to id 4, against the reference orientation.
/// assert_eq!(bits & 1, 1);
/// ```
///
/// # Panics
///
/// If `vertex_ids.len()` is not the cell's vertex count.
pub fn orientation_bits(cell: CellType, vertex_ids: &[i64]) -> u32 {
    assert_eq!(
        vertex_ids.len(),
        vertices(cell).len(),
        "orientation_bits: {cell:?} has {} vertices, got {} ids",
        vertices(cell).len(),
        vertex_ids.len()
    );
    let td = tdim(cell);
    let mut bits = 0u32;
    let mut n = 0;
    if td >= 2 {
        for edge in sub_entities(cell, 1) {
            if vertex_ids[edge[0]] > vertex_ids[edge[1]] {
                bits |= 1 << n;
            }
            n += 1;
        }
    }
    if td == 3 {
        for (face, face_type) in sub_entities(cell, 2).iter().zip(sub_entity_types(cell, 2)) {
            // The face cycle as slots of the face's vertex list.
            let cycle: &[usize] = match face_type {
                CellType::Triangle => &[0, 1, 2],
                CellType::Quadrilateral => &[0, 1, 3, 2],
                _ => unreachable!("faces are triangles or quadrilaterals"),
            };
            let id = |position: usize| vertex_ids[face[cycle[position % cycle.len()]]];
            let len = cycle.len();
            let rotation = (0..len)
                .min_by_key(|&k| id(k))
                .expect("a face has vertices");
            let successor = id(rotation + 1);
            let predecessor = id(rotation + len - 1);
            bits |= (rotation as u32) << n;
            if successor > predecessor {
                bits |= 1 << (n + 2);
            }
            n += 3;
        }
    }
    bits
}

#[cfg(test)]
mod tests {
    use super::*;
    use CellType::*;
    use proptest::prelude::*;

    const CELLS: [CellType; 8] = [
        Point,
        Interval,
        Triangle,
        Quadrilateral,
        Tetrahedron,
        Hexahedron,
        Prism,
        Pyramid,
    ];

    /// A.6, "Bits used".
    fn bits_used(cell: CellType) -> u32 {
        match cell {
            Point | Interval => 0,
            Triangle => 3,
            Quadrilateral => 4,
            Tetrahedron => 18,
            Hexahedron => 30,
            Prism => 24,
            Pyramid => 23,
        }
    }

    /// All permutations of 0..n (in insertion order; the order does not matter here).
    fn permutations(n: usize) -> Vec<Vec<i64>> {
        if n == 0 {
            return vec![vec![]];
        }
        let mut out = vec![];
        for rest in permutations(n - 1) {
            for k in 0..n {
                let mut p = rest.clone();
                p.insert(k, (n - 1) as i64);
                out.push(p);
            }
        }
        out
    }

    #[test]
    fn worked_example_of_a6() {
        // Quadrilateral face 0 of a hexahedron with ids 17, 4, 9, 12 in its tensor slots.
        let bits = orientation_bits(Hexahedron, &[17, 4, 9, 12, 100, 101, 102, 103]);
        assert_eq!((bits >> 12) & 0b111, 0b001);
        // Triangle face 0 of a tetrahedron with ids (9, 30, 4): the lowest id sits at position
        // 2; its successor 9 is lower than its predecessor 30, so no reflection.
        let bits = orientation_bits(Tetrahedron, &[9, 30, 4, 50]);
        assert_eq!((bits >> 6) & 0b111, 0b010);
        // Edge with ids (30, 4): edge 0 of a triangle, reversed.
        assert_eq!(orientation_bits(Triangle, &[30, 4, 50]) & 1, 1);
    }

    #[test]
    fn no_bits_below_dimension_two() {
        assert_eq!(orientation_bits(Point, &[7]), 0);
        assert_eq!(orientation_bits(Interval, &[9, 3]), 0);
    }

    #[test]
    fn bits_stay_within_the_table_of_a6() {
        for cell in CELLS {
            let nv = vertices(cell).len();
            let mut seen = 0u32;
            for ids in permutations(nv) {
                seen |= orientation_bits(cell, &ids);
            }
            let used = bits_used(cell);
            // Every bit below the count is set by some numbering, and none above it.
            let mask = if used == 0 {
                0
            } else {
                u32::MAX >> (32 - used)
            };
            assert_eq!(seen, mask, "{cell:?}");
            assert_eq!(seen >> 30, 0, "{cell:?}");
        }
    }

    #[test]
    fn negative_ids() {
        let ids = [-5, -9_000_000_000, 3, i64::MIN, i64::MAX, 0];
        let ranks = [2, 1, 4, 0, 5, 3];
        assert_eq!(
            orientation_bits(Prism, &ids),
            orientation_bits(Prism, &ranks)
        );
    }

    #[test]
    #[should_panic(expected = "has 4 vertices")]
    fn wrong_vertex_count() {
        orientation_bits(Tetrahedron, &[0, 1, 2]);
    }

    /// Strategy: a cell with D_TOP ≥ 2 and two sets of distinct ids of its vertex count.
    fn cell_and_two_id_sets() -> impl Strategy<Value = (CellType, Vec<i64>, Vec<i64>)> {
        proptest::sample::select(&CELLS[2..]).prop_flat_map(|cell| {
            let nv = vertices(cell).len();
            let ids = proptest::collection::btree_set(any::<i64>(), nv)
                .prop_map(|s| s.into_iter().collect::<Vec<_>>());
            (Just(cell), ids.clone().prop_shuffle(), ids)
        })
    }

    proptest! {
        /// A strictly increasing relabelling of the ids keeps the word (A.6 depends only on
        /// the order of the origin ids).
        #[test]
        fn bits_depend_only_on_order((cell, ids, targets) in cell_and_two_id_sets()) {
            let mut sorted = ids.clone();
            sorted.sort();
            // targets is sorted ascending (from a BTreeSet): map the k-th smallest id to the
            // k-th smallest target.
            let relabelled: Vec<i64> = ids
                .iter()
                .map(|id| targets[sorted.binary_search(id).unwrap()])
                .collect();
            prop_assert_eq!(orientation_bits(cell, &ids), orientation_bits(cell, &relabelled));
        }
    }
}
