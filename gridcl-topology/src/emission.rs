//! Entity creation, step 1: the emission plan (host; ADR-0012). Block b emits sub-entity i of
//! cell c at e_b + i · n̂_b + c, with n̂_b the cell count rounded up to `VECTOR_PAD`, so the
//! lanes of a vector are aligned consecutive cells. Padding positions (c ≥ n_b) hold sentinel
//! keys. The padded total must stay below 2^32, because payloads are u32 emission indices.

use gridcl_device::{GridError, Result, VECTOR_PAD};
use gridcl_dims::CellType;
use gridcl_refcell::{sub_entities, sub_entity_types};

/// The emission range of one cell block for one entity type.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EmissionRange {
    /// The cell block.
    pub block: usize,
    /// e_b, the first emission index; a multiple of `VECTOR_PAD`.
    pub start: usize,
    /// n_b, the cells of the block.
    pub cells: usize,
    /// n̂_b, the cell count rounded up to `VECTOR_PAD`.
    pub padded: usize,
    /// The cell's local sub-entities of this type, in local order: sub-entity `local[i]` of
    /// cell c is emitted at start + i · padded + c.
    pub local: Vec<usize>,
}

impl EmissionRange {
    /// One past the last emission index.
    pub fn end(&self) -> usize {
        self.start + self.local.len() * self.padded
    }
}

/// The emission plan of one entity type of one dimension.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct EmissionPlan {
    /// The entity type.
    pub entity: CellType,
    /// One range per cell block whose cells have entities of this type, in block order.
    pub ranges: Vec<EmissionRange>,
    /// The padded total of emitted keys.
    pub total: usize,
}

impl EmissionPlan {
    /// Whether some position holds a sentinel key.
    pub fn has_sentinels(&self) -> bool {
        self.ranges.iter().any(|r| r.cells < r.padded)
    }
}

/// The emission plans of dimension `dim` for cell blocks of the given types and cell counts:
/// one per entity type, in `CellType` order.
///
/// # Errors
///
/// `TooLarge` if a padded total reaches 2^32 or overflows.
pub(crate) fn emission_plans(
    blocks: &[(CellType, usize)],
    dim: usize,
) -> Result<Vec<EmissionPlan>> {
    let mut types: Vec<CellType> = blocks
        .iter()
        .flat_map(|&(cell, _)| sub_entity_types(cell, dim).iter().copied())
        .collect();
    types.sort_unstable();
    types.dedup();
    types
        .into_iter()
        .map(|entity| {
            let mut ranges = Vec::new();
            let mut start = 0usize;
            for (block, &(cell, cells)) in blocks.iter().enumerate() {
                let local: Vec<usize> = sub_entity_types(cell, dim)
                    .iter()
                    .enumerate()
                    .filter(|&(_, &t)| t == entity)
                    .map(|(i, _)| i)
                    .collect();
                if local.is_empty() {
                    continue;
                }
                let padded = cells
                    .checked_next_multiple_of(VECTOR_PAD)
                    .ok_or(GridError::TooLarge)?;
                let range = EmissionRange {
                    block,
                    start,
                    cells,
                    padded,
                    local,
                };
                start = range
                    .local
                    .len()
                    .checked_mul(padded)
                    .and_then(|n| n.checked_add(start))
                    .ok_or(GridError::TooLarge)?;
                ranges.push(range);
            }
            if start > u32::MAX as usize {
                return Err(GridError::TooLarge);
            }
            Ok(EmissionPlan {
                entity,
                ranges,
                total: start,
            })
        })
        .collect()
}

/// The local vertex lists of the sub-entities `local` of `cell` of dimension `dim`.
pub(crate) fn sub_entity_vertices(cell: CellType, dim: usize, local: &[usize]) -> Vec<Vec<usize>> {
    local
        .iter()
        .map(|&i| sub_entities(cell, dim)[i].to_vec())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use CellType::*;

    #[test]
    fn prism_faces_split_by_type() {
        let plans = emission_plans(&[(Prism, 20), (Tetrahedron, 3)], 2).unwrap();
        assert_eq!(plans.len(), 2);
        let tri = &plans[0];
        assert_eq!(tri.entity, Triangle);
        // Prism faces 0 and 4 are triangles; 20 cells pad to 32.
        assert_eq!(tri.ranges[0].local, [0, 4]);
        assert_eq!((tri.ranges[0].start, tri.ranges[0].padded), (0, 32));
        assert_eq!(tri.ranges[1].local, [0, 1, 2, 3]);
        assert_eq!((tri.ranges[1].start, tri.ranges[1].padded), (64, 16));
        assert_eq!(tri.total, 64 + 64);
        assert!(tri.has_sentinels());
        let quad = &plans[1];
        assert_eq!(quad.entity, Quadrilateral);
        assert_eq!(quad.ranges.len(), 1);
        assert_eq!(quad.ranges[0].local, [1, 2, 3]);
        assert_eq!(quad.total, 3 * 32);
        for plan in &plans {
            for r in &plan.ranges {
                assert_eq!(r.start % VECTOR_PAD, 0);
                assert_eq!(r.padded % VECTOR_PAD, 0);
            }
        }
    }

    #[test]
    fn exact_multiples_have_no_sentinels() {
        let plans = emission_plans(&[(Triangle, 32), (Quadrilateral, 16)], 1).unwrap();
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[0].total, 3 * 32 + 4 * 16);
        assert!(!plans[0].has_sentinels());
    }

    #[test]
    fn empty_blocks_emit_nothing() {
        let plans = emission_plans(&[(Hexahedron, 0)], 1).unwrap();
        assert_eq!(plans[0].total, 0);
        assert_eq!(plans[0].ranges[0].padded, 0);
    }

    #[test]
    fn totals_at_or_above_2_pow_32_are_too_large() {
        // 12 edges per hexahedron: 12 · 357_913_942 = 4_294_967_304 > u32::MAX after padding.
        let n = 357_913_942;
        assert!(matches!(
            emission_plans(&[(Hexahedron, n)], 1),
            Err(GridError::TooLarge)
        ));
        // The largest padded total that fits.
        let n = (u32::MAX as usize / 12) / VECTOR_PAD * VECTOR_PAD;
        assert!(emission_plans(&[(Hexahedron, n)], 1).is_ok());
        assert!(matches!(
            emission_plans(&[(Hexahedron, usize::MAX - 3)], 1),
            Err(GridError::TooLarge)
        ));
    }
}
