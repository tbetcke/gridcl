//! M1b acceptance: sub-entity tables equal ndelement's for all eight cells (A.1–A.3).

use gridcl_conformance::{CELLS, reference_cell_type};
use gridcl_refcell::{num_sub_entities, sub_entities, sub_entity_types, tdim, vertices};
use ndelement::reference_cell;

#[test]
fn sub_entities_match_ndelement() {
    for cell in CELLS {
        let nd = reference_cell_type(cell);
        assert_eq!(tdim(cell), reference_cell::dim(nd), "tdim of {cell:?}");

        // ndelement gives the point cell no vertex coordinates; gridcl gives it one vertex with
        // none (A.1). Both flatten to the same coordinates.
        let ours: Vec<f64> = vertices(cell)
            .iter()
            .flat_map(|v| v.iter().copied())
            .collect();
        let theirs: Vec<f64> = reference_cell::vertices::<f64>(nd)
            .into_iter()
            .flatten()
            .collect();
        assert_eq!(ours, theirs, "vertices of {cell:?}");
        assert_eq!(
            vertices(cell).len(),
            reference_cell::entity_counts(nd)[0],
            "{cell:?}"
        );

        let types = reference_cell::entity_types(nd);
        for (dim, types) in types.iter().enumerate() {
            let theirs: Vec<Vec<usize>> = match dim {
                0 => (0..reference_cell::entity_counts(nd)[0])
                    .map(|v| vec![v])
                    .collect(),
                1 => reference_cell::edges(nd),
                2 => reference_cell::faces(nd),
                _ => reference_cell::volumes(nd),
            };
            let ours: Vec<Vec<usize>> =
                sub_entities(cell, dim).iter().map(|e| e.to_vec()).collect();
            assert_eq!(ours, theirs, "sub-entities of dimension {dim} of {cell:?}");
            assert_eq!(
                num_sub_entities(cell, dim),
                theirs.len(),
                "{cell:?}, dimension {dim}"
            );

            let ours: Vec<_> = sub_entity_types(cell, dim)
                .iter()
                .map(|&t| reference_cell_type(t))
                .collect();
            assert_eq!(
                &ours, types,
                "sub-entity types of dimension {dim} of {cell:?}"
            );
        }
    }
}
