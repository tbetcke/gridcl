//! M1b acceptance: orientation bits equal `compute_orientation` for 10 000 random vertex
//! numberings per cell type with D_TOP ≥ 2 (A.6).

use gridcl_conformance::{CELLS, reference_cell_type};
use gridcl_refcell::{orientation_bits, tdim, vertices};
use ndelement::orientation::compute_orientation;

const NUMBERINGS: usize = 10_000;

/// splitmix64 (Steele, Lea and Flood 2014), so the test needs no random-number crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
}

#[test]
fn bits_match_compute_orientation() {
    let mut rng = SplitMix64(0x6772_6964_636c_4d31); // "gridclM1"
    for cell in CELLS.into_iter().filter(|&c| tdim(c) >= 2) {
        let nv = vertices(cell).len();
        for _ in 0..NUMBERINGS {
            // Distinct non-negative ids below 2^62.
            let mut ids: Vec<u64> = Vec::with_capacity(nv);
            while ids.len() < nv {
                let id = rng.next() >> 2;
                if !ids.contains(&id) {
                    ids.push(id);
                }
            }
            let ours: Vec<i64> = ids.iter().map(|&i| i64::try_from(i).unwrap()).collect();
            let theirs: Vec<usize> = ids.iter().map(|&i| usize::try_from(i).unwrap()).collect();
            let want = compute_orientation(reference_cell_type(cell), &theirs) as u32;
            assert_eq!(
                orientation_bits(cell, &ours),
                want,
                "{cell:?} with origin ids {ids:?}"
            );
        }
    }
}
