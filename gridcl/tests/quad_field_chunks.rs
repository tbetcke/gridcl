//! M3 acceptance: with an injected limit, `QuadField` results are identical with 1 and 7
//! chunks. The mesh and its E-vector are built inside the `with_properties` scope, so the
//! E-vector is chunked by the same limit.

mod common;

use common::*;
use cubecl::prelude::Client;
use gridcl::prelude::*;
use gridcl_device::testing::with_properties;

const N: usize = 22; // 968 triangles
const DEGREE: u32 = 4;

fn jacobians<F: Real>(client: &Client) -> (Vec<F>, usize, usize, usize) {
    let mesh = build::<F, D2, D2>(&BoxKind::Triangle.host(N, 1).perturbed(0.05), client);
    let rule = QuadRule::for_grid(&mesh, DEGREE).expect("rule");
    let j = mesh.jacobians(&rule).expect("jacobians");
    let field = &j.blocks[0];
    (
        flat(field),
        field.chunks().len(),
        field.tile(),
        field.num_points(),
    )
}

fn check<F: Real>(client: &Client) {
    let (one, chunks, tile, num_points) = jacobians::<F>(client);
    assert_eq!(chunks, 1, "without a limit the field is one chunk");
    let num_cells = 2 * N * N;
    // The smallest chunk of whole tiles that needs 7 chunks, and a limit that fits it exactly.
    let chunk_cells = num_cells.div_ceil(7).next_multiple_of(tile);
    assert_eq!(num_cells.div_ceil(chunk_cells), 7);
    let len = 4; // JacShape<D2, D2>
    assert!(
        num_points * len >= PAD,
        "the limit below must not admit another cell"
    );
    let elems = (chunk_cells * num_points * len).next_multiple_of(PAD);
    let mut props = client.properties().clone();
    props.memory.max_page_size = (elems * size_of::<F>()) as u64;
    let (seven, chunks, _, _) = with_properties(props, || jacobians::<F>(client));
    assert_eq!(chunks, 7, "{}: chunks with the injected limit", name::<F>());
    assert_bits(&seven, &one, "1 vs 7 chunks");
}

#[test]
fn quad_field_identical_with_1_and_7_chunks() {
    for_each_precision!(client(), check);
}
