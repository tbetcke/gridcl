//! M2 acceptance: property tests of every primitive against CPU references, with sizes up to
//! 10^5 and 256 cases, for every `DeviceScalar` the device supports and keys of one to four
//! words. Edge cases of the index-driven primitives (out-of-range and duplicate indices) are
//! fixed tests at the end.

mod common;

use common::*;
use cubecl::prelude::*;
use gridcl_device::copy;
use gridcl_device::*;
use gridcl_testing::test_client;
use proptest::prelude::*;

fn check_fill<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let v = T::any(&mut rng);
    let mut x = upload(client, &any_vec::<T>(len, seed ^ 1));
    fill(&mut x, v).expect("fill");
    assert_bits(&x.to_host().expect("read"), &vec![v; len], "fill");
}

fn check_copy<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let src_host = any_vec::<T>(len, seed);
    let src = upload(client, &src_host);
    let mut dst = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
    copy(&src, &mut dst).expect("copy");
    assert_bits(&dst.to_host().expect("read"), &src_host, "copy");
    let clone = src.deep_clone().expect("deep_clone");
    assert_eq!(clone.len(), len);
    assert_bits(&clone.to_host().expect("read"), &src_host, "deep_clone");
}

/// Copy moves u32 words, so it serves a type the typed primitives do not have.
fn check_copy_u64(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let src_host: Vec<u64> = (0..len).map(|_| rng.next_u64()).collect();
    let src = upload(client, &src_host);
    let mut dst = DeviceBuffer::<u64>::uninit(client, len).expect("uninit");
    copy(&src, &mut dst).expect("copy u64");
    assert_eq!(dst.to_host().expect("read"), src_host, "copy u64");
}

fn check_reduce<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    for op in OPS {
        let x_host = reduce_input::<T>(len, seed, op);
        let x = upload(client, &x_host);
        let out = reduce(&x, op).expect("reduce");
        assert_eq!(out.len(), 1, "reduce returns one value");
        let got = out.to_host().expect("read")[0];
        let (want, scale) = reference_reduce(&x_host, op);
        assert_value(
            got,
            want,
            scale,
            matches!(op, ReduceOp::Sum),
            &format!("reduce {op:?} of {len}"),
        );
    }
}

fn check_segmented<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let max_segments = [0, 1, 7, 100, len.max(1)][rng.below(5) as usize];
    let offsets_host = random_offsets(len, max_segments, &mut rng);
    let offsets = upload(client, &offsets_host);
    for op in OPS {
        let values_host = reduce_input::<T>(len, seed ^ 7, op);
        let values = upload(client, &values_host);
        let out = segmented_reduce(&values, &offsets, op).expect("segmented_reduce");
        let want = reference_segmented(&values_host, &offsets_host, op);
        let got = out.to_host().expect("read");
        assert_eq!(got.len(), want.len(), "one value per segment");
        for (k, (&g, &(w, scale))) in got.iter().zip(&want).enumerate() {
            assert_value(
                g,
                w,
                scale,
                matches!(op, ReduceOp::Sum),
                &format!("segment {k} {op:?}"),
            );
        }
    }
}

fn check_inclusive_scan<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let x_host = bounded_vec::<T>(len, seed);
    let out = inclusive_scan(&upload(client, &x_host)).expect("inclusive_scan");
    let (want, scales) = reference_inclusive_scan(&x_host);
    assert_scan(
        &out.to_host().expect("read"),
        &want,
        &scales,
        "inclusive_scan",
    );
}

fn check_exclusive_scan<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let x_host = bounded_vec::<T>(len, seed);
    let out = exclusive_scan(&upload(client, &x_host)).expect("exclusive_scan");
    assert_eq!(out.len(), len + 1, "the exclusive scan has n + 1 elements");
    let (incl, incl_scales) = reference_inclusive_scan(&x_host);
    let mut want = vec![0.0];
    want.extend(incl);
    let mut scales = vec![0.0];
    scales.extend(incl_scales);
    assert_scan(
        &out.to_host().expect("read"),
        &want,
        &scales,
        "exclusive_scan",
    );
}

fn check_gather<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let src_len = if len == 0 {
        rng.below(3) as usize
    } else {
        1 + rng.below(len as u64 * 2) as usize
    };
    let src_host = any_vec::<T>(src_len, seed ^ 3);
    let idx_host: Vec<u32> = (0..len).map(|_| rng.below(src_len as u64) as u32).collect();
    let out = gather(&upload(client, &src_host), &upload(client, &idx_host)).expect("gather");
    let want: Vec<T> = idx_host.iter().map(|&i| src_host[i as usize]).collect();
    assert_bits(&out.to_host().expect("read"), &want, "gather");
}

/// `len` distinct indices into `0..dst_len`, `dst_len >= len`, in random order.
fn distinct_indices(len: usize, dst_len: usize, rng: &mut SplitMix) -> Vec<u32> {
    let mut perm: Vec<u32> = (0..dst_len as u32).collect();
    for i in (1..perm.len()).rev() {
        perm.swap(i, rng.below(i as u64 + 1) as usize);
    }
    perm.truncate(len);
    perm
}

fn check_scatter<T: TestScalar>(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let dst_len = len + rng.below(len as u64 + 2) as usize;
    let idx_host = distinct_indices(len, dst_len, &mut rng);
    let src_host = any_vec::<T>(len, seed ^ 5);
    let dst_host = any_vec::<T>(dst_len, seed ^ 9);
    let mut dst = upload(client, &dst_host);
    scatter(
        &upload(client, &src_host),
        &upload(client, &idx_host),
        &mut dst,
    )
    .expect("scatter");
    let mut want = dst_host;
    for (&i, &v) in idx_host.iter().zip(&src_host) {
        want[i as usize] = v;
    }
    assert_bits(&dst.to_host().expect("read"), &want, "scatter");
}

fn check_histogram(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    // 70 000 bins exceed shared memory, so the bins are processed in chunks.
    let num_bins = [0u32, 1, 2, 17, 256, 1000, 70_000][rng.below(7) as usize];
    let range = u64::from(num_bins) + 1 + u64::from(num_bins / 10);
    let values_host: Vec<u32> = (0..len)
        .map(|_| {
            if rng.below(50) == 0 {
                rng.next_u32()
            } else {
                rng.below(range) as u32
            }
        })
        .collect();
    let out = histogram(&upload(client, &values_host), num_bins).expect("histogram");
    let mut want = vec![0u32; num_bins as usize];
    for &v in &values_host {
        if let Some(c) = want.get_mut(v as usize) {
            *c += 1;
        }
    }
    assert_eq!(
        out.to_host().expect("read"),
        want,
        "histogram of {len} into {num_bins} bins"
    );
}

fn check_radix_sort(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let num_words = 1 + rng.below(4) as usize;
    // Small word ranges give many duplicates, which check stability through the payload.
    let dup_range = [0u32, 2, 3, 256, 1 << 16][rng.below(5) as usize];
    let keys_host = random_keys(len, num_words, dup_range, &mut rng);
    let payload_host: Vec<u32> = (0..len as u32).collect();
    let keys = upload_keys(client, &keys_host, num_words);
    let (sorted, payload) = radix_sort(&keys, &upload(client, &payload_host)).expect("radix_sort");
    let (want_keys, want_payload) = reference_sort(&keys_host, num_words, &payload_host);
    assert_eq!(sorted.len(), len);
    assert_eq!(sorted.num_words(), num_words);
    assert_eq!(
        sorted.to_host().expect("read keys"),
        want_keys,
        "sorted keys ({num_words} words)"
    );
    assert_eq!(
        payload.to_host().expect("read payload"),
        want_payload,
        "payload: stability"
    );
}

fn check_run_length(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let num_words = 1 + rng.below(4) as usize;
    let keys_host = run_keys(len, num_words, &mut rng);
    let runs = run_length(&upload_keys(client, &keys_host, num_words)).expect("run_length");
    let (want_keys, want_offsets) = reference_runs(&keys_host, num_words);
    assert_eq!(runs.keys.num_words(), num_words);
    assert_eq!(
        runs.keys.to_host().expect("read keys"),
        want_keys,
        "run keys"
    );
    assert_eq!(
        runs.offsets.to_host().expect("read offsets"),
        want_offsets,
        "run offsets"
    );
}

fn check_unique_by_key(client: &Client, len: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let num_words = 1 + rng.below(4) as usize;
    let unsorted = run_keys(len, num_words, &mut rng);
    let iota: Vec<u32> = (0..len as u32).collect();
    let (keys_host, _) = reference_sort(&unsorted, num_words, &iota);
    let payload_host: Vec<u32> = (0..len).map(|_| rng.next_u32()).collect();
    let unique = unique_by_key(
        &upload_keys(client, &keys_host, num_words),
        &upload(client, &payload_host),
    )
    .expect("unique_by_key");
    let (want_keys, offsets) = reference_runs(&keys_host, num_words);
    let want_first: Vec<u32> = offsets[..offsets.len() - 1]
        .iter()
        .map(|&o| payload_host[o as usize])
        .collect();
    let mut want_group = vec![0u32; len];
    for g in 0..offsets.len() - 1 {
        want_group[offsets[g] as usize..offsets[g + 1] as usize].fill(g as u32);
    }
    assert_eq!(
        unique.keys.to_host().expect("read keys"),
        want_keys,
        "unique keys"
    );
    assert_eq!(
        unique.first.to_host().expect("read first"),
        want_first,
        "first payloads"
    );
    assert_eq!(
        unique.group.to_host().expect("read group"),
        want_group,
        "groups"
    );
}

proptest! {
    #![proptest_config(prop_config())]

    #[test]
    fn prop_fill(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_fill, len, seed);
    }

    #[test]
    fn prop_copy(len in prop_len(), seed in any::<u64>()) {
        let client = test_client();
        each_scalar!(&client, check_copy, len, seed);
        check_copy_u64(&client, len, seed);
    }

    #[test]
    fn prop_reduce(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_reduce, len, seed);
    }

    #[test]
    fn prop_segmented_reduce(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_segmented, len, seed);
    }

    #[test]
    fn prop_inclusive_scan(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_inclusive_scan, len, seed);
    }

    #[test]
    fn prop_exclusive_scan(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_exclusive_scan, len, seed);
    }

    #[test]
    fn prop_gather(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_gather, len, seed);
    }

    #[test]
    fn prop_scatter(len in prop_len(), seed in any::<u64>()) {
        each_scalar!(&test_client(), check_scatter, len, seed);
    }

    #[test]
    fn prop_histogram(len in prop_len(), seed in any::<u64>()) {
        check_histogram(&test_client(), len, seed);
    }

    #[test]
    fn prop_radix_sort(len in prop_len(), seed in any::<u64>()) {
        check_radix_sort(&test_client(), len, seed);
    }

    #[test]
    fn prop_run_length(len in prop_len(), seed in any::<u64>()) {
        check_run_length(&test_client(), len, seed);
    }

    #[test]
    fn prop_unique_by_key(len in prop_len(), seed in any::<u64>()) {
        check_unique_by_key(&test_client(), len, seed);
    }
}

#[test]
fn copy_length_mismatch() {
    let client = test_client();
    let src = upload(&client, &[1u32, 2, 3]);
    let mut dst = DeviceBuffer::<u32>::uninit(&client, 4).expect("uninit");
    match copy(&src, &mut dst) {
        Err(GridError::LengthMismatch {
            expected: 3,
            found: 4,
        }) => {}
        other => panic!("copy of 3 into 4: {:?}", other.err()),
    }
}

#[test]
fn radix_sort_length_mismatch() {
    let client = test_client();
    let keys = upload_keys(&client, &[3, 1, 2], 1);
    match radix_sort(&keys, &upload(&client, &[0u32, 1])) {
        Err(GridError::LengthMismatch {
            expected: 3,
            found: 2,
        }) => {}
        other => panic!("3 keys with 2 payloads: {:?}", other.err()),
    }
}

/// Both ranking schemes of the radix sort on every device, through injected properties: the
/// split scheme of a GPU class (one item per unit), and the blocks scheme of a CPU class with
/// shared memory for eight digits per chunk (Grace's L1 does not hold all 256), at every vector
/// size the device offers. Both cover several cubes: two units per cube in the blocks case.
#[test]
fn radix_sort_both_schemes() {
    let client = test_client();
    let mut split = client.properties().clone();
    split.hardware.num_cpu_cores = None;
    split.hardware.max_units_per_cube = split.hardware.max_units_per_cube.min(64);
    let mut blocks = client.properties().clone();
    blocks.hardware.num_cpu_cores = Some(4);
    blocks.hardware.max_units_per_cube = blocks.hardware.max_units_per_cube.min(2);
    blocks.hardware.max_shared_memory_size =
        blocks.hardware.max_units_per_cube as usize * size_of::<u32>() * 8;
    for props in [split, blocks] {
        testing::with_properties(props, || {
            for n in vector_sizes(&client, size_of::<u32>()) {
                testing::with_vector_size(VectorSizeOverride::Fixed(n), || {
                    for (len, seed) in [(1, 1), (17, 2), (1000, 3), (20_001, 4), (20_001, 5)] {
                        check_radix_sort(&client, len, seed);
                    }
                });
            }
        });
    }
}

#[test]
fn scatter_out_of_range_is_dropped() {
    let client = test_client();
    let mut dst = upload(&client, &[10u32, 11, 12, 13]);
    let src = upload(&client, &[1u32, 2, 3]);
    scatter(&src, &upload(&client, &[2u32, 4, u32::MAX]), &mut dst).expect("scatter");
    assert_eq!(dst.to_host().expect("read"), vec![10, 11, 1, 13]);
}

/// Debug builds check gather's indices and panic on one out of range.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "index out of range")]
fn gather_out_of_range_panics_in_debug() {
    let client = test_client();
    let src = upload(&client, &[1.0f32, 2.0, 3.0]);
    let _ = gather(&src, &upload(&client, &[0u32, 3]));
}

/// Debug builds check scatter's indices and panic on a duplicate.
#[cfg(debug_assertions)]
#[test]
#[should_panic(expected = "duplicate index")]
fn scatter_duplicate_panics_in_debug() {
    let client = test_client();
    let mut dst = upload(&client, &[0u32; 4]);
    let _ = scatter(
        &upload(&client, &[1u32, 2]),
        &upload(&client, &[1u32, 1]),
        &mut dst,
    );
}

/// Release builds mask out-of-range indices to 0, also for an empty source.
#[cfg(not(debug_assertions))]
#[test]
fn gather_out_of_range_masks_to_zero() {
    let client = test_client();
    let src = upload(&client, &[5i32, 6, 7]);
    let out = gather(&src, &upload(&client, &[2u32, 3, 0, u32::MAX])).expect("gather");
    assert_eq!(out.to_host().expect("read"), vec![7, 0, 5, 0]);
    let empty = upload::<i32>(&client, &[]);
    let out = gather(&empty, &upload(&client, &[0u32, 1])).expect("gather from empty");
    assert_eq!(out.to_host().expect("read"), vec![0, 0]);
}

/// Release builds accept duplicates: memory-safe, one of the values lands.
#[cfg(not(debug_assertions))]
#[test]
fn scatter_duplicate_is_memory_safe() {
    let client = test_client();
    let mut dst = upload(&client, &[0u32; 4]);
    scatter(
        &upload(&client, &[1u32, 2, 3]),
        &upload(&client, &[1u32, 1, 3]),
        &mut dst,
    )
    .expect("scatter");
    let got = dst.to_host().expect("read");
    assert!(got[1] == 1 || got[1] == 2, "dst[1] = {}", got[1]);
    assert_eq!([got[0], got[2], got[3]], [0, 0, 3]);
}
