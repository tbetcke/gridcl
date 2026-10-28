//! `pack`, `unpack` and `DeviceBuffer::write_host` (docs/plans/M6.md, T4): property tests
//! against a host reference, bitwise, for f32, f64, i32 and u32 at block sizes 1–4.

mod common;

use common::*;
use cubecl::prelude::*;
use gridcl_device::*;
use gridcl_testing::test_client;
use proptest::prelude::*;

/// `n` distinct indices below `m` (n <= m), in random order.
fn distinct(n: usize, m: usize, rng: &mut SplitMix) -> Vec<u32> {
    let mut all: Vec<u32> = (0..m as u32).collect();
    for i in 0..n.min(m.saturating_sub(1)) {
        let j = i + rng.below((m - i) as u64) as usize;
        all.swap(i, j);
    }
    all.truncate(n);
    all
}

/// The packed blocks of `src` at `idx`.
fn reference_pack<T: Copy>(src: &[T], idx: &[u32], block: usize) -> Vec<T> {
    idx.iter()
        .flat_map(|&k| {
            src[k as usize * block..(k as usize + 1) * block]
                .iter()
                .copied()
        })
        .collect()
}

/// `dst` with the blocks of `packed` written at `idx`.
fn reference_unpack<T: Copy>(packed: &[T], idx: &[u32], block: usize, dst: &[T]) -> Vec<T> {
    let mut out = dst.to_vec();
    for (k, &i) in idx.iter().enumerate() {
        out[i as usize * block..(i as usize + 1) * block]
            .copy_from_slice(&packed[k * block..(k + 1) * block]);
    }
    out
}

/// Packs `n` random blocks of a source with n / 2 + 1 blocks, and unpacks `n` blocks to
/// distinct positions of a destination with n + n / 3 + 1 blocks.
fn check_pack_unpack<T: TestScalar>(client: &Client, n: usize, block: usize, seed: u64) {
    let mut rng = SplitMix::new(seed);
    let src_blocks = n / 2 + 1;
    let src = any_vec::<T>(src_blocks * block, seed);
    let idx: Vec<u32> = (0..n)
        .map(|_| rng.below(src_blocks as u64) as u32)
        .collect();
    let mut out = DeviceBuffer::<T>::uninit(client, n * block).expect("uninit");
    pack(
        &upload(client, &src),
        &upload(client, &idx),
        block,
        &mut out,
    )
    .expect("pack");
    let packed = reference_pack(&src, &idx, block);
    assert_bits(
        &out.to_host().expect("read"),
        &packed,
        &format!("pack {n}x{block}"),
    );

    let dst_blocks = n + n / 3 + 1;
    let dst = any_vec::<T>(dst_blocks * block, seed ^ 1);
    let didx = distinct(n, dst_blocks, &mut rng);
    let mut d = upload(client, &dst);
    unpack(
        &upload(client, &packed),
        &upload(client, &didx),
        block,
        &mut d,
    )
    .expect("unpack");
    let want = reference_unpack(&packed, &didx, block, &dst);
    assert_bits(
        &d.to_host().expect("read"),
        &want,
        &format!("unpack {n}x{block}"),
    );
}

proptest! {
    #![proptest_config(prop_config())]

    #[test]
    fn prop_pack_unpack(n in prop_len(), block in 1..=4_usize, seed in any::<u64>()) {
        each_scalar!(&test_client(), check_pack_unpack, n, block, seed);
    }
}

#[test]
fn lengths_are_checked() {
    let client = test_client();
    let src = upload(&client, &[1.0f32; 6]);
    let idx = upload(&client, &[0u32, 1]);
    let mut out = DeviceBuffer::<f32>::uninit(&client, 4).unwrap();
    // 6 is not a whole number of blocks of 4.
    assert!(matches!(
        pack(&src, &idx, 4, &mut out),
        Err(GridError::LengthMismatch {
            expected: 8,
            found: 6
        })
    ));
    // Two indices of blocks of 3 need 6 packed values, not 4.
    assert!(matches!(
        pack(&src, &idx, 3, &mut out),
        Err(GridError::LengthMismatch {
            expected: 6,
            found: 4
        })
    ));
    assert!(matches!(
        unpack(&out, &idx, 0, &mut src.deep_clone().unwrap()),
        Err(GridError::LengthMismatch { .. })
    ));
}

#[test]
fn empty_index_lists_launch_nothing() {
    let client = test_client();
    let src = upload(&client, &[1u32, 2, 3]);
    let idx = DeviceBuffer::<u32>::uninit(&client, 0).unwrap();
    let mut out = DeviceBuffer::<u32>::uninit(&client, 0).unwrap();
    let ((), plans) = gridcl_device::testing::record_plans(|| {
        pack(&src, &idx, 1, &mut out).unwrap();
        let mut dst = src.deep_clone().unwrap();
        unpack(&out, &idx, 1, &mut dst).unwrap();
        assert_eq!(dst.to_host().unwrap(), vec![1, 2, 3]);
    });
    assert!(plans.iter().all(|(k, _)| *k != KernelKind::Pack));
}

#[test]
fn pack_past_the_end_gives_zeros() {
    let client = test_client();
    let src = upload(&client, &[5i32, 6]);
    let idx = upload(&client, &[1u32, 9, 0]);
    let mut out = DeviceBuffer::<i32>::uninit(&client, 3).unwrap();
    pack(&src, &idx, 1, &mut out).unwrap();
    assert_eq!(out.to_host().unwrap(), vec![6, 0, 5]);
    let empty = DeviceBuffer::<i32>::uninit(&client, 0).unwrap();
    pack(&empty, &idx, 1, &mut out).unwrap();
    assert_eq!(out.to_host().unwrap(), vec![0, 0, 0]);
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "unpack: duplicate index")]
fn unpack_rejects_duplicates_in_debug() {
    let client = test_client();
    let packed = upload(&client, &[1u32, 2]);
    let idx = upload(&client, &[3u32, 3]);
    let mut dst = upload(&client, &[0u32; 4]);
    unpack(&packed, &idx, 1, &mut dst).unwrap();
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "unpack: index out of range")]
fn unpack_rejects_out_of_range_in_debug() {
    let client = test_client();
    let packed = upload(&client, &[1u32]);
    let idx = upload(&client, &[4u32]);
    let mut dst = upload(&client, &[0u32; 4]);
    unpack(&packed, &idx, 1, &mut dst).unwrap();
}

#[test]
fn write_host_round_trip() {
    let client = test_client();
    for len in [0, 1, 15, 16, 17, 1000] {
        let data: Vec<f32> = (0..len).map(|i| i as f32 * 0.5).collect();
        let mut b = DeviceBuffer::<f32>::zeros(&client, len).unwrap();
        b.write_host(&data).unwrap();
        assert_eq!(b.to_host().unwrap(), data);
        let next: Vec<f32> = data.iter().map(|x| -x).collect();
        b.write_host(&next).unwrap();
        assert_eq!(b.to_host().unwrap(), next);
        assert!(matches!(
            b.write_host(&vec![0.0; len + 1]),
            Err(GridError::LengthMismatch { .. })
        ));
    }
    let mut b = DeviceBuffer::from_host(&client, &[1u32, 2]).unwrap();
    assert!(matches!(
        b.write_host(&[1, 2, 3]),
        Err(GridError::LengthMismatch {
            expected: 2,
            found: 3
        })
    ));
}
