//! M2 acceptance: every vector-generic primitive gives the same results at vector size 1 and at
//! every larger power of two up to 16 that the device offers, for lengths 0, 1, 15, 16, 17 and
//! 10^6 + 3: integers and data moves bitwise, float reductions and scans within the tolerances.
//! Each case asserts through `record_plans` that its vectorized phase ran at the forced size.

mod common;

use common::*;
use cubecl::prelude::*;
use gridcl_device::copy;
use gridcl_device::*;
use gridcl_testing::test_client;

const SEED: u64 = 0x7665_6374_6f72_0002;

/// Runs `run` at every vector size for every length. `check` compares a result with the
/// reference of its length; results at sizes above 1 must also equal the size-1 result bitwise
/// when `bitwise` holds.
fn sweep<T: TestScalar>(
    client: &Client,
    kind: KernelKind,
    elem_bytes: usize,
    bitwise: bool,
    run: impl Fn(usize) -> Vec<T>,
    check: impl Fn(usize, &[T]),
) {
    let sizes = vector_sizes(client, elem_bytes);
    for len in INVARIANCE_LENS {
        let mut scalar: Option<Vec<T>> = None;
        for &n in &sizes {
            let got = at_vector_size(n, kind, len > 0, || run(len));
            check(len, &got);
            match &scalar {
                None => scalar = Some(got),
                Some(s) if bitwise => assert_bits(
                    &got,
                    s,
                    &format!("{kind:?}, length {len}, vector size {n} vs 1"),
                ),
                Some(_) => {}
            }
        }
    }
}

fn fill_t<T: TestScalar>(client: &Client) {
    let v = T::any(&mut SplitMix::new(SEED));
    sweep::<T>(
        client,
        KernelKind::Fill,
        size_of::<T>(),
        true,
        |len| {
            let mut x = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
            fill(&mut x, v).expect("fill");
            x.to_host().expect("read")
        },
        |len, got| assert_bits(got, &vec![v; len], "fill"),
    );
}

fn copy_t<T: TestScalar>(client: &Client) {
    sweep::<T>(
        client,
        KernelKind::Copy,
        size_of::<u32>(),
        true,
        |len| {
            let src = upload(client, &any_vec::<T>(len, SEED));
            let mut dst = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
            copy(&src, &mut dst).expect("copy");
            dst.to_host().expect("read")
        },
        |len, got| assert_bits(got, &any_vec::<T>(len, SEED), "copy"),
    );
}

fn reduce_t<T: TestScalar>(client: &Client) {
    for op in OPS {
        let sum = matches!(op, ReduceOp::Sum);
        sweep::<T>(
            client,
            KernelKind::Reduce,
            size_of::<T>(),
            !(T::IS_FLOAT && sum),
            |len| {
                let x = upload(client, &reduce_input::<T>(len, SEED, op));
                reduce(&x, op).expect("reduce").to_host().expect("read")
            },
            |len, got| {
                let (want, scale) = reference_reduce(&reduce_input::<T>(len, SEED, op), op);
                assert_value(got[0], want, scale, sum, &format!("reduce {op:?} of {len}"));
            },
        );
    }
}

fn scan_t<T: TestScalar>(client: &Client) {
    for exclusive in [false, true] {
        sweep::<T>(
            client,
            KernelKind::ScanTile,
            size_of::<T>(),
            !T::IS_FLOAT,
            |len| {
                let x = upload(client, &bounded_vec::<T>(len, SEED));
                let out = if exclusive {
                    exclusive_scan(&x)
                } else {
                    inclusive_scan(&x)
                };
                out.expect("scan").to_host().expect("read")
            },
            |len, got| {
                let (want, scales) = reference_inclusive_scan(&bounded_vec::<T>(len, SEED));
                if exclusive {
                    assert_eq!(got.len(), len + 1, "exclusive scan length");
                    assert_value(got[0], 0.0, 0.0, true, "exclusive_scan[0]");
                    assert_scan(&got[1..], &want, &scales, "exclusive_scan");
                } else {
                    assert_scan(got, &want, &scales, "inclusive_scan");
                }
            },
        );
    }
}

fn gather_t<T: TestScalar>(client: &Client) {
    let inputs = |len: usize| {
        let mut rng = SplitMix::new(SEED ^ len as u64);
        let src = any_vec::<T>(len / 2 + 1, SEED);
        let idx: Vec<u32> = (0..len)
            .map(|_| rng.below(src.len() as u64) as u32)
            .collect();
        (src, idx)
    };
    sweep::<T>(
        client,
        KernelKind::Gather,
        size_of::<T>(),
        true,
        |len| {
            let (src, idx) = inputs(len);
            gather(&upload(client, &src), &upload(client, &idx))
                .expect("gather")
                .to_host()
                .expect("read")
        },
        |len, got| {
            let (src, idx) = inputs(len);
            let want: Vec<T> = idx.iter().map(|&i| src[i as usize]).collect();
            assert_bits(got, &want, "gather");
        },
    );
}

/// Pack and unpack of `len` blocks of 2: pack from a source of len / 2 + 1 blocks at random
/// indices, unpack to the first `len` blocks of a random permutation of len + 3 blocks.
fn pack_inputs<T: TestScalar>(len: usize) -> (Vec<T>, Vec<u32>, Vec<u32>) {
    let mut rng = SplitMix::new(SEED ^ len as u64);
    let src = any_vec::<T>(2 * (len / 2 + 1), SEED);
    let idx: Vec<u32> = (0..len)
        .map(|_| rng.below((len / 2 + 1) as u64) as u32)
        .collect();
    let mut perm: Vec<u32> = (0..len as u32 + 3).collect();
    for i in (1..perm.len()).rev() {
        perm.swap(i, rng.below(i as u64 + 1) as usize);
    }
    perm.truncate(len);
    (src, idx, perm)
}

fn pack_t<T: TestScalar>(client: &Client) {
    sweep::<T>(
        client,
        KernelKind::Pack,
        size_of::<u32>(),
        true,
        |len| {
            let (src, idx, _) = pack_inputs::<T>(len);
            let mut out = DeviceBuffer::<T>::uninit(client, 2 * len).expect("uninit");
            pack(&upload(client, &src), &upload(client, &idx), 2, &mut out).expect("pack");
            out.to_host().expect("read")
        },
        |len, got| {
            let (src, idx, _) = pack_inputs::<T>(len);
            let want: Vec<T> = idx
                .iter()
                .flat_map(|&k| src[2 * k as usize..2 * k as usize + 2].iter().copied())
                .collect();
            assert_bits(got, &want, "pack");
        },
    );
}

fn unpack_t<T: TestScalar>(client: &Client) {
    sweep::<T>(
        client,
        KernelKind::Pack,
        size_of::<u32>(),
        true,
        |len| {
            let (_, _, perm) = pack_inputs::<T>(len);
            let packed = any_vec::<T>(2 * len, SEED ^ 2);
            let mut dst = upload(client, &any_vec::<T>(2 * (len + 3), SEED ^ 3));
            unpack(
                &upload(client, &packed),
                &upload(client, &perm),
                2,
                &mut dst,
            )
            .expect("unpack");
            dst.to_host().expect("read")
        },
        |len, got| {
            let (_, _, perm) = pack_inputs::<T>(len);
            let packed = any_vec::<T>(2 * len, SEED ^ 2);
            let mut want = any_vec::<T>(2 * (len + 3), SEED ^ 3);
            for (k, &i) in perm.iter().enumerate() {
                want[2 * i as usize..2 * i as usize + 2].copy_from_slice(&packed[2 * k..2 * k + 2]);
            }
            assert_bits(got, &want, "unpack");
        },
    );
}

#[test]
fn pack_sizes() {
    each_scalar!(&test_client(), pack_t);
}

#[test]
fn unpack_sizes() {
    each_scalar!(&test_client(), unpack_t);
}

#[test]
fn fill_sizes() {
    each_scalar!(&test_client(), fill_t);
}

#[test]
fn copy_sizes() {
    each_scalar!(&test_client(), copy_t);
}

#[test]
fn reduce_sizes() {
    each_scalar!(&test_client(), reduce_t);
}

#[test]
fn scan_sizes() {
    each_scalar!(&test_client(), scan_t);
}

#[test]
fn gather_sizes() {
    each_scalar!(&test_client(), gather_t);
}

#[test]
fn sort_digits_sizes() {
    let client = test_client();
    for num_words in [1usize, 2] {
        let inputs = |len: usize| {
            let keys = random_keys(
                len,
                num_words,
                1 << 12,
                &mut SplitMix::new(SEED ^ len as u64),
            );
            let payload: Vec<u32> = (0..len as u32).collect();
            (keys, payload)
        };
        sweep::<u32>(
            &client,
            KernelKind::SortDigits,
            size_of::<u32>(),
            true,
            |len| {
                let (keys, payload) = inputs(len);
                let (k, p) = radix_sort(
                    &upload_keys(&client, &keys, num_words),
                    &upload(&client, &payload),
                )
                .expect("radix_sort");
                let mut out = k.to_host().expect("read keys");
                out.extend(p.to_host().expect("read payload"));
                out
            },
            |len, got| {
                let (keys, payload) = inputs(len);
                let (mut want, want_payload) = reference_sort(&keys, num_words, &payload);
                want.extend(want_payload);
                assert_bits(got, &want, &format!("radix_sort, {num_words} words"));
            },
        );
    }
}
