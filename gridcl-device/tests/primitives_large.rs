//! M2 acceptance: the primitives against CPU references at the fixed sizes 10^6 and 10^7.

mod common;

use common::*;
use cubecl::prelude::*;
use gridcl_device::copy;
use gridcl_device::*;
use gridcl_testing::test_client;

const SEED: u64 = 0x6772_6964_636c_0002;

fn fill_one<T: TestScalar>(client: &Client, len: usize) {
    let mut x = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
    let v = T::any(&mut SplitMix::new(SEED));
    fill(&mut x, v).expect("fill");
    assert_bits(&x.to_host().expect("read"), &vec![v; len], "fill");
}

fn copy_one<T: TestScalar>(client: &Client, len: usize) {
    let src_host = any_vec::<T>(len, SEED);
    let mut dst = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
    copy(&upload(client, &src_host), &mut dst).expect("copy");
    assert_bits(&dst.to_host().expect("read"), &src_host, "copy");
}

fn reduce_one<T: TestScalar>(client: &Client, len: usize) {
    for op in OPS {
        let x_host = reduce_input::<T>(len, SEED, op);
        let got = reduce(&upload(client, &x_host), op)
            .expect("reduce")
            .to_host()
            .expect("read")[0];
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

fn segmented_one<T: TestScalar>(client: &Client, len: usize) {
    let mut rng = SplitMix::new(SEED);
    for max_segments in [3, 1000, len / 3] {
        let offsets_host = random_offsets(len, max_segments, &mut rng);
        let offsets = upload(client, &offsets_host);
        for op in OPS {
            let values_host = reduce_input::<T>(len, SEED ^ 1, op);
            let got = segmented_reduce(&upload(client, &values_host), &offsets, op)
                .expect("segmented_reduce")
                .to_host()
                .expect("read");
            let want = reference_segmented(&values_host, &offsets_host, op);
            assert_eq!(got.len(), want.len());
            for (k, (&g, &(w, s))) in got.iter().zip(&want).enumerate() {
                assert_value(
                    g,
                    w,
                    s,
                    matches!(op, ReduceOp::Sum),
                    &format!("segment {k} {op:?}"),
                );
            }
        }
    }
}

fn scan_one<T: TestScalar>(client: &Client, len: usize) {
    let x_host = bounded_vec::<T>(len, SEED);
    let x = upload(client, &x_host);
    let (want, scales) = reference_inclusive_scan(&x_host);
    let incl = inclusive_scan(&x)
        .expect("inclusive_scan")
        .to_host()
        .expect("read");
    assert_scan(&incl, &want, &scales, "inclusive_scan");
    let excl = exclusive_scan(&x)
        .expect("exclusive_scan")
        .to_host()
        .expect("read");
    assert_eq!(excl.len(), len + 1);
    assert_value(excl[0], 0.0, 0.0, true, "exclusive_scan[0]");
    assert_scan(&excl[1..], &want, &scales, "exclusive_scan[1..]");
}

fn gather_one<T: TestScalar>(client: &Client, len: usize) {
    let mut rng = SplitMix::new(SEED);
    let src_host = any_vec::<T>(len / 2 + 1, SEED);
    let idx_host: Vec<u32> = (0..len)
        .map(|_| rng.below(src_host.len() as u64) as u32)
        .collect();
    let out = gather(&upload(client, &src_host), &upload(client, &idx_host)).expect("gather");
    let want: Vec<T> = idx_host.iter().map(|&i| src_host[i as usize]).collect();
    assert_bits(&out.to_host().expect("read"), &want, "gather");
}

fn scatter_one<T: TestScalar>(client: &Client, len: usize) {
    // A permutation of 0..len scatters every element once.
    let mut rng = SplitMix::new(SEED);
    let mut idx_host: Vec<u32> = (0..len as u32).collect();
    for i in (1..len).rev() {
        idx_host.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let src_host = any_vec::<T>(len, SEED ^ 2);
    let mut dst = DeviceBuffer::<T>::uninit(client, len).expect("uninit");
    scatter(
        &upload(client, &src_host),
        &upload(client, &idx_host),
        &mut dst,
    )
    .expect("scatter");
    let mut want = src_host.clone();
    for (&i, &v) in idx_host.iter().zip(&src_host) {
        want[i as usize] = v;
    }
    assert_bits(&dst.to_host().expect("read"), &want, "scatter");
}

#[test]
fn fill_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, fill_one, len);
    }
}

#[test]
fn copy_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, copy_one, len);
    }
}

#[test]
fn reduce_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, reduce_one, len);
    }
}

#[test]
fn segmented_reduce_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, segmented_one, len);
    }
}

#[test]
fn scan_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, scan_one, len);
    }
}

#[test]
fn gather_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, gather_one, len);
    }
}

#[test]
fn scatter_large() {
    let client = test_client();
    for len in LARGE_LENS {
        each_scalar!(&client, scatter_one, len);
    }
}

#[test]
fn histogram_large() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    for len in LARGE_LENS {
        for num_bins in [256u32, 100_000] {
            let values_host: Vec<u32> = (0..len)
                .map(|_| rng.below(u64::from(num_bins) + 7) as u32)
                .collect();
            let got = histogram(&upload(&client, &values_host), num_bins).expect("histogram");
            let mut want = vec![0u32; num_bins as usize];
            for &v in &values_host {
                if let Some(c) = want.get_mut(v as usize) {
                    *c += 1;
                }
            }
            assert_eq!(
                got.to_host().expect("read"),
                want,
                "histogram {len} into {num_bins}"
            );
        }
    }
}

#[test]
fn sort_large() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    for len in LARGE_LENS {
        let keys_host = random_keys(len, 2, 0, &mut rng);
        let payload_host: Vec<u32> = (0..len as u32).collect();
        let (keys, payload) = radix_sort(
            &upload_keys(&client, &keys_host, 2),
            &upload(&client, &payload_host),
        )
        .expect("radix_sort");
        let (want_keys, want_payload) = reference_sort(&keys_host, 2, &payload_host);
        assert!(
            keys.to_host().expect("read keys") == want_keys,
            "sorted keys of {len}"
        );
        assert!(
            payload.to_host().expect("read payload") == want_payload,
            "payload of {len}"
        );
    }
}

#[test]
fn run_length_large() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    for len in LARGE_LENS {
        let keys_host = run_keys(len, 2, &mut rng);
        let runs = run_length(&upload_keys(&client, &keys_host, 2)).expect("run_length");
        let (want_keys, want_offsets) = reference_runs(&keys_host, 2);
        assert!(
            runs.keys.to_host().expect("read") == want_keys,
            "run keys of {len}"
        );
        assert!(
            runs.offsets.to_host().expect("read") == want_offsets,
            "run offsets of {len}"
        );
    }
}

#[test]
fn unique_large() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    for len in LARGE_LENS {
        let mut keys_host = random_keys(len, 1, 1 << 18, &mut rng);
        keys_host.sort_unstable();
        let payload_host: Vec<u32> = (0..len).map(|_| rng.next_u32()).collect();
        let unique = unique_by_key(
            &upload_keys(&client, &keys_host, 1),
            &upload(&client, &payload_host),
        )
        .expect("unique_by_key");
        let (want_keys, offsets) = reference_runs(&keys_host, 1);
        let want_first: Vec<u32> = offsets[..offsets.len() - 1]
            .iter()
            .map(|&o| payload_host[o as usize])
            .collect();
        let mut want_group = vec![0u32; len];
        for g in 0..offsets.len() - 1 {
            want_group[offsets[g] as usize..offsets[g + 1] as usize].fill(g as u32);
        }
        assert!(
            unique.keys.to_host().expect("read") == want_keys,
            "unique keys of {len}"
        );
        assert!(
            unique.first.to_host().expect("read") == want_first,
            "first of {len}"
        );
        assert!(
            unique.group.to_host().expect("read") == want_group,
            "group of {len}"
        );
    }
}
