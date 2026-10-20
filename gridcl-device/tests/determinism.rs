//! M2 acceptance: results are bitwise identical across two runs on the same device.

mod common;

use common::*;
use cubecl::prelude::*;
use gridcl_device::copy;
use gridcl_device::*;
use gridcl_testing::test_client;

/// Not a multiple of any vector, cube or tile size, and large enough for several cubes.
const LEN: usize = 1_000_003;
const SEED: u64 = 0x6465_7465_726d_0002;

/// Runs `f` twice and panics unless both results are bitwise equal.
fn twice<T: TestScalar>(what: &str, f: impl Fn() -> Vec<T>) {
    let first = f();
    let second = f();
    assert_bits(&second, &first, &format!("{what}: second run"));
}

fn fill_t<T: TestScalar>(client: &Client) {
    twice::<T>("fill", || {
        let mut x = DeviceBuffer::<T>::uninit(client, LEN).expect("uninit");
        fill(&mut x, T::any(&mut SplitMix::new(SEED))).expect("fill");
        x.to_host().expect("read")
    });
}

fn copy_t<T: TestScalar>(client: &Client) {
    let src = upload(client, &any_vec::<T>(LEN, SEED));
    twice::<T>("copy", || {
        let mut dst = DeviceBuffer::<T>::uninit(client, LEN).expect("uninit");
        copy(&src, &mut dst).expect("copy");
        dst.to_host().expect("read")
    });
}

fn reduce_t<T: TestScalar>(client: &Client) {
    for op in OPS {
        let x = upload(client, &reduce_input::<T>(LEN, SEED, op));
        twice::<T>(&format!("reduce {op:?}"), || {
            reduce(&x, op).expect("reduce").to_host().expect("read")
        });
    }
}

fn segmented_t<T: TestScalar>(client: &Client) {
    let offsets = upload(client, &random_offsets(LEN, 5000, &mut SplitMix::new(SEED)));
    for op in OPS {
        let x = upload(client, &reduce_input::<T>(LEN, SEED ^ 1, op));
        twice::<T>(&format!("segmented_reduce {op:?}"), || {
            segmented_reduce(&x, &offsets, op)
                .expect("segmented_reduce")
                .to_host()
                .expect("read")
        });
    }
}

fn inclusive_t<T: TestScalar>(client: &Client) {
    let x = upload(client, &bounded_vec::<T>(LEN, SEED));
    twice::<T>("inclusive_scan", || {
        inclusive_scan(&x).expect("scan").to_host().expect("read")
    });
}

fn exclusive_t<T: TestScalar>(client: &Client) {
    let x = upload(client, &bounded_vec::<T>(LEN, SEED));
    twice::<T>("exclusive_scan", || {
        exclusive_scan(&x).expect("scan").to_host().expect("read")
    });
}

fn gather_t<T: TestScalar>(client: &Client) {
    let mut rng = SplitMix::new(SEED);
    let src = upload(client, &any_vec::<T>(LEN / 3, SEED));
    let idx: Vec<u32> = (0..LEN)
        .map(|_| rng.below((LEN / 3) as u64) as u32)
        .collect();
    let idx = upload(client, &idx);
    twice::<T>("gather", || {
        gather(&src, &idx).expect("gather").to_host().expect("read")
    });
}

fn scatter_t<T: TestScalar>(client: &Client) {
    let mut rng = SplitMix::new(SEED);
    let mut perm: Vec<u32> = (0..LEN as u32).collect();
    for i in (1..LEN).rev() {
        perm.swap(i, rng.below(i as u64 + 1) as usize);
    }
    let src = upload(client, &any_vec::<T>(LEN, SEED));
    let idx = upload(client, &perm);
    twice::<T>("scatter", || {
        let mut dst = DeviceBuffer::<T>::zeros(client, LEN).expect("zeros");
        scatter(&src, &idx, &mut dst).expect("scatter");
        dst.to_host().expect("read")
    });
}

#[test]
fn fill_twice() {
    each_scalar!(&test_client(), fill_t);
}

#[test]
fn copy_twice() {
    each_scalar!(&test_client(), copy_t);
}

#[test]
fn reduce_twice() {
    each_scalar!(&test_client(), reduce_t);
}

#[test]
fn segmented_reduce_twice() {
    each_scalar!(&test_client(), segmented_t);
}

#[test]
fn inclusive_scan_twice() {
    each_scalar!(&test_client(), inclusive_t);
}

#[test]
fn exclusive_scan_twice() {
    each_scalar!(&test_client(), exclusive_t);
}

#[test]
fn gather_twice() {
    each_scalar!(&test_client(), gather_t);
}

#[test]
fn scatter_twice() {
    each_scalar!(&test_client(), scatter_t);
}

#[test]
fn histogram_twice() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    let values: Vec<u32> = (0..LEN).map(|_| rng.below(70_000) as u32).collect();
    let values = upload(&client, &values);
    twice::<u32>("histogram", || {
        histogram(&values, 65_536)
            .expect("histogram")
            .to_host()
            .expect("read")
    });
}

#[test]
fn sort_twice() {
    let client = test_client();
    let keys = upload_keys(
        &client,
        &random_keys(LEN, 2, 1 << 10, &mut SplitMix::new(SEED)),
        2,
    );
    let payload = upload(&client, &(0..LEN as u32).collect::<Vec<_>>());
    twice::<u32>("radix_sort", || {
        let (k, p) = radix_sort(&keys, &payload).expect("radix_sort");
        let mut out = k.to_host().expect("read keys");
        out.extend(p.to_host().expect("read payload"));
        out
    });
}

#[test]
fn run_length_twice() {
    let client = test_client();
    let keys = upload_keys(&client, &run_keys(LEN, 2, &mut SplitMix::new(SEED)), 2);
    twice::<u32>("run_length", || {
        let runs = run_length(&keys).expect("run_length");
        let mut out = runs.keys.to_host().expect("read keys");
        out.extend(runs.offsets.to_host().expect("read offsets"));
        out
    });
}

#[test]
fn unique_twice() {
    let client = test_client();
    let mut rng = SplitMix::new(SEED);
    let mut keys_host = random_keys(LEN, 1, 1 << 16, &mut rng);
    keys_host.sort_unstable();
    let keys = upload_keys(&client, &keys_host, 1);
    let payload = upload(
        &client,
        &(0..LEN).map(|_| rng.next_u32()).collect::<Vec<_>>(),
    );
    twice::<u32>("unique_by_key", || {
        let u = unique_by_key(&keys, &payload).expect("unique_by_key");
        let mut out = u.keys.to_host().expect("read keys");
        out.extend(u.first.to_host().expect("read first"));
        out.extend(u.group.to_host().expect("read group"));
        out
    });
}
