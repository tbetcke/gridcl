//! Shared helpers of the gridcl-device acceptance tests: CPU references, random inputs and the
//! size lists of docs/design.md, "Testing and validation".
//!
//! Every integration test file includes this module; not every file uses every helper.
#![allow(dead_code, unused_macros, unused_imports)]

use std::fmt::Debug;

use cubecl::prelude::*;
use gridcl_device::testing::{record_plans, with_vector_size};
use gridcl_device::{
    DeviceBuffer, DeviceScalar, KernelKind, LaunchPlan, ReduceOp, SortKeys, VectorSizeOverride,
};
use gridcl_testing::{TOL_F32, TOL_F64, supports};

/// Cases per property test.
pub const PROP_CASES: u32 = 256;
/// Largest length of a property test.
pub const PROP_MAX_LEN: usize = 100_000;
/// The fixed lengths of the large tests.
pub const LARGE_LENS: [usize; 2] = [1_000_000, 10_000_000];
/// The lengths of the vector-size invariance tests.
pub const INVARIANCE_LENS: [usize; 6] = [0, 1, 15, 16, 17, 1_000_003];

/// The proptest configuration of every property test.
pub fn prop_config() -> proptest::test_runner::Config {
    proptest::test_runner::Config {
        cases: PROP_CASES,
        ..proptest::test_runner::Config::default()
    }
}

/// Lengths for property tests: small ones often, up to `PROP_MAX_LEN`.
pub fn prop_len() -> impl proptest::strategy::Strategy<Value = usize> {
    use proptest::prelude::*;
    prop_oneof![0usize..=64, 0usize..=PROP_MAX_LEN]
}

/// SplitMix64: a small deterministic generator, so proptest shrinks a seed, not a vector.
#[derive(Clone, Debug)]
pub struct SplitMix(u64);

impl SplitMix {
    pub fn new(seed: u64) -> Self {
        SplitMix(seed)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }
    pub fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
    /// Uniform in `0..n`; `n` > 0.
    pub fn below(&mut self, n: u64) -> u64 {
        assert!(n > 0);
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }
    /// Uniform in [0, 1).
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A scalar of the typed primitives with its host reference arithmetic.
pub trait TestScalar: DeviceScalar + PartialEq + Debug {
    const NAME: &'static str;
    const IS_FLOAT: bool;
    /// Relative tolerance of float sums; 0 for integers (exact).
    const TOL: f64;
    /// A value for sums: small enough that no sum of 10^7 of them overflows.
    fn bounded(rng: &mut SplitMix) -> Self;
    /// A value for moves, min and max: finite, spread over the type's range.
    fn any(rng: &mut SplitMix) -> Self;
    fn host_f64(self) -> f64;
    /// Exact for integers in range; rounded for floats.
    fn of_f64(x: f64) -> Self;
    fn of_i64(x: i64) -> Self;
    fn host_i64(self) -> i64;
    fn bits(self) -> u64;
    /// The identity of `op` that the primitives return for empty inputs and segments:
    /// 0, `T::max_value()` or `T::min_value()` (the largest finite values for floats).
    fn identity(op: ReduceOp) -> Self {
        match op {
            ReduceOp::Sum => Self::of_i64(0),
            ReduceOp::Min => <Self as Numeric>::max_value(),
            ReduceOp::Max => <Self as Numeric>::min_value(),
        }
    }
}

impl TestScalar for f32 {
    const NAME: &'static str = "f32";
    const IS_FLOAT: bool = true;
    const TOL: f64 = TOL_F32;
    fn bounded(rng: &mut SplitMix) -> Self {
        (rng.unit() * 2.0 - 1.0) as f32
    }
    fn any(rng: &mut SplitMix) -> Self {
        ((rng.unit() * 2.0 - 1.0) * 1e3) as f32
    }
    fn host_f64(self) -> f64 {
        f64::from(self)
    }
    fn of_f64(x: f64) -> Self {
        x as f32
    }
    fn of_i64(x: i64) -> Self {
        x as f32
    }
    fn host_i64(self) -> i64 {
        self as i64
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl TestScalar for f64 {
    const NAME: &'static str = "f64";
    const IS_FLOAT: bool = true;
    const TOL: f64 = TOL_F64;
    fn bounded(rng: &mut SplitMix) -> Self {
        rng.unit() * 2.0 - 1.0
    }
    fn any(rng: &mut SplitMix) -> Self {
        (rng.unit() * 2.0 - 1.0) * 1e3
    }
    fn host_f64(self) -> f64 {
        self
    }
    fn of_f64(x: f64) -> Self {
        x
    }
    fn of_i64(x: i64) -> Self {
        x as f64
    }
    fn host_i64(self) -> i64 {
        self as i64
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
}

impl TestScalar for i32 {
    const NAME: &'static str = "i32";
    const IS_FLOAT: bool = false;
    const TOL: f64 = 0.0;
    fn bounded(rng: &mut SplitMix) -> Self {
        rng.below(201) as i32 - 100
    }
    fn any(rng: &mut SplitMix) -> Self {
        rng.next_u32() as i32
    }
    fn host_f64(self) -> f64 {
        f64::from(self)
    }
    fn of_f64(x: f64) -> Self {
        x as i32
    }
    fn of_i64(x: i64) -> Self {
        i32::try_from(x).expect("i32 reference out of range")
    }
    fn host_i64(self) -> i64 {
        i64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self as u32)
    }
}

impl TestScalar for u32 {
    const NAME: &'static str = "u32";
    const IS_FLOAT: bool = false;
    const TOL: f64 = 0.0;
    fn bounded(rng: &mut SplitMix) -> Self {
        rng.below(401) as u32
    }
    fn any(rng: &mut SplitMix) -> Self {
        rng.next_u32()
    }
    fn host_f64(self) -> f64 {
        f64::from(self)
    }
    fn of_f64(x: f64) -> Self {
        x as u32
    }
    fn of_i64(x: i64) -> Self {
        u32::try_from(x).expect("u32 reference out of range")
    }
    fn host_i64(self) -> i64 {
        i64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self)
    }
}

/// Runs `$f::<T>($args)` for every `DeviceScalar` the device supports. f32, i32 and u32 must
/// be supported everywhere; f64 runs where `supports_type` allows it.
macro_rules! each_scalar {
    ($client:expr, $f:ident $(, $arg:expr)* $(,)?) => {{
        let client: &cubecl::prelude::Client = $client;
        for (name, ok) in [
            ("f32", gridcl_testing::supports::<f32>(client)),
            ("i32", gridcl_testing::supports::<i32>(client)),
            ("u32", gridcl_testing::supports::<u32>(client)),
        ] {
            assert!(ok, "{name} must be supported on every device");
        }
        $f::<f32>(client $(, $arg)*);
        if gridcl_testing::supports::<f64>(client) {
            $f::<f64>(client $(, $arg)*);
        } else {
            println!("{}: f64 skipped, unsupported on {}", stringify!($f), client.name());
        }
        $f::<i32>(client $(, $arg)*);
        $f::<u32>(client $(, $arg)*);
    }};
}
pub(crate) use each_scalar;

pub const OPS: [ReduceOp; 3] = [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max];

pub fn bounded_vec<T: TestScalar>(len: usize, seed: u64) -> Vec<T> {
    let mut rng = SplitMix::new(seed);
    (0..len).map(|_| T::bounded(&mut rng)).collect()
}

pub fn any_vec<T: TestScalar>(len: usize, seed: u64) -> Vec<T> {
    let mut rng = SplitMix::new(seed);
    (0..len).map(|_| T::any(&mut rng)).collect()
}

/// Inputs of a reduction with `op`: bounded for sums, spread for min and max.
pub fn reduce_input<T: TestScalar>(len: usize, seed: u64, op: ReduceOp) -> Vec<T> {
    match op {
        ReduceOp::Sum => bounded_vec(len, seed),
        ReduceOp::Min | ReduceOp::Max => any_vec(len, seed),
    }
}

/// The reference reduction of `x` in index order: f64 for float sums, exact otherwise.
/// Returns the value and Σ|x| (the scale of the float tolerance).
pub fn reference_reduce<T: TestScalar>(x: &[T], op: ReduceOp) -> (f64, f64) {
    let abs: f64 = x.iter().map(|v| v.host_f64().abs()).sum();
    let value = match op {
        ReduceOp::Sum if T::IS_FLOAT => x.iter().map(|v| v.host_f64()).sum(),
        ReduceOp::Sum => x.iter().map(|v| v.host_i64()).sum::<i64>() as f64,
        ReduceOp::Min => x
            .iter()
            .copied()
            .fold(T::identity(op), |a, b| if b < a { b } else { a })
            .host_f64(),
        ReduceOp::Max => x
            .iter()
            .copied()
            .fold(T::identity(op), |a, b| if b > a { b } else { a })
            .host_f64(),
    };
    (value, abs)
}

/// Panics unless `got` matches the reference `want`: float sums to `T::TOL · scale`, everything
/// else exactly.
pub fn assert_value<T: TestScalar>(got: T, want: f64, scale: f64, sum: bool, what: &str) {
    if T::IS_FLOAT && sum {
        let err = (got.host_f64() - want).abs();
        assert!(
            err <= T::TOL * scale.max(f64::MIN_POSITIVE),
            "{what} ({}): got {got:?}, want {want} (error {err:e}, allowed {:e})",
            T::NAME,
            T::TOL * scale
        );
    } else {
        let want_t = if T::IS_FLOAT {
            T::of_f64(want)
        } else {
            T::of_i64(want as i64)
        };
        assert!(
            got.bits() == want_t.bits(),
            "{what} ({}): got {got:?}, want {want_t:?}",
            T::NAME
        );
    }
}

/// Panics at the first element where `got` and `want` differ bitwise.
pub fn assert_bits<T: TestScalar>(got: &[T], want: &[T], what: &str) {
    assert_eq!(got.len(), want.len(), "{what} ({}): length", T::NAME);
    if let Some(i) = (0..got.len()).find(|&i| got[i].bits() != want[i].bits()) {
        panic!(
            "{what} ({}): element {i} of {} is {:?}, want {:?}",
            T::NAME,
            got.len(),
            got[i],
            want[i]
        );
    }
}

/// Inclusive prefix sums of `x` in f64 (exact for the bounded integers), with the prefix
/// sums of |x| as the tolerance scale.
pub fn reference_inclusive_scan<T: TestScalar>(x: &[T]) -> (Vec<f64>, Vec<f64>) {
    let mut sum = 0.0;
    let mut int_sum = 0i64;
    let mut abs = 0.0;
    let mut values = Vec::with_capacity(x.len());
    let mut scales = Vec::with_capacity(x.len());
    for v in x {
        sum += v.host_f64();
        int_sum += v.host_i64();
        abs += v.host_f64().abs();
        values.push(if T::IS_FLOAT { sum } else { int_sum as f64 });
        scales.push(abs);
    }
    (values, scales)
}

/// Panics unless `got` matches the scan reference elementwise (sums: floats to the tolerance).
pub fn assert_scan<T: TestScalar>(got: &[T], want: &[f64], scales: &[f64], what: &str) {
    assert_eq!(got.len(), want.len(), "{what} ({}): length", T::NAME);
    for i in 0..got.len() {
        assert_value(got[i], want[i], scales[i], true, &format!("{what}[{i}]"));
    }
}

/// CSR offsets over `len` values with up to `max_segments` segments. Some cases add offsets
/// past `len` and decreasing pairs, which the primitive clamps and treats as empty.
pub fn random_offsets(len: usize, max_segments: usize, rng: &mut SplitMix) -> Vec<u32> {
    let segments = rng.below(max_segments as u64 + 1) as usize;
    let mut cuts: Vec<u32> = (0..segments.saturating_sub(1))
        .map(|_| rng.below(len as u64 + 1) as u32)
        .collect();
    cuts.sort_unstable();
    let mut offsets = Vec::with_capacity(segments + 1);
    if segments > 0 {
        offsets.push(0);
        offsets.extend(cuts);
        offsets.push(len as u32);
    }
    if offsets.len() > 2 && rng.below(4) == 0 {
        let i = 1 + rng.below(offsets.len() as u64 - 2) as usize;
        offsets[i] = (len as u32).saturating_add(1 + rng.below(1000) as u32);
    }
    if offsets.len() > 2 && rng.below(4) == 0 {
        let i = rng.below(offsets.len() as u64 - 1) as usize;
        offsets.swap(i, i + 1);
    }
    offsets
}

/// The reference segmented reduction: one (value, scale) per segment, offsets clamped to
/// `[0, len]`, a decreasing pair empty.
pub fn reference_segmented<T: TestScalar>(
    values: &[T],
    offsets: &[u32],
    op: ReduceOp,
) -> Vec<(f64, f64)> {
    let n = values.len();
    offsets
        .windows(2)
        .map(|w| {
            let lo = (w[0] as usize).min(n);
            let hi = (w[1] as usize).min(n);
            if hi <= lo {
                (T::identity(op).host_f64(), 0.0)
            } else {
                reference_reduce(&values[lo..hi], op)
            }
        })
        .collect()
}

/// `len` random keys of `num_words` words, key-major. With `dup_range` > 0 the words come from
/// `0..dup_range`, so keys repeat.
pub fn random_keys(len: usize, num_words: usize, dup_range: u32, rng: &mut SplitMix) -> Vec<u32> {
    (0..len * num_words)
        .map(|_| {
            if dup_range > 0 {
                rng.below(u64::from(dup_range)) as u32
            } else {
                rng.next_u32()
            }
        })
        .collect()
}

/// Stable sort of key-major keys with their payload: (sorted keys, permuted payload).
pub fn reference_sort(keys: &[u32], num_words: usize, payload: &[u32]) -> (Vec<u32>, Vec<u32>) {
    let len = payload.len();
    assert_eq!(keys.len(), len * num_words);
    let mut order: Vec<usize> = (0..len).collect();
    order.sort_by(|&a, &b| {
        keys[a * num_words..(a + 1) * num_words].cmp(&keys[b * num_words..(b + 1) * num_words])
    });
    let sorted_keys = order
        .iter()
        .flat_map(|&i| keys[i * num_words..(i + 1) * num_words].iter().copied())
        .collect();
    let sorted_payload = order.iter().map(|&i| payload[i]).collect();
    (sorted_keys, sorted_payload)
}

/// Runs of consecutive equal keys: (run keys, offsets with runs + 1 entries).
pub fn reference_runs(keys: &[u32], num_words: usize) -> (Vec<u32>, Vec<u32>) {
    let len = keys.len() / num_words;
    let key = |i: usize| &keys[i * num_words..(i + 1) * num_words];
    let mut run_keys = Vec::new();
    let mut offsets = Vec::new();
    for i in 0..len {
        if i == 0 || key(i) != key(i - 1) {
            run_keys.extend_from_slice(key(i));
            offsets.push(i as u32);
        }
    }
    offsets.push(len as u32);
    (run_keys, offsets)
}

/// Keys of `len` keys made of runs (random lengths, random keys), key-major.
pub fn run_keys(len: usize, num_words: usize, rng: &mut SplitMix) -> Vec<u32> {
    let mut keys = Vec::with_capacity(len * num_words);
    while keys.len() < len * num_words {
        let run = 1 + rng.below(8) as usize;
        let key: Vec<u32> = (0..num_words).map(|_| rng.below(4) as u32).collect();
        for _ in 0..run {
            if keys.len() < len * num_words {
                keys.extend_from_slice(&key);
            }
        }
    }
    keys
}

/// The vector sizes of the invariance tests: 1 plus every power of two from 2 to 16 no wider
/// than the device's widest IO-optimized size for `elem_bytes`.
pub fn vector_sizes(client: &Client, elem_bytes: usize) -> Vec<usize> {
    let widest = client
        .io_optimized_vector_sizes(elem_bytes)
        .next()
        .unwrap_or(1);
    let mut sizes = vec![1];
    sizes.extend([2, 4, 8, 16].into_iter().filter(|&n| n <= widest));
    sizes
}

/// Runs `f` at vector size `n` and returns its result. Unless `expect_launch` is false (empty
/// inputs never launch), asserts that the first plan of `kind` ran at exactly `n`.
pub fn at_vector_size<R>(
    n: usize,
    kind: KernelKind,
    expect_launch: bool,
    f: impl FnOnce() -> R,
) -> R {
    let (result, plans): (R, Vec<(KernelKind, LaunchPlan)>) =
        record_plans(|| with_vector_size(VectorSizeOverride::Fixed(n), f));
    let first = plans.iter().find(|(k, _)| *k == kind);
    if expect_launch {
        let (_, plan) = first.unwrap_or_else(|| panic!("no {kind:?} plan was made"));
        assert_eq!(
            plan.vector_size, n,
            "the {kind:?} phase ran at vector size {}, not {n}",
            plan.vector_size
        );
    } else {
        assert!(first.is_none(), "an empty input made a {kind:?} plan");
    }
    result
}

/// Uploads `data`.
pub fn upload<T: CubeElement>(client: &Client, data: &[T]) -> DeviceBuffer<T> {
    DeviceBuffer::from_host(client, data).expect("upload")
}

/// Uploads key-major keys.
pub fn upload_keys(client: &Client, keys: &[u32], num_words: usize) -> SortKeys {
    SortKeys::from_host(client, keys, num_words).expect("upload keys")
}
