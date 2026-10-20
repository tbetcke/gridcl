//! M0 acceptance test: an element-wise add on the device chosen by `GRIDCL_TEST_DEVICE`.
//!
//! The kernel is test code and scalar. Its launch shape comes from
//! `LaunchPlan::elementwise(…, KernelKind::Copy, …)` with `max_vector = 1`, and it runs the
//! kernel loop of Listing B5 (docs/plans/M0.md, question 2; docs/plans/M2.md, T10). Copy is
//! the memory-bound element-wise family.
//!
//! On the CPU runtime the result is compared bitwise with the host sum; on GPUs within the
//! tolerances of docs/design.md, "Testing and validation".

use cubecl::prelude::*;
use gridcl_device::{DeviceBuffer, KernelKind, LaunchPlan};
use gridcl_testing::test_client;

/// Lengths of "Testing and validation": the empty case, one element, around a power of two,
/// and a large length that is not a multiple of any vector or cube size.
const LENGTHS: [usize; 6] = [0, 1, 15, 16, 17, 1_000_003];

/// A precision the test runs in. Host arithmetic goes through f64, never through `Self`'s
/// own methods (CLAUDE.md, "Host maths").
trait TestFloat: Float + CubeElement + Copy + std::fmt::Debug {
    const NAME: &'static str;
    /// Relative tolerance on GPUs.
    const TOLERANCE: f64;
    fn widen(self) -> f64;
    fn narrow(x: f64) -> Self;
    fn to_bits_u64(self) -> u64;
}

impl TestFloat for f32 {
    const NAME: &'static str = "f32";
    const TOLERANCE: f64 = 1e-5;
    fn widen(self) -> f64 {
        f64::from(self)
    }
    fn narrow(x: f64) -> Self {
        x as f32
    }
    fn to_bits_u64(self) -> u64 {
        u64::from(self.to_bits())
    }
}

impl TestFloat for f64 {
    const NAME: &'static str = "f64";
    const TOLERANCE: f64 = 1e-12;
    fn widen(self) -> f64 {
        self
    }
    fn narrow(x: f64) -> Self {
        x
    }
    fn to_bits_u64(self) -> u64 {
        self.to_bits()
    }
}

/// `out = a + b` on the first `len` elements, in the kernel loop of Listing B5: each unit adds
/// `items_per_unit` consecutive elements (vector size 1), skipping indices at or above
/// `num_vectors` (the padded length) and `len`, so the padding is never touched.
#[cube(launch_unchecked)]
fn add_kernel<F: Float>(
    a: &[F],
    b: &[F],
    out: &mut [F],
    len: u32,
    num_vectors: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize && v < len as usize {
            out[v] = a[v] + b[v];
        }
    }
}

/// The value of `GRIDCL_TEST_DEVICE`, empty when unset.
fn test_device() -> String {
    std::env::var("GRIDCL_TEST_DEVICE").unwrap_or_default()
}

/// Whether the device is the CPU runtime, where results are compared bitwise.
fn on_cpu_runtime() -> bool {
    matches!(test_device().as_str(), "" | "cpu")
}

/// Deterministic inputs whose sums mostly round, so the comparison checks the rounding.
fn inputs<F: TestFloat>(len: usize) -> (Vec<F>, Vec<F>) {
    let a = (0..len)
        .map(|i| F::narrow((i % 8191) as f64 * 0.1 - 300.0))
        .collect();
    let b = (0..len)
        .map(|i| F::narrow(((i * 7) % 4099) as f64 / 3.0))
        .collect();
    (a, b)
}

/// The host sum. Rounding the f64 sum to f32 equals the f32 sum, since f64 carries more than
/// twice f32's precision.
fn host_add<F: TestFloat>(a: &[F], b: &[F]) -> Vec<F> {
    a.iter()
        .zip(b)
        .map(|(&x, &y)| F::narrow(x.widen() + y.widen()))
        .collect()
}

/// Runs the kernel on `len` elements and returns the device result.
fn device_add<F: TestFloat>(client: &Client, a: &[F], b: &[F]) -> Vec<F> {
    let len = a.len();
    assert_eq!(b.len(), len, "inputs of different lengths");
    assert!(len > 0, "an empty add is not launched");
    let len_u32 = u32::try_from(len).expect("the kernel takes its length as u32");

    // Three vectors live (a, b and the sum); max_vector 1 keeps the kernel scalar, whatever
    // GRIDCL_VECTOR_SIZE asks for.
    let plan = LaunchPlan::elementwise(client, KernelKind::Copy, len, size_of::<F>(), 1, 3)
        .expect("no launch plan for the add");
    assert_eq!(plan.vector_size, 1, "the add kernel is scalar");
    let CubeCount::Static(x, y, z) = plan.cube_count else {
        panic!("LaunchPlan returned a dynamic cube count");
    };
    let covered = [x, y, z, plan.cube_dim.num_elems(), plan.items_per_unit]
        .into_iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(n as usize))
        .expect("covered vector count overflows usize");
    assert!(
        plan.num_vectors as usize >= len && covered >= plan.num_vectors as usize,
        "the plan covers {covered} of {} vectors, for {len} elements",
        plan.num_vectors
    );

    let a = DeviceBuffer::from_host(client, a).expect("upload of a failed");
    let b = DeviceBuffer::from_host(client, b).expect("upload of b failed");
    let out = DeviceBuffer::<F>::uninit(client, len).expect("allocation of out failed");
    // SAFETY: the three buffers hold `len` elements of `F` each, padded to VECTOR_PAD, and
    // `arg()` declares `len`. The kernel reads and writes only indices `v < len`, which it is
    // given explicitly (in CubeCL 0.11 the handle size, not the declared length, bounds the
    // slice, ADR-0009). `out` is a distinct allocation from `a` and `b`, so nothing aliases a
    // written buffer. The launch shape is the plan's, checked above to cover every index.
    unsafe {
        add_kernel::launch_unchecked::<F>(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            a.arg(),
            b.arg(),
            out.arg(),
            len_u32,
            plan.num_vectors,
            plan.items_per_unit,
        );
    }
    out.to_host().expect("kernel or read-back failed")
}

/// Panics at the first element where `got` differs from `want`: bitwise on the CPU runtime,
/// beyond `F::TOLERANCE` relative elsewhere.
fn assert_matches<F: TestFloat>(got: &[F], want: &[F], len: usize) {
    assert_eq!(got.len(), want.len(), "{} length {len}", F::NAME);
    let bitwise = on_cpu_runtime();
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        let ok = if bitwise {
            g.to_bits_u64() == w.to_bits_u64()
        } else {
            (g.widen() - w.widen()).abs() <= F::TOLERANCE * w.widen().abs()
        };
        assert!(
            ok,
            "{} length {len}: element {i} is {g:?}, want {w:?} ({})",
            F::NAME,
            if bitwise {
                "bitwise"
            } else {
                "relative tolerance"
            }
        );
    }
}

fn check_add<F: TestFloat>(client: &Client) {
    for len in LENGTHS {
        let (a, b) = inputs::<F>(len);
        let want = host_add(&a, &b);
        if len == 0 {
            assert!(want.is_empty());
            continue;
        }
        let got = device_add(client, &a, &b);
        assert_matches(&got, &want, len);
    }
}

#[test]
fn elementwise_add_f32() {
    let client = test_client();
    check_add::<f32>(&client);
}

#[test]
fn elementwise_add_f64() {
    let client = test_client();
    let supported = client.properties().supports_type(f64::cube_type());
    let device = test_device();
    if matches!(device.as_str(), "" | "cpu" | "cuda") {
        assert!(
            supported,
            "f64 must be supported on GRIDCL_TEST_DEVICE={device:?}"
        );
    }
    if !supported {
        println!("elementwise_add_f64 skipped: GRIDCL_TEST_DEVICE={device} does not support f64");
        return;
    }
    check_add::<f64>(&client);
}
