//! M0 acceptance test: an element-wise add on the device chosen by `GRIDCL_TEST_DEVICE`.
//!
//! The kernel is test code and scalar. Its cube count comes from `calculate_cube_count_elemwise`
//! rather than `LaunchPlan`, which arrives in M2 (docs/plans/M0.md, question 2).
//!
//! On the CPU runtime the result is compared bitwise with the host sum; on GPUs within the
//! tolerances of docs/design.md, "Testing and validation".

use cubecl::calculate_cube_count_elemwise;
use cubecl::prelude::*;
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

/// `out = a + b`; units past the end do nothing.
#[cube(launch_unchecked)]
fn add_kernel<F: Float>(a: &[F], b: &[F], out: &mut [F]) {
    if ABSOLUTE_POS < out.len() {
        out[ABSOLUTE_POS] = a[ABSOLUTE_POS] + b[ABSOLUTE_POS];
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
    let bytes = len
        .checked_mul(size_of::<F>())
        .expect("buffer size overflows usize");
    let max_page = client.properties().memory.max_page_size;
    assert!(
        u64::try_from(bytes).is_ok_and(|bytes| bytes <= max_page),
        "a buffer of {bytes} bytes exceeds max_page_size {max_page}"
    );
    u32::try_from(len).expect("the kernel indexes in u32");

    let cube_dim = CubeDim::new(client, len);
    let cube_count = calculate_cube_count_elemwise(client, len, cube_dim);
    let CubeCount::Static(x, y, z) = cube_count else {
        panic!("calculate_cube_count_elemwise returned a dynamic count");
    };
    let launched = [x, y, z, cube_dim.num_elems()]
        .into_iter()
        .try_fold(1usize, |acc, n| acc.checked_mul(n as usize))
        .expect("launched unit count overflows usize");
    assert!(
        launched >= len,
        "the launch covers {launched} units, fewer than {len}"
    );

    let a_handle = client.create_from_slice(F::as_bytes(a));
    let b_handle = client.create_from_slice(F::as_bytes(b));
    let out_handle = client.empty(bytes);
    // SAFETY: each buffer holds `len` elements of `F` (`bytes = len * size_of::<F>()`), the
    // length declared to from_raw_parts. The kernel reads and writes index ABSOLUTE_POS only
    // when it is below out.len(), which equals the other two lengths, and has no loops.
    unsafe {
        add_kernel::launch_unchecked::<F>(
            client,
            cube_count,
            cube_dim,
            BufferArg::from_raw_parts(a_handle, len),
            BufferArg::from_raw_parts(b_handle, len),
            BufferArg::from_raw_parts(out_handle.clone(), len),
        );
    }
    let bytes = client
        .read_one(out_handle)
        .expect("kernel or read-back failed");
    let got = F::from_bytes(&bytes);
    got.get(..len)
        .unwrap_or_else(|| panic!("read back {} elements, fewer than {len}", got.len()))
        .to_vec()
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
