//! Device-primitive families: Fill (f32, f64, u32), Copy (u32) and Reduce (sum; f32, f64,
//! u32). Each times the primitive itself (docs/plans/M3s.md, "One timed region per family").

use std::any::Any;
use std::fmt::Debug;

use cubecl::prelude::Client;
use gridcl_device::{DeviceBuffer, DeviceScalar, KernelKind, Real, ReduceOp, copy, fill, reduce};

use crate::protocol::Workload;
use crate::record::Scalar;

/// Seed of the random inputs.
const SEED: u64 = 0x6772_6964_636c_0003;

/// The workload of `kind` in `scalar` at size `n`, or `None` if this module does not measure it.
pub fn workload(
    client: &Client,
    kind: KernelKind,
    scalar: Scalar,
    n: usize,
) -> Option<Result<Box<dyn Workload>, String>> {
    fn boxed<W: Workload + 'static>(
        w: gridcl_device::Result<W>,
    ) -> Result<Box<dyn Workload>, String> {
        w.map(|w| Box::new(w) as Box<dyn Workload>)
            .map_err(|e| e.to_string())
    }
    Some(match (kind, scalar) {
        (KernelKind::Fill, Scalar::F32) => boxed(Fill::<f32>::new(client, n)),
        (KernelKind::Fill, Scalar::F64) => boxed(Fill::<f64>::new(client, n)),
        (KernelKind::Fill, Scalar::U32) => boxed(Fill::<u32>::new(client, n)),
        (KernelKind::Copy, Scalar::U32) => boxed(CopyWords::new(client, n)),
        (KernelKind::Reduce, Scalar::F32) => boxed(Reduce::<f32>::new(client, n)),
        (KernelKind::Reduce, Scalar::F64) => boxed(Reduce::<f64>::new(client, n)),
        (KernelKind::Reduce, Scalar::U32) => boxed(Reduce::<u32>::new(client, n)),
        _ => return None,
    })
}

/// A scalar type of the device families, with its host reference arithmetic.
pub(crate) trait Elem: DeviceScalar + Copy + PartialEq + Debug + 'static {
    /// The small integer `k` in this type (exact).
    fn small(k: u32) -> Self;
    /// A summand of the Reduce input from the random word `r`: `r` itself in u32, its low bit
    /// in floats, so that every partial sum is exact.
    fn summand(r: u32) -> Self;
    /// Whether `got` is the sum of `x` within the tolerance of CLAUDE.md (1e-5 relative in f32,
    /// 1e-12 in f64; wrapping and exact in u32). Float sums are formed in f64 with `widen`.
    fn sum_matches(x: &[Self], got: Self) -> bool;
}

fn float_sum_matches<F: Real>(x: &[F], got: F, tolerance: f64) -> bool {
    let want: f64 = x.iter().map(|&v| v.widen()).sum();
    (got.widen() - want).abs() <= tolerance * want.abs().max(1.0)
}

impl Elem for f32 {
    fn summand(r: u32) -> Self {
        (r & 1) as f32
    }
    fn small(k: u32) -> Self {
        k as f32
    }
    fn sum_matches(x: &[Self], got: Self) -> bool {
        float_sum_matches(x, got, 1e-5)
    }
}

impl Elem for f64 {
    fn summand(r: u32) -> Self {
        f64::from(r & 1)
    }
    fn small(k: u32) -> Self {
        f64::from(k)
    }
    fn sum_matches(x: &[Self], got: Self) -> bool {
        float_sum_matches(x, got, 1e-12)
    }
}

impl Elem for u32 {
    fn summand(r: u32) -> Self {
        r
    }
    fn small(k: u32) -> Self {
        k
    }
    fn sum_matches(x: &[Self], got: Self) -> bool {
        x.iter().fold(0u32, |a, &v| a.wrapping_add(v)) == got
    }
}

/// SplitMix64, a small deterministic generator.
pub(crate) struct SplitMix(pub(crate) u64);

impl SplitMix {
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    pub(crate) fn next_u32(&mut self) -> u32 {
        (self.next_u64() >> 32) as u32
    }
}

fn host_error(e: gridcl_device::GridError) -> String {
    e.to_string()
}

/// `fill(x, v)` with a value that changes every call, so the check sees the warm-up's write.
struct Fill<T: Elem> {
    x: DeviceBuffer<T>,
    calls: u32,
}

impl<T: Elem> Fill<T> {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        Ok(Fill {
            x: DeviceBuffer::uninit(client, n)?,
            calls: 0,
        })
    }

    fn value(calls: u32) -> T {
        T::small(1 + calls % 7)
    }
}

impl<T: Elem> Workload for Fill<T> {
    fn items(&self) -> usize {
        self.x.len()
    }

    fn bytes(&self) -> Option<usize> {
        Some(self.x.len() * size_of::<T>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        self.calls = self.calls.wrapping_add(1);
        fill(&mut self.x, Self::value(self.calls)).map_err(host_error)?;
        Ok(Box::new(()))
    }

    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        let want = Self::value(self.calls);
        let got = self.x.to_host().map_err(host_error)?;
        match got.iter().position(|&v| v != want) {
            None => Ok(()),
            Some(i) => Err(format!("fill: element {i} is {:?}, not {want:?}", got[i])),
        }
    }
}

/// `copy(src, dst)` of random u32 words; `reset` zeroes `dst`.
struct CopyWords {
    src_host: Vec<u32>,
    src: DeviceBuffer<u32>,
    dst: DeviceBuffer<u32>,
}

impl CopyWords {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED);
        let src_host: Vec<u32> = (0..n).map(|_| rng.next_u32()).collect();
        Ok(CopyWords {
            src: DeviceBuffer::from_host(client, &src_host)?,
            dst: DeviceBuffer::uninit(client, n)?,
            src_host,
        })
    }
}

impl Workload for CopyWords {
    fn items(&self) -> usize {
        self.src.len()
    }

    /// One read and one write per word.
    fn bytes(&self) -> Option<usize> {
        Some(2 * self.src.len() * size_of::<u32>())
    }

    fn reset(&mut self) -> Result<(), String> {
        fill(&mut self.dst, 0).map_err(host_error)
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        copy(&self.src, &mut self.dst).map_err(host_error)?;
        Ok(Box::new(()))
    }

    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        let got = self.dst.to_host().map_err(host_error)?;
        match got.iter().zip(&self.src_host).position(|(g, w)| g != w) {
            None if got.len() == self.src_host.len() => Ok(()),
            None => Err(format!(
                "copy: {} words, not {}",
                got.len(),
                self.src_host.len()
            )),
            Some(i) => Err(format!("copy: word {i} differs")),
        }
    }
}

/// `reduce(x, Sum)` of [`Elem::summand`]s.
struct Reduce<T: Elem> {
    x_host: Vec<T>,
    x: DeviceBuffer<T>,
}

impl<T: Elem> Reduce<T> {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 1);
        let x_host: Vec<T> = (0..n).map(|_| T::summand(rng.next_u32())).collect();
        Ok(Reduce {
            x: DeviceBuffer::from_host(client, &x_host)?,
            x_host,
        })
    }
}

impl<T: Elem> Workload for Reduce<T> {
    fn items(&self) -> usize {
        self.x.len()
    }

    fn bytes(&self) -> Option<usize> {
        Some(self.x.len() * size_of::<T>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let sum = reduce(&self.x, ReduceOp::Sum).map_err(host_error)?;
        Ok(Box::new(sum))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let sum = out
            .downcast::<DeviceBuffer<T>>()
            .map_err(|_| "reduce: unexpected output".to_owned())?;
        let got = sum.to_host().map_err(host_error)?;
        match got.as_slice() {
            [v] if T::sum_matches(&self.x_host, *v) => Ok(()),
            _ => Err(format!(
                "reduce: sum {got:?} differs from the host reference"
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn host_sums_follow_the_tolerances() {
        assert!(f32::sum_matches(&[1.0, 0.0, 1.0], 2.0));
        assert!(!f32::sum_matches(&[1.0, 0.0, 1.0], 2.001));
        assert!(f64::sum_matches(&[0.5; 4], 2.0));
        assert!(!f64::sum_matches(&[0.5; 4], 2.0 + 1e-9));
        assert!(u32::sum_matches(&[u32::MAX, 2], 1));
        assert!(!u32::sum_matches(&[u32::MAX, 2], 0));
    }
}
