//! Device-primitive families, each timing the smallest call whose launches all take the
//! family's plan (docs/plans/M3s.md, "One timed region per family"):
//!
//! - Fill (f32, f64, u32), Copy (u32), Reduce (sum; f32, f64, u32), Gather (f32, f64, u32,
//!   random in-range indices): the primitive;
//! - ScanTile (u32): `inclusive_scan` then `exclusive_scan`, one timed region;
//! - SortDigits (u32): `radix_sort` of two-word keys; its inner `exclusive_scan` keeps its own
//!   vector size, because the override is per kind;
//! - Scalar, for its CPU blocks only (ADR-0015, item 7): `histogram` (u32) and
//!   `segmented_reduce` (sum, f64).

use std::any::Any;
use std::fmt::Debug;

use cubecl::prelude::Client;
use gridcl_device::{
    DeviceBuffer, DeviceScalar, KernelKind, Real, ReduceOp, SortKeys, copy, exclusive_scan, fill,
    gather, histogram, inclusive_scan, radix_sort, reduce, segmented_reduce,
};

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
        (KernelKind::ScanTile, Scalar::U32) => boxed(Scan::new(client, n)),
        (KernelKind::SortDigits, Scalar::U32) => boxed(Sort::new(client, n)),
        (KernelKind::Gather, Scalar::F32) => boxed(Gather::<f32>::new(client, n)),
        (KernelKind::Gather, Scalar::F64) => boxed(Gather::<f64>::new(client, n)),
        (KernelKind::Gather, Scalar::U32) => boxed(Gather::<u32>::new(client, n)),
        (KernelKind::Scalar, Scalar::U32) => boxed(Histogram::new(client, n)),
        (KernelKind::Scalar, Scalar::F64) => boxed(Segmented::new(client, n)),
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

/// Scan inputs are below this bound, so the sum of 10^7 values does not wrap.
const SCAN_BOUND: u32 = 256;

/// `inclusive_scan(x)` and `exclusive_scan(x)` of random u32 values below `SCAN_BOUND`.
struct Scan {
    x_host: Vec<u32>,
    x: DeviceBuffer<u32>,
}

impl Scan {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 2);
        let x_host: Vec<u32> = (0..n).map(|_| rng.next_u32() % SCAN_BOUND).collect();
        Ok(Scan {
            x: DeviceBuffer::from_host(client, &x_host)?,
            x_host,
        })
    }
}

impl Workload for Scan {
    fn items(&self) -> usize {
        self.x.len()
    }

    /// Each scan reads n and writes n (+ 1) words.
    fn bytes(&self) -> Option<usize> {
        Some(4 * self.x.len() * size_of::<u32>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let inclusive = inclusive_scan(&self.x).map_err(host_error)?;
        let exclusive = exclusive_scan(&self.x).map_err(host_error)?;
        Ok(Box::new((inclusive, exclusive)))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let (inclusive, exclusive) = *out
            .downcast::<(DeviceBuffer<u32>, DeviceBuffer<u32>)>()
            .map_err(|_| "scan: unexpected output".to_owned())?;
        let mut want = Vec::with_capacity(self.x_host.len() + 1);
        want.push(0u32);
        for &v in &self.x_host {
            want.push(want.last().unwrap().wrapping_add(v));
        }
        let first_difference = |got: &[u32], want: &[u32]| {
            (got.len() != want.len())
                .then_some(got.len().min(want.len()))
                .or_else(|| got.iter().zip(want).position(|(g, w)| g != w))
        };
        let got = inclusive.to_host().map_err(host_error)?;
        if let Some(i) = first_difference(&got, &want[1..]) {
            return Err(format!("inclusive scan: first difference at {i}"));
        }
        let got = exclusive.to_host().map_err(host_error)?;
        if let Some(i) = first_difference(&got, &want) {
            return Err(format!("exclusive scan: first difference at {i}"));
        }
        Ok(())
    }
}

/// Words per sort key, as in M2's sort benchmark (docs/perf/m2-primitives.md).
const SORT_WORDS: usize = 2;

/// `radix_sort` of random two-word keys with an iota payload.
struct Sort {
    keys_host: Vec<u32>,
    keys: SortKeys,
    payload: DeviceBuffer<u32>,
}

impl Sort {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 3);
        let keys_host: Vec<u32> = (0..n * SORT_WORDS).map(|_| rng.next_u32()).collect();
        let payload: Vec<u32> = (0..n)
            .map(|i| u32::try_from(i).map_err(|_| gridcl_device::GridError::TooLarge))
            .collect::<gridcl_device::Result<_>>()?;
        Ok(Sort {
            keys: SortKeys::from_host(client, &keys_host, SORT_WORDS)?,
            payload: DeviceBuffer::from_host(client, &payload)?,
            keys_host,
        })
    }
}

impl Workload for Sort {
    fn items(&self) -> usize {
        self.keys.len()
    }

    /// Several passes over keys and payload; not a single stream.
    fn bytes(&self) -> Option<usize> {
        None
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let sorted = radix_sort(&self.keys, &self.payload).map_err(host_error)?;
        Ok(Box::new(sorted))
    }

    /// Against the host's stable sort of the same keys.
    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let (sorted, perm) = *out
            .downcast::<(SortKeys, DeviceBuffer<u32>)>()
            .map_err(|_| "sort: unexpected output".to_owned())?;
        let w = SORT_WORDS;
        let key = |i: usize| &self.keys_host[i * w..(i + 1) * w];
        let mut want: Vec<u32> = (0..self.keys.len() as u32).collect();
        want.sort_by(|&a, &b| key(a as usize).cmp(key(b as usize)));
        if perm.to_host().map_err(host_error)? != want {
            return Err("sort: the payload is not the stable sort's permutation".to_owned());
        }
        let want_keys: Vec<u32> = want
            .iter()
            .flat_map(|&i| key(i as usize).to_vec())
            .collect();
        if sorted.to_host().map_err(host_error)? != want_keys {
            return Err("sort: the sorted keys differ".to_owned());
        }
        Ok(())
    }
}

/// `gather(src, idx)` with `n` random values and `n` random in-range indices.
struct Gather<T: Elem> {
    src_host: Vec<T>,
    idx_host: Vec<u32>,
    src: DeviceBuffer<T>,
    idx: DeviceBuffer<u32>,
}

impl<T: Elem> Gather<T> {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 4);
        let src_host: Vec<T> = (0..n).map(|_| T::small(rng.next_u32() % 1000)).collect();
        let len = n as u64;
        let idx_host: Vec<u32> = (0..n).map(|_| (rng.next_u64() % len) as u32).collect();
        Ok(Gather {
            src: DeviceBuffer::from_host(client, &src_host)?,
            idx: DeviceBuffer::from_host(client, &idx_host)?,
            src_host,
            idx_host,
        })
    }
}

impl<T: Elem> Workload for Gather<T> {
    fn items(&self) -> usize {
        self.idx.len()
    }

    /// Reads an index and a value, writes a value.
    fn bytes(&self) -> Option<usize> {
        Some(self.idx.len() * (size_of::<u32>() + 2 * size_of::<T>()))
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let out = gather(&self.src, &self.idx).map_err(host_error)?;
        Ok(Box::new(out))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let out = out
            .downcast::<DeviceBuffer<T>>()
            .map_err(|_| "gather: unexpected output".to_owned())?;
        let got = out.to_host().map_err(host_error)?;
        let want = self.idx_host.iter().map(|&i| self.src_host[i as usize]);
        match got.iter().zip(want).position(|(&g, w)| g != w) {
            None if got.len() == self.idx_host.len() => Ok(()),
            None => Err(format!(
                "gather: {} values, not {}",
                got.len(),
                self.idx_host.len()
            )),
            Some(i) => Err(format!("gather: value {i} differs")),
        }
    }
}

/// Bins of the histogram workload.
const HISTOGRAM_BINS: u32 = 256;

/// `histogram(values, HISTOGRAM_BINS)` of random values below the bin count (Scalar, 4 bytes).
struct Histogram {
    values_host: Vec<u32>,
    values: DeviceBuffer<u32>,
}

impl Histogram {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 5);
        let values_host: Vec<u32> = (0..n).map(|_| rng.next_u32() % HISTOGRAM_BINS).collect();
        Ok(Histogram {
            values: DeviceBuffer::from_host(client, &values_host)?,
            values_host,
        })
    }
}

impl Workload for Histogram {
    fn items(&self) -> usize {
        self.values.len()
    }

    fn bytes(&self) -> Option<usize> {
        Some(self.values.len() * size_of::<u32>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let counts = histogram(&self.values, HISTOGRAM_BINS).map_err(host_error)?;
        Ok(Box::new(counts))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let counts = out
            .downcast::<DeviceBuffer<u32>>()
            .map_err(|_| "histogram: unexpected output".to_owned())?;
        let mut want = vec![0u32; HISTOGRAM_BINS as usize];
        for &v in &self.values_host {
            want[v as usize] += 1;
        }
        if counts.to_host().map_err(host_error)? != want {
            return Err("histogram: the counts differ".to_owned());
        }
        Ok(())
    }
}

/// Longest segment of the segmented-reduce workload; lengths are uniform in 0..=this.
const SEGMENT_MAX: u64 = 16;

/// `segmented_reduce(values, offsets, Sum)` of `n` f64 values in random segments of 0 to
/// `SEGMENT_MAX` values (Scalar, 8 bytes). Values are small integers, so the sums are exact.
struct Segmented {
    values_host: Vec<f64>,
    offsets_host: Vec<u32>,
    values: DeviceBuffer<f64>,
    offsets: DeviceBuffer<u32>,
}

impl Segmented {
    fn new(client: &Client, n: usize) -> gridcl_device::Result<Self> {
        let mut rng = SplitMix(SEED ^ 6);
        let values_host: Vec<f64> = (0..n).map(|_| f64::small(rng.next_u32() % 100)).collect();
        let mut offsets_host = vec![0u32];
        let mut end = 0usize;
        while end < n {
            end = (end + (rng.next_u64() % (SEGMENT_MAX + 1)) as usize).min(n);
            offsets_host.push(u32::try_from(end).map_err(|_| gridcl_device::GridError::TooLarge)?);
        }
        Ok(Segmented {
            values: DeviceBuffer::from_host(client, &values_host)?,
            offsets: DeviceBuffer::from_host(client, &offsets_host)?,
            values_host,
            offsets_host,
        })
    }
}

impl Workload for Segmented {
    fn items(&self) -> usize {
        self.values.len()
    }

    /// Reads every value and the offsets, writes one sum per segment.
    fn bytes(&self) -> Option<usize> {
        let segments = self.offsets.len() - 1;
        Some(
            self.values.len() * size_of::<f64>()
                + self.offsets.len() * size_of::<u32>()
                + segments * size_of::<f64>(),
        )
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let sums =
            segmented_reduce(&self.values, &self.offsets, ReduceOp::Sum).map_err(host_error)?;
        Ok(Box::new(sums))
    }

    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let sums = out
            .downcast::<DeviceBuffer<f64>>()
            .map_err(|_| "segmented reduce: unexpected output".to_owned())?;
        let got = sums.to_host().map_err(host_error)?;
        let want = self.offsets_host.windows(2).map(|w| {
            let segment = &self.values_host[w[0] as usize..w[1] as usize];
            (segment, segment.iter().map(|v| v.widen()).sum::<f64>())
        });
        if got.len() != self.offsets_host.len() - 1 {
            return Err(format!("segmented reduce: {} sums", got.len()));
        }
        match got
            .iter()
            .zip(want)
            .position(|(&g, (segment, _))| !f64::sum_matches(segment, g))
        {
            None => Ok(()),
            Some(i) => Err(format!("segmented reduce: segment {i} differs")),
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
