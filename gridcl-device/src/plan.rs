//! `LaunchPlan` (Listing S2, ADR-0010).

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;

use crate::buffer::pad_len;
use crate::policy::is_vector_size;
use crate::{
    CpuUnits, DeviceClass, GridError, KernelKind, Result, VectorPolicy, VectorSizeOverride, testing,
};

/// Units per cube on GPUs, or `max_units_per_cube` if that is smaller.
const GPU_CUBE_DIM: u32 = 256;
/// Cache-line size CPU blocks are a whole number of.
const CPU_LINE_BYTES: usize = 128;
/// Buffers a library kernel may bind (docs/design.md, "Device limits").
const MAX_KERNEL_BUFFERS: u32 = 6;

/// Shape of one launch on the current device. Not `Copy`: `CubeCount` is not `Copy` in
/// CubeCL 0.11 (ADR-0010).
///
/// Kernels compute `first = ABSOLUTE_POS * items_per_unit` and loop over `items_per_unit`
/// vectors, checking `v < num_vectors` (Listing B5). A unit owns `items_per_unit ·
/// vector_size` consecutive scalars, and a cube owns `cube_dim` times as many.
///
/// # Example
///
/// ```
/// use gridcl_device::{KernelKind, LaunchPlan};
///
/// let client = gridcl_testing::test_client();
/// let plan = LaunchPlan::elementwise(&client, KernelKind::Fill, 1_000_000, 4, 16, 1).unwrap();
/// let cubes = match plan.cube_count {
///     cubecl::prelude::CubeCount::Static(x, y, z) => (x * y * z) as u64,
///     _ => unreachable!(),
/// };
/// let covered = cubes * plan.cube_dim.num_elems() as u64 * plan.items_per_unit as u64;
/// assert!(covered >= plan.num_vectors as u64);
/// ```
#[derive(Clone, Debug)]
pub struct LaunchPlan {
    pub cube_count: CubeCount,
    pub cube_dim: CubeDim,
    /// Lanes per vector; 1 is a scalar launch.
    pub vector_size: usize,
    /// Consecutive vectors each unit processes: 1 on GPUs, a block on CPUs.
    pub items_per_unit: u32,
    /// Vectors covered, padding included.
    pub num_vectors: u32,
}

impl LaunchPlan {
    /// `items` scalar items; `max_vector` the widest size the data layout allows (a divisor of
    /// `VECTOR_PAD`); `live` the vectors the kernel keeps live, for the register budget.
    ///
    /// - **Vector size:** the largest power of two not above the policy's size, `max_vector`,
    ///   the device's `hardware.max_vector_size` (ADR-0011) and the widest of
    ///   `vector_sizes_in_registers(elem_bytes, live)`; overridden by `GRIDCL_VECTOR_SIZE`,
    ///   `testing::with_vector_size` or `testing::with_kind_vector_size` (see
    ///   [`VectorSizeOverride`]).
    /// - **GPUs:** `cube_dim` is 256, or `max_units_per_cube` (at most `max_cube_dim.0`) if
    ///   that is smaller; `items_per_unit` is 1.
    /// - **CPUs:** `cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`,
    ///   ADR-0011), one unit per hardware thread, or the performance-core count where
    ///   `VectorPolicy::cpu_units(kind, elem_bytes)` (or `testing::with_cpu_units`) says
    ///   `Performance` and the CPU reports a smaller count (macOS; ADR-0017). Each unit takes
    ///   blocks of `VectorPolicy::cpu_block_bytes(kind, elem_bytes)` of the widest stream
    ///   (4, 16 or 64 KiB), rounded to whole 128-byte lines (ADR-0015).
    /// - **Cube count:** x up to `max_cube_count.0`, then y, then z (ADR-0010).
    ///
    /// # Errors
    ///
    /// `TooLarge` if the padded length, the vector count or the launched units overflow u32,
    /// or if no cube count within `max_cube_count` covers the vectors.
    ///
    /// # Panics
    ///
    /// If `max_vector` is not a power of two dividing `VECTOR_PAD`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{KernelKind, LaunchPlan, VECTOR_PAD};
    ///
    /// let client = gridcl_testing::test_client();
    /// // 100 items pad to 112 = 7 · VECTOR_PAD scalars, whatever the vector size.
    /// let plan = LaunchPlan::elementwise(&client, KernelKind::Copy, 100, 4, VECTOR_PAD, 2).unwrap();
    /// assert_eq!(plan.num_vectors as usize * plan.vector_size, 7 * VECTOR_PAD);
    /// ```
    pub fn elementwise(
        client: &Client,
        kind: KernelKind,
        items: usize,
        elem_bytes: usize,
        max_vector: usize,
        live: usize,
    ) -> Result<Self> {
        let props = testing::properties(client);
        let plan = plan_for(
            &props,
            kind,
            items,
            elem_bytes,
            max_vector,
            live,
            VectorSizeOverride::current(kind),
        )?;
        testing::record(kind, &plan);
        Ok(plan)
    }

    /// The number of cubes launched: the product of the cube count's axes.
    #[allow(dead_code)] // The multi-phase primitives of T4 onwards call it.
    pub(crate) fn num_cubes(&self) -> usize {
        match self.cube_count {
            CubeCount::Static(x, y, z) => x as usize * y as usize * z as usize,
            CubeCount::Dynamic(_) => unreachable!("plans have static cube counts"),
        }
    }
}

/// The plan for `items` items under `props` and the vector-size override `vector`.
fn plan_for(
    props: &DeviceProperties,
    kind: KernelKind,
    items: usize,
    elem_bytes: usize,
    max_vector: usize,
    live: usize,
    vector: VectorSizeOverride,
) -> Result<LaunchPlan> {
    assert!(
        is_vector_size(max_vector),
        "max_vector {max_vector} is not a power of two dividing VECTOR_PAD"
    );
    assert!(elem_bytes > 0, "elem_bytes is 0");
    let hw = &props.hardware;
    let class = DeviceClass::from_properties(props);
    let n = vector_size(props, class, kind, elem_bytes, max_vector, live, vector);

    let num_vectors = pad_len(items)? / n;
    let num_vectors_u32 = u32::try_from(num_vectors).map_err(|_| GridError::TooLarge)?;

    let policy = VectorPolicy::for_class(class);
    let cpu_block = policy.cpu_block_bytes(kind, elem_bytes);
    let (cube_dim, items_per_unit) = if let Some(block) = cpu_block {
        let bytes = n.checked_mul(elem_bytes).ok_or(GridError::TooLarge)?;
        let all = hw.max_units_per_cube.min(hw.max_cube_dim.0);
        let units = match policy.cpu_units(kind, elem_bytes) {
            Some(CpuUnits::Performance) => testing::performance_cores()
                .filter(|&p| p >= 1 && p < all)
                .unwrap_or(all),
            _ => all,
        };
        (units, cpu_items_per_unit(bytes, block))
    } else {
        (
            GPU_CUBE_DIM
                .min(hw.max_units_per_cube)
                .min(hw.max_cube_dim.0),
            1,
        )
    };
    let cube_dim = cube_dim.max(1);
    let per_cube = (cube_dim as usize)
        .checked_mul(items_per_unit)
        .ok_or(GridError::TooLarge)?;
    let cubes = num_vectors.div_ceil(per_cube);
    let (x, y, z) = spread_cubes(cubes, hw.max_cube_count)?;

    let covered = [x as usize, y as usize, z as usize, per_cube]
        .into_iter()
        .try_fold(1usize, |acc, f| acc.checked_mul(f))
        .ok_or(GridError::TooLarge)?;
    u32::try_from(covered).map_err(|_| GridError::TooLarge)?;
    Ok(LaunchPlan {
        cube_count: CubeCount::Static(x, y, z),
        cube_dim: CubeDim::new_1d(cube_dim),
        vector_size: n,
        items_per_unit: u32::try_from(items_per_unit).map_err(|_| GridError::TooLarge)?,
        num_vectors: num_vectors_u32,
    })
}

/// The vector size of a plan: the override's, else the policy rule, clamped by `max_vector`
/// and the device's `max_vector_size`; always a power of two dividing `VECTOR_PAD`.
fn vector_size(
    props: &DeviceProperties,
    class: DeviceClass,
    kind: KernelKind,
    elem_bytes: usize,
    max_vector: usize,
    live: usize,
    vector: VectorSizeOverride,
) -> usize {
    let wanted = match vector {
        VectorSizeOverride::Scalar => 1,
        VectorSizeOverride::Fixed(n) => n,
        VectorSizeOverride::Widest => props
            .io_optimized_vector_sizes(elem_bytes)
            .next()
            .unwrap_or(1),
        VectorSizeOverride::Policy => {
            let policy = VectorPolicy::for_class(class).vector_size(kind, elem_bytes);
            let registers = props
                .vector_sizes_in_registers(elem_bytes, live)
                .next()
                .unwrap_or(1);
            policy.min(registers)
        }
    };
    let n = wanted
        .min(max_vector)
        .min(props.hardware.max_vector_size)
        .max(1);
    1 << n.ilog2()
}

/// Vectors of `vector_bytes` bytes per CPU block: about `block_bytes`, and a whole number of
/// `CPU_LINE_BYTES` lines.
fn cpu_items_per_unit(vector_bytes: usize, block_bytes: usize) -> usize {
    // The fewest vectors that fill whole lines.
    let step = CPU_LINE_BYTES / gcd(CPU_LINE_BYTES, vector_bytes);
    let fit = block_bytes / vector_bytes;
    (fit - fit % step).max(step)
}

fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// A cube count of at least `cubes` cubes within `max` (ADR-0010): x = min(n, mx), then
/// y = min(⌈n / x⌉, my), then z = ⌈n / (x·y)⌉; `TooLarge` if z exceeds `max.2`. At least one
/// cube, so a plan is always launchable.
fn spread_cubes(cubes: usize, max: (u32, u32, u32)) -> Result<(u32, u32, u32)> {
    let n = cubes.max(1);
    let (mx, my, mz) = (max.0.max(1) as usize, max.1.max(1) as usize, max.2 as usize);
    let x = n.min(mx);
    let y = n.div_ceil(x).min(my);
    let z = n.div_ceil(x.checked_mul(y).ok_or(GridError::TooLarge)?);
    if z > mz {
        return Err(GridError::TooLarge);
    }
    let fit = |v: usize| u32::try_from(v).map_err(|_| GridError::TooLarge);
    Ok((fit(x)?, fit(y)?, fit(z)?))
}

/// Debug check of a launch's buffer count: at most six, and at most the device's
/// `max_bindings` (read through [`device_properties`](crate::device_properties)). Every kernel
/// launcher calls it once per launch; release builds skip the check.
///
/// # Panics
///
/// In debug builds, if `n` exceeds either limit.
///
/// # Example
///
/// ```
/// use gridcl_device::debug_assert_bindings;
///
/// let client = gridcl_testing::test_client();
/// debug_assert_bindings(&client, 4);
/// ```
pub fn debug_assert_bindings(client: &Client, n: u32) {
    if cfg!(debug_assertions) {
        let max = testing::properties(client).hardware.max_bindings;
        assert!(
            n <= MAX_KERNEL_BUFFERS.min(max),
            "a kernel binds {n} buffers; the limit is min(6, max_bindings = {max})"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn props() -> DeviceProperties {
        gridcl_testing::test_client().properties().clone()
    }

    #[test]
    fn fixed_gives_its_size_when_max_vector_allows() {
        let mut props = props();
        props.hardware.max_vector_size = 16;
        for n in [1, 2, 4, 8, 16] {
            let plan = plan_for(
                &props,
                KernelKind::Fill,
                1000,
                4,
                16,
                1,
                VectorSizeOverride::Fixed(n),
            )
            .unwrap();
            assert_eq!(plan.vector_size, n);
            let plan = plan_for(
                &props,
                KernelKind::Fill,
                1000,
                4,
                4,
                1,
                VectorSizeOverride::Fixed(n),
            )
            .unwrap();
            assert_eq!(plan.vector_size, n.min(4), "clamped by max_vector");
        }
    }

    #[test]
    fn widest_follows_io_optimized_vector_sizes() {
        let mut props = props();
        props.hardware.max_vector_size = 16;
        for (width, bytes) in [(128, 4), (128, 8), (256, 4), (512, 4), (512, 8)] {
            props.hardware.load_width = width;
            props.io_width_override = None;
            let widest = props.io_optimized_vector_sizes(bytes).next().unwrap();
            let plan = plan_for(
                &props,
                KernelKind::Copy,
                1000,
                bytes,
                16,
                1,
                VectorSizeOverride::Widest,
            )
            .unwrap();
            assert_eq!(
                plan.vector_size,
                widest.min(16),
                "load width {width}, {bytes} bytes"
            );
            let plan = plan_for(
                &props,
                KernelKind::Copy,
                1000,
                bytes,
                2,
                1,
                VectorSizeOverride::Widest,
            )
            .unwrap();
            assert_eq!(plan.vector_size, widest.min(2));
        }
    }

    #[test]
    fn policy_follows_the_entry_and_scalar_gives_one() {
        let props = props();
        let class = DeviceClass::from_properties(&props);
        for kind in KernelKind::ALL {
            for elem_bytes in [4, 8] {
                let entry = VectorPolicy::for_class(class).vector_size(kind, elem_bytes);
                let registers = props
                    .vector_sizes_in_registers(elem_bytes, 2)
                    .next()
                    .unwrap_or(1);
                let want = entry
                    .min(registers)
                    .min(props.hardware.max_vector_size)
                    .max(1);
                let policy = VectorSizeOverride::Policy;
                let plan = plan_for(&props, kind, 1 << 20, elem_bytes, 16, 2, policy).unwrap();
                assert_eq!(plan.vector_size, 1 << want.ilog2(), "{kind:?} {elem_bytes}");
                let scalar = VectorSizeOverride::Scalar;
                let plan = plan_for(&props, kind, 1 << 20, elem_bytes, 16, 2, scalar).unwrap();
                assert_eq!(plan.vector_size, 1, "{kind:?} {elem_bytes}");
            }
        }
    }

    #[test]
    fn cpu_blocks_are_whole_lines_of_the_policy_block() {
        let policy = VectorPolicy::for_class(DeviceClass::CpuSimd128);
        let blocks = KernelKind::ALL
            .into_iter()
            .flat_map(|kind| [4, 8].map(|bytes| policy.cpu_block_bytes(kind, bytes).unwrap()));
        for target in blocks.chain([4 << 10, 16 << 10, 64 << 10]) {
            // Vectors of 1 to 16 lanes of 4 or 8 bytes.
            for bytes in [4, 8, 16, 32, 64, 128] {
                let block = cpu_items_per_unit(bytes, target) * bytes;
                assert!(block.is_multiple_of(CPU_LINE_BYTES), "{bytes}: {block}");
                assert!(block <= target && block > target / 2, "{bytes}: {block}");
            }
        }
        for bytes in [12, 24, 256] {
            let block = cpu_items_per_unit(bytes, 16 << 10) * bytes;
            assert!(block.is_multiple_of(CPU_LINE_BYTES), "{bytes}: {block}");
            assert!(block <= 16 << 10, "{bytes}: {block}");
        }
        assert_eq!(cpu_items_per_unit(4, 16 << 10), 4096);
        assert_eq!(cpu_items_per_unit(128, 16 << 10), 128);
    }

    /// The block override sets the target at 4, 16 and 64 KiB; blocks stay whole lines.
    #[test]
    fn block_override_rounds_to_whole_lines() {
        let mut props = props();
        props.hardware.num_cpu_cores = Some(8);
        props.hardware.max_units_per_cube = 8;
        props.hardware.max_cube_dim = (8, 8, 8);
        props.hardware.load_width = 128;
        props.hardware.max_vector_size = 16;
        for kib in [4usize, 16, 64] {
            let target = kib << 10;
            for elem_bytes in [4usize, 8] {
                for n in [1usize, 2, 4, 8, 16] {
                    let plan = testing::with_cpu_block_bytes(target, || {
                        plan_for(
                            &props,
                            KernelKind::Gather,
                            10_000_000,
                            elem_bytes,
                            16,
                            1,
                            VectorSizeOverride::Fixed(n),
                        )
                    })
                    .unwrap();
                    let block = plan.items_per_unit as usize * n * elem_bytes;
                    assert!(block.is_multiple_of(CPU_LINE_BYTES), "{kib} KiB: {block}");
                    assert_eq!(block, target, "{kib} KiB, {n} × {elem_bytes} bytes");
                }
            }
            // No effect on GPUs.
            let mut gpu = props.clone();
            gpu.hardware.num_cpu_cores = None;
            let plan = testing::with_cpu_block_bytes(target, || {
                plan_for(
                    &gpu,
                    KernelKind::Gather,
                    1000,
                    4,
                    16,
                    1,
                    VectorSizeOverride::Scalar,
                )
            })
            .unwrap();
            assert_eq!(plan.items_per_unit, 1);
        }
    }

    #[test]
    fn block_override_is_bounded() {
        for bad in [0usize, 2048, 3 << 12, 128 << 10, usize::MAX] {
            let result = std::panic::catch_unwind(|| testing::with_cpu_block_bytes(bad, || ()));
            assert!(result.is_err(), "{bad} bytes accepted");
        }
        for good in [4usize << 10, 8 << 10, 32 << 10, 64 << 10] {
            testing::with_cpu_block_bytes(good, || ());
        }
    }

    /// The per-kind override wins over the scoped override (and so over the environment) for its
    /// kind only; nested calls for other kinds add to it, and every scope is restored.
    #[test]
    fn kind_override_takes_precedence() {
        use VectorSizeOverride::{Fixed, Policy, Scalar};
        let current = VectorSizeOverride::current;
        testing::with_vector_size(Fixed(4), || {
            assert_eq!(current(KernelKind::SortDigits), Fixed(4));
            testing::with_kind_vector_size(KernelKind::SortDigits, Fixed(2), || {
                assert_eq!(current(KernelKind::SortDigits), Fixed(2));
                assert_eq!(current(KernelKind::ScanTile), Fixed(4));
                testing::with_kind_vector_size(KernelKind::ScanTile, Policy, || {
                    assert_eq!(current(KernelKind::SortDigits), Fixed(2));
                    assert_eq!(current(KernelKind::ScanTile), Policy);
                    testing::with_kind_vector_size(KernelKind::SortDigits, Scalar, || {
                        assert_eq!(current(KernelKind::SortDigits), Scalar);
                    });
                    assert_eq!(current(KernelKind::SortDigits), Fixed(2));
                });
                assert_eq!(current(KernelKind::ScanTile), Fixed(4));
            });
            assert_eq!(current(KernelKind::SortDigits), Fixed(4));
        });
        assert_eq!(testing::kind_vector_override(KernelKind::SortDigits), None);
        let result = std::panic::catch_unwind(|| {
            testing::with_kind_vector_size(KernelKind::Fill, Fixed(3), || ())
        });
        assert!(result.is_err(), "Fixed(3) accepted");
    }

    /// CPU properties with 16 units per cube.
    fn cpu16() -> DeviceProperties {
        let mut props = props();
        props.hardware.num_cpu_cores = Some(16);
        props.hardware.max_units_per_cube = 16;
        props.hardware.max_cube_dim = (16, 16, 16);
        props.hardware.load_width = 128;
        props
    }

    /// The units per cube of a scalar Reduce plan under `props`.
    fn units(props: &DeviceProperties) -> u32 {
        plan_for(
            props,
            KernelKind::Reduce,
            1 << 20,
            4,
            16,
            1,
            VectorSizeOverride::Scalar,
        )
        .unwrap()
        .cube_dim
        .num_elems()
    }

    /// `Performance` uses the performance-core count only where it is smaller than the units
    /// the plan would use otherwise, so a lower injected limit stays (ADR-0017).
    #[test]
    fn performance_uses_the_core_count_only_when_smaller() {
        let mut props = cpu16();
        let perf = |cores, props: &DeviceProperties| {
            testing::with_performance_cores(cores, || {
                testing::with_cpu_units(CpuUnits::Performance, || units(props))
            })
        };
        assert_eq!(perf(Some(12), &props), 12);
        assert_eq!(perf(Some(1), &props), 1);
        assert_eq!(perf(Some(16), &props), 16);
        assert_eq!(perf(Some(24), &props), 16);
        assert_eq!(perf(Some(0), &props), 16);
        props.hardware.max_units_per_cube = 8;
        assert_eq!(
            perf(Some(12), &props),
            8,
            "with_properties' lower limit stays"
        );
        props.hardware.max_units_per_cube = 16;
        props.hardware.max_cube_dim.0 = 4;
        assert_eq!(perf(Some(12), &props), 4, "at most max_cube_dim.0");
        // No effect on GPUs.
        let mut gpu = cpu16();
        gpu.hardware.num_cpu_cores = None;
        assert_eq!(perf(Some(12), &gpu), 16.min(GPU_CUBE_DIM));
    }

    /// `with_cpu_units` takes precedence over the policy for every kind; `All` keeps every
    /// hardware thread whatever the core count.
    #[test]
    fn cpu_units_override_takes_precedence_over_the_policy() {
        let props = cpu16();
        let policy = VectorPolicy::for_class(DeviceClass::CpuSimd128);
        testing::with_performance_cores(Some(12), || {
            for kind in KernelKind::ALL {
                for bytes in [4usize, 8] {
                    let plan = |units: Option<CpuUnits>| {
                        let make = || {
                            plan_for(&props, kind, 1000, bytes, 16, 1, VectorSizeOverride::Scalar)
                                .unwrap()
                                .cube_dim
                                .num_elems()
                        };
                        match units {
                            Some(u) => testing::with_cpu_units(u, make),
                            None => make(),
                        }
                    };
                    let from_policy = match policy.cpu_units(kind, bytes) {
                        Some(CpuUnits::Performance) => 12,
                        _ => 16,
                    };
                    assert_eq!(plan(None), from_policy, "{kind:?} {bytes}");
                    assert_eq!(plan(Some(CpuUnits::All)), 16, "{kind:?} {bytes}");
                    assert_eq!(plan(Some(CpuUnits::Performance)), 12, "{kind:?} {bytes}");
                }
            }
            testing::with_cpu_units(CpuUnits::Performance, || {
                assert_eq!(
                    policy.cpu_units(KernelKind::Fill, 4),
                    Some(CpuUnits::Performance)
                );
                let gpu = VectorPolicy::for_class(DeviceClass::Gpu);
                assert_eq!(gpu.cpu_units(KernelKind::Fill, 4), None);
            });
        });
        assert_eq!(testing::cpu_units_override(), None);
    }

    /// `with_performance_cores(None)`: a CPU without separate performance cores keeps all
    /// hardware threads under `Performance`; the override is restored afterwards.
    #[test]
    fn no_performance_core_count_keeps_all_threads() {
        let props = cpu16();
        let detected = testing::performance_cores();
        testing::with_performance_cores(None, || {
            assert_eq!(testing::performance_cores(), None);
            testing::with_cpu_units(CpuUnits::Performance, || assert_eq!(units(&props), 16));
            testing::with_performance_cores(Some(4), || {
                assert_eq!(testing::performance_cores(), Some(4));
            });
            assert_eq!(testing::performance_cores(), None);
        });
        assert_eq!(testing::performance_cores(), detected);
    }

    #[test]
    fn spread_fills_x_then_y_then_z() {
        assert_eq!(spread_cubes(0, (10, 10, 10)).unwrap(), (1, 1, 1));
        assert_eq!(
            spread_cubes(7, (65_535, 65_535, 65_535)).unwrap(),
            (7, 1, 1)
        );
        assert_eq!(spread_cubes(27, (3, 3, 3)).unwrap(), (3, 3, 3));
        assert_eq!(spread_cubes(10, (3, 3, 3)).unwrap(), (3, 3, 2));
        assert!(matches!(
            spread_cubes(28, (3, 3, 3)),
            Err(GridError::TooLarge)
        ));
        assert_eq!(
            spread_cubes(100_000, (65_535, 65_535, 65_535)).unwrap(),
            (65_535, 2, 1)
        );
    }

    #[test]
    fn overflowing_lengths_are_too_large() {
        let props = props();
        assert!(matches!(
            plan_for(
                &props,
                KernelKind::Fill,
                usize::MAX,
                4,
                1,
                1,
                VectorSizeOverride::Scalar
            ),
            Err(GridError::TooLarge)
        ));
        assert!(matches!(
            plan_for(
                &props,
                KernelKind::Fill,
                1 << 33,
                4,
                1,
                1,
                VectorSizeOverride::Scalar
            ),
            Err(GridError::TooLarge)
        ));
    }
}
