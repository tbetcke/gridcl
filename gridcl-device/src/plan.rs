//! `LaunchPlan` (Listing S2, ADR-0010).

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;

use crate::buffer::pad_len;
use crate::policy::is_vector_size;
use crate::{
    DeviceClass, GridError, KernelKind, Result, VectorPolicy, VectorSizeOverride, testing,
};

/// Units per cube on GPUs, or `max_units_per_cube` if that is smaller.
const GPU_CUBE_DIM: u32 = 256;
/// Bytes of the widest stream each CPU unit processes in one block: about 16 KiB.
const CPU_BLOCK_BYTES: usize = 16 << 10;
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
    ///   `vector_sizes_in_registers(elem_bytes, live)`; overridden by
    ///   `GRIDCL_VECTOR_SIZE` or `testing::with_vector_size` (see [`VectorSizeOverride`]).
    /// - **GPUs:** `cube_dim` is 256, or `max_units_per_cube` (at most `max_cube_dim.0`) if
    ///   that is smaller; `items_per_unit` is 1.
    /// - **CPUs:** `cube_dim` equals `max_units_per_cube` (at most `max_cube_dim.0`,
    ///   ADR-0011), one unit per hardware thread, and each unit takes blocks of about 16 KiB of
    ///   whole 128-byte lines.
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
            VectorSizeOverride::current(),
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

    let (cube_dim, items_per_unit) = if class.is_cpu() {
        let bytes = n.checked_mul(elem_bytes).ok_or(GridError::TooLarge)?;
        (
            hw.max_units_per_cube.min(hw.max_cube_dim.0),
            cpu_items_per_unit(bytes),
        )
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

/// Vectors of `vector_bytes` bytes per CPU block: about `CPU_BLOCK_BYTES`, and a whole number
/// of `CPU_LINE_BYTES` lines.
fn cpu_items_per_unit(vector_bytes: usize) -> usize {
    // The fewest vectors that fill whole lines.
    let step = CPU_LINE_BYTES / gcd(CPU_LINE_BYTES, vector_bytes);
    let fit = CPU_BLOCK_BYTES / vector_bytes;
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
/// `max_bindings`.
#[allow(dead_code)] // The kernels of T3 onwards call it.
pub(crate) fn debug_assert_bindings(client: &Client, n: u32) {
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
    fn policy_and_scalar_give_one_until_benchmarked() {
        let props = props();
        for v in [VectorSizeOverride::Policy, VectorSizeOverride::Scalar] {
            let plan = plan_for(&props, KernelKind::Reduce, 1 << 20, 4, 16, 2, v).unwrap();
            assert_eq!(plan.vector_size, 1);
        }
    }

    #[test]
    fn cpu_blocks_are_whole_lines_of_about_16_kib() {
        for bytes in [4, 8, 16, 32, 64, 128, 12, 24, 256] {
            let ipu = cpu_items_per_unit(bytes);
            let block = ipu * bytes;
            assert!(block.is_multiple_of(CPU_LINE_BYTES), "{bytes}: {block}");
            assert!(
                block <= CPU_BLOCK_BYTES.max(CPU_LINE_BYTES.max(bytes) * 2),
                "{bytes}: {block}"
            );
        }
        assert_eq!(cpu_items_per_unit(4), 4096);
        assert_eq!(cpu_items_per_unit(128), 128);
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
