//! Reduce (KernelKind::Reduce), generic over the vector size.
//!
//! Each level writes one partial per cube; the next level reduces the partials with a fresh
//! plan, until one cube remains. Within a unit the values are combined within each lane
//! first, then the lanes in order, then the units of a cube by a fixed shared-memory tree.
//! No plane operations and no atomics, so the result is bitwise reproducible.

use cubecl::prelude::*;

use crate::plan::debug_assert_bindings;
use crate::{
    DeviceBuffer, DeviceScalar, GridError, KernelKind, LaunchPlan, ReduceOp, Result, VECTOR_PAD,
    fill,
};

/// `ReduceOp` as a comptime kernel parameter.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum Op {
    Sum,
    Min,
    Max,
}

impl From<ReduceOp> for Op {
    fn from(op: ReduceOp) -> Op {
        match op {
            ReduceOp::Sum => Op::Sum,
            ReduceOp::Min => Op::Min,
            ReduceOp::Max => Op::Max,
        }
    }
}

/// The identity of `op` on the host: 0, `T::max_value()` or `T::min_value()` (for floats the
/// largest finite values).
pub(crate) fn host_identity<T: DeviceScalar>(op: ReduceOp) -> T {
    match op {
        ReduceOp::Sum => T::from_int(0),
        ReduceOp::Min => T::max_value(),
        ReduceOp::Max => T::min_value(),
    }
}

/// The identity of `op` in a kernel.
#[cube]
pub(crate) fn identity<T: Numeric>(#[comptime] op: Op) -> T {
    match op {
        Op::Sum => T::from_int(0),
        Op::Min => T::max_value(),
        Op::Max => T::min_value(),
    }
}

/// `a op b` for scalars.
#[cube]
pub(crate) fn combine<T: Numeric>(a: T, b: T, #[comptime] op: Op) -> T {
    match op {
        Op::Sum => a + b,
        Op::Min => min(a, b),
        Op::Max => max(a, b),
    }
}

/// `a op b` lane by lane.
#[cube]
fn combine_vector<T: Numeric, N: Size>(
    a: Vector<T, N>,
    b: Vector<T, N>,
    #[comptime] op: Op,
) -> Vector<T, N> {
    match op {
        Op::Sum => a + b,
        Op::Min => min(a, b),
        Op::Max => max(a, b),
    }
}

/// Reduces `shared[0..cube_dim]` into `shared[0]` by a fixed pairwise tree: at stride s
/// (from `half`, the largest power of two below `cube_dim` rounded up, down to 1) unit u
/// combines `shared[u + s]` into `shared[u]`. Every unit of the cube must call it.
#[cube]
pub(crate) fn cube_tree<T: Numeric>(
    shared: &mut Shared<[T]>,
    half: u32,
    #[comptime] cube_dim: u32,
    #[comptime] op: Op,
) {
    let u = UNIT_POS;
    let mut stride = half;
    while stride > 0 {
        if u < stride && u + stride < cube_dim {
            shared[u as usize] =
                combine::<T>(shared[u as usize], shared[(u + stride) as usize], op);
        }
        sync_cube();
        stride /= 2;
    }
}

/// One partial per cube: `partials[CUBE_POS]` combines the scalars `x[0..len]` that the
/// cube's units own (Listing B5 loop); lanes at index >= `len` take the identity.
#[cube(launch_unchecked)]
fn reduce_kernel<T: Numeric, N: Size>(
    x: &[Vector<T, N>],
    partials: &mut [T],
    len: u32,
    num_vectors: u32,
    items_per_unit: u32,
    half: u32,
    #[comptime] cube_dim: u32,
    #[comptime] op: Op,
) {
    let lanes = N::value();
    let ident = identity::<T>(op);
    let mut acc = Vector::<T, N>::broadcast(ident);
    let len = len as usize;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let start = v * lanes;
            if start + lanes <= len {
                acc = combine_vector(acc, x[v], op);
            } else if start < len {
                // The one vector that holds the end: lanes past `len` keep the identity.
                let value = x[v];
                let mut masked = Vector::<T, N>::broadcast(ident);
                #[unroll]
                for l in 0..lanes {
                    if start + l < len {
                        masked.insert(l, value.extract(l));
                    }
                }
                acc = combine_vector(acc, masked, op);
            }
        }
    }
    let mut s = acc.extract(0usize);
    #[unroll]
    for l in 1..lanes {
        s = combine::<T>(s, acc.extract(l), op);
    }
    let mut shared = Shared::<[T]>::new_slice(cube_dim as usize);
    shared[UNIT_POS as usize] = s;
    sync_cube();
    cube_tree(&mut shared, half, cube_dim, op);
    if UNIT_POS == 0 {
        partials[CUBE_POS] = shared[0];
    }
}

/// The largest power of two below `cube_dim` rounded up to a power of two: the first stride
/// of [`cube_tree`]; 0 for one unit.
pub(crate) fn tree_half(cube_dim: u32) -> u32 {
    cube_dim.next_power_of_two() / 2
}

/// Reduces `x` with `op`. Length 1; an empty input gives the identity of `op`: 0,
/// `T::max_value()` for `Min`, `T::min_value()` for `Max` (for floats the largest finite
/// values, not infinities).
///
/// The result is bitwise identical across runs on one device. Float sums depend on the vector
/// size and the launch shape, which the vector policy fixes per device class.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, ReduceOp, reduce};
///
/// let client = gridcl_testing::test_client();
/// let x = DeviceBuffer::from_host(&client, &[3i32, -1, 4, 1, -5]).unwrap();
/// assert_eq!(reduce(&x, ReduceOp::Sum).unwrap().to_host().unwrap(), vec![2]);
/// assert_eq!(reduce(&x, ReduceOp::Min).unwrap().to_host().unwrap(), vec![-5]);
/// assert_eq!(reduce(&x, ReduceOp::Max).unwrap().to_host().unwrap(), vec![4]);
/// ```
pub fn reduce<T: DeviceScalar>(x: &DeviceBuffer<T>, op: ReduceOp) -> Result<DeviceBuffer<T>> {
    let client = x.client();
    if x.len() == 0 {
        let mut out = DeviceBuffer::uninit(client, 1)?;
        fill(&mut out, host_identity::<T>(op))?;
        return Ok(out);
    }
    let mut level = x.share();
    loop {
        let len = level.len();
        let plan = LaunchPlan::elementwise(
            client,
            KernelKind::Reduce,
            len,
            size_of::<T>(),
            VECTOR_PAD,
            2,
        )?;
        let cubes = plan.num_cubes();
        if cubes > 1 && cubes >= len {
            // A level that does not shrink would never finish (one unit, one item per cube).
            return Err(GridError::TooLarge);
        }
        let partials = DeviceBuffer::<T>::uninit(client, cubes)?;
        launch_reduce(&level, &partials, &plan, op)?;
        if cubes == 1 {
            return Ok(partials);
        }
        level = partials;
    }
}

/// One level: the partials of `x` under `plan`, one per cube.
fn launch_reduce<T: DeviceScalar>(
    x: &DeviceBuffer<T>,
    partials: &DeviceBuffer<T>,
    plan: &LaunchPlan,
    op: ReduceOp,
) -> Result<()> {
    let client = x.client();
    let n = plan.vector_size;
    let len = u32::try_from(x.len()).map_err(|_| GridError::TooLarge)?;
    let cube_dim = plan.cube_dim.num_elems();
    debug_assert_bindings(client, 2);
    // SAFETY: `x` holds `padded_len() >= num_vectors · n` elements (the plan's num_vectors is
    // pad_len(len) / n), and the kernel reads only vectors below num_vectors. `partials` has
    // `num_cubes` elements, and each cube writes only partials[CUBE_POS]. The shared array
    // has cube_dim entries, indexed by UNIT_POS < cube_dim.
    unsafe {
        reduce_kernel::launch_unchecked::<T>(
            client,
            plan.cube_count.clone(),
            plan.cube_dim,
            n,
            x.vector_arg(n),
            partials.arg(),
            len,
            plan.num_vectors,
            plan.items_per_unit,
            tree_half(cube_dim),
            cube_dim,
            Op::from(op),
        );
    }
    Ok(())
}
