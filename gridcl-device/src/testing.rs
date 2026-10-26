//! Test hooks (ADR-0009). Only with the Cargo feature `testing`, which is off by default and not
//! part of the stable API; gridcl-testing turns it on for every workspace test.
//!
//! The overrides are scoped and thread-local, because gridcl runs one host thread per mesh.
//! Precedence: the scoped per-kind override, then the scoped override, then the environment
//! variable (`GRIDCL_VECTOR_SIZE`, `GRIDCL_QUAD_TILE`), then vector-policy.toml (ADR-0015).
//! The CPU units and the performance-core count have scoped overrides only (ADR-0017).
//!
//! Without the feature this module is private and holds only the accessors the rest of the
//! crate reads its properties through.

use std::sync::Arc;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;

#[cfg(feature = "testing")]
use std::cell::RefCell;

use crate::{CpuUnits, KernelKind, LaunchPlan, VectorSizeOverride};

#[cfg(feature = "testing")]
thread_local! {
    static PROPERTIES: RefCell<Option<Arc<DeviceProperties>>> = const { RefCell::new(None) };
    static VECTOR_SIZE: RefCell<Option<VectorSizeOverride>> = const { RefCell::new(None) };
    static QUAD_TILE: RefCell<Option<usize>> = const { RefCell::new(None) };
    /// The per-kind overrides in scope, innermost last.
    static KIND_VECTOR_SIZE: RefCell<Option<Vec<(KernelKind, VectorSizeOverride)>>> =
        const { RefCell::new(None) };
    static CPU_BLOCK: RefCell<Option<usize>> = const { RefCell::new(None) };
    static CPU_UNITS: RefCell<Option<CpuUnits>> = const { RefCell::new(None) };
    /// `Some(cores)` replaces the detected performance-core count.
    static PERFORMANCE_CORES: RefCell<Option<Option<u32>>> = const { RefCell::new(None) };
    static PLANS: RefCell<Option<Vec<(KernelKind, LaunchPlan)>>> = const { RefCell::new(None) };
}

/// The properties gridcl-device reads for `client`: the injected ones inside
/// `testing::with_properties` (feature `testing`), else the client's own. Kernel launchers in
/// every crate read device limits through it, so injected limits reach them (ADR-0012).
///
/// # Example
///
/// ```
/// use gridcl_device::device_properties;
///
/// let client = gridcl_testing::test_client();
/// let props = device_properties(&client);
/// assert_eq!(props.memory.max_page_size, client.properties().memory.max_page_size);
/// ```
pub fn device_properties(client: &Client) -> Arc<DeviceProperties> {
    properties(client)
}

/// The properties gridcl-device reads for `client`: the injected ones inside
/// [`with_properties`], else the client's own.
pub(crate) fn properties(client: &Client) -> Arc<DeviceProperties> {
    #[cfg(feature = "testing")]
    if let Some(props) = PROPERTIES.with(|p| p.borrow().clone()) {
        return props;
    }
    client.properties_shared()
}

/// The scoped vector-size override of [`with_vector_size`], if any.
pub(crate) fn vector_override() -> Option<VectorSizeOverride> {
    #[cfg(feature = "testing")]
    return VECTOR_SIZE.with(|v| *v.borrow());
    #[cfg(not(feature = "testing"))]
    None
}

/// The innermost scoped override of [`with_kind_vector_size`] for `kind`, if any.
pub(crate) fn kind_vector_override(kind: KernelKind) -> Option<VectorSizeOverride> {
    #[cfg(feature = "testing")]
    return KIND_VECTOR_SIZE.with(|v| {
        v.borrow().as_ref().and_then(|stack| {
            stack
                .iter()
                .rev()
                .find(|(k, _)| *k == kind)
                .map(|(_, v)| *v)
        })
    });
    #[cfg(not(feature = "testing"))]
    {
        let _ = kind;
        None
    }
}

/// The scoped CPU block override of [`with_cpu_block_bytes`], if any.
pub(crate) fn cpu_block_override() -> Option<usize> {
    #[cfg(feature = "testing")]
    return CPU_BLOCK.with(|b| *b.borrow());
    #[cfg(not(feature = "testing"))]
    None
}

/// The scoped CPU units override of [`with_cpu_units`], if any.
pub(crate) fn cpu_units_override() -> Option<CpuUnits> {
    #[cfg(feature = "testing")]
    return CPU_UNITS.with(|u| *u.borrow());
    #[cfg(not(feature = "testing"))]
    None
}

/// The performance-core count plans use: the scoped one of [`with_performance_cores`], else the
/// detected one (macOS), else `None`.
pub(crate) fn performance_cores() -> Option<u32> {
    #[cfg(feature = "testing")]
    if let Some(cores) = PERFORMANCE_CORES.with(|c| *c.borrow()) {
        return cores;
    }
    crate::cores::performance_cores()
}

/// The scoped tile-width override of [`with_quad_tile`], if any.
pub(crate) fn quad_tile_override() -> Option<usize> {
    #[cfg(feature = "testing")]
    return QUAD_TILE.with(|t| *t.borrow());
    #[cfg(not(feature = "testing"))]
    None
}

/// Records `plan` when [`record_plans`] is active on this thread.
pub(crate) fn record(kind: KernelKind, plan: &LaunchPlan) {
    #[cfg(feature = "testing")]
    PLANS.with(|p| {
        if let Some(plans) = p.borrow_mut().as_mut() {
            plans.push((kind, plan.clone()));
        }
    });
    #[cfg(not(feature = "testing"))]
    let _ = (kind, plan);
}

/// Sets a thread-local override for the duration of `f` and restores the previous value
/// afterwards, also when `f` panics.
#[cfg(feature = "testing")]
fn scoped<T: 'static, R>(
    key: &'static std::thread::LocalKey<RefCell<Option<T>>>,
    value: T,
    f: impl FnOnce() -> R,
) -> R {
    struct Restore<T: 'static> {
        key: &'static std::thread::LocalKey<RefCell<Option<T>>>,
        previous: Option<T>,
    }
    impl<T: 'static> Drop for Restore<T> {
        fn drop(&mut self) {
            let previous = self.previous.take();
            self.key.with(|cell| *cell.borrow_mut() = previous);
        }
    }
    let previous = key.with(|cell| cell.borrow_mut().replace(value));
    let _restore = Restore { key, previous };
    f()
}

/// Replaces the properties gridcl-device reads (limits, device class, vector sizes) for buffers
/// and plans made inside `f`; tests clone `client.properties()` and modify it. Nested calls
/// restore the outer properties when they return.
///
/// # Example
///
/// ```
/// use gridcl_device::testing::with_properties;
/// use gridcl_device::{DeviceBuffer, GridError};
///
/// let client = gridcl_testing::test_client();
/// let mut props = client.properties().clone();
/// props.memory.max_page_size = 1024;
/// with_properties(props, || {
///     assert!(DeviceBuffer::<u32>::uninit(&client, 256).is_ok());
///     assert!(matches!(
///         DeviceBuffer::<u32>::uninit(&client, 257),
///         Err(GridError::AllocationTooLarge { bytes: 1088, max: 1024 })
///     ));
/// });
/// ```
#[cfg(feature = "testing")]
pub fn with_properties<R>(props: DeviceProperties, f: impl FnOnce() -> R) -> R {
    scoped(&PROPERTIES, Arc::new(props), f)
}

/// Forces the vector size of plans made inside `f`. It takes precedence over
/// `GRIDCL_VECTOR_SIZE`; `Fixed(n)` and `Widest` stay clamped by the kernel's `max_vector`
/// and the device's `hardware.max_vector_size` (ADR-0011).
///
/// # Example
///
/// ```
/// use gridcl_device::testing::with_vector_size;
/// use gridcl_device::{KernelKind, LaunchPlan, VectorSizeOverride};
///
/// let client = gridcl_testing::test_client();
/// let plan = with_vector_size(VectorSizeOverride::Fixed(2), || {
///     LaunchPlan::elementwise(&client, KernelKind::Copy, 100, 4, 16, 1)
/// })
/// .unwrap();
/// assert_eq!(plan.vector_size, 2);
/// ```
#[cfg(feature = "testing")]
pub fn with_vector_size<R>(v: VectorSizeOverride, f: impl FnOnce() -> R) -> R {
    if let VectorSizeOverride::Fixed(n) = v {
        assert!(
            crate::policy::is_vector_size(n),
            "with_vector_size: Fixed({n}) is not a power of two dividing VECTOR_PAD"
        );
    }
    scoped(&VECTOR_SIZE, v, f)
}

/// Forces `VectorPolicy::quad_tile` inside `f`. It takes precedence over `GRIDCL_QUAD_TILE`.
///
/// # Example
///
/// ```
/// use gridcl_device::VectorPolicy;
/// use gridcl_device::testing::with_quad_tile;
///
/// let client = gridcl_testing::test_client();
/// assert_eq!(with_quad_tile(4, || VectorPolicy::for_client(&client).quad_tile(8)), 4);
/// ```
#[cfg(feature = "testing")]
pub fn with_quad_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R {
    assert!(
        crate::policy::is_vector_size(tile),
        "with_quad_tile: {tile} is not 1, 2, 4, 8 or 16"
    );
    scoped(&QUAD_TILE, tile, f)
}

/// Vector size of one family; takes precedence over [`with_vector_size`], the environment and
/// the policy for that kind. Other kinds are unaffected (ADR-0015). Nested calls for other
/// kinds add to it; the innermost call for a kind wins. `Fixed(n)` and `Widest` stay clamped
/// as in [`with_vector_size`].
///
/// # Example
///
/// ```
/// use gridcl_device::testing::{with_kind_vector_size, with_vector_size};
/// use gridcl_device::{KernelKind, LaunchPlan, VectorSizeOverride};
///
/// let client = gridcl_testing::test_client();
/// let plan = |kind| LaunchPlan::elementwise(&client, kind, 100, 4, 16, 1).unwrap().vector_size;
/// let (sort, scan) = with_vector_size(VectorSizeOverride::Fixed(2), || {
///     with_kind_vector_size(KernelKind::SortDigits, VectorSizeOverride::Scalar, || {
///         (plan(KernelKind::SortDigits), plan(KernelKind::ScanTile))
///     })
/// });
/// assert_eq!(sort, 1);
/// assert_eq!(scan, 2.min(client.properties().hardware.max_vector_size));
/// ```
#[cfg(feature = "testing")]
pub fn with_kind_vector_size<R>(
    kind: KernelKind,
    v: VectorSizeOverride,
    f: impl FnOnce() -> R,
) -> R {
    if let VectorSizeOverride::Fixed(n) = v {
        assert!(
            crate::policy::is_vector_size(n),
            "with_kind_vector_size: Fixed({n}) is not a power of two dividing VECTOR_PAD"
        );
    }
    let mut stack = KIND_VECTOR_SIZE.with(|s| s.borrow().clone().unwrap_or_default());
    stack.push((kind, v));
    scoped(&KIND_VECTOR_SIZE, stack, f)
}

/// CPU block target, a power of two from 4 to 64 KiB, for plans made inside `f` (ADR-0015). It
/// replaces `VectorPolicy::cpu_block_bytes` for every kind on CPU devices and has no effect on
/// GPUs; plans still round blocks to whole 128-byte lines.
///
/// # Panics
///
/// If `bytes` is not a power of two from 4 KiB to 64 KiB.
///
/// # Example
///
/// ```
/// use gridcl_device::testing::with_cpu_block_bytes;
/// use gridcl_device::{DeviceClass, KernelKind, VectorPolicy};
///
/// let client = gridcl_testing::test_client();
/// let block = with_cpu_block_bytes(64 << 10, || {
///     VectorPolicy::for_client(&client).cpu_block_bytes(KernelKind::Gather, 8)
/// });
/// if DeviceClass::for_client(&client) != DeviceClass::Gpu {
///     assert_eq!(block, Some(64 << 10));
/// }
/// ```
#[cfg(feature = "testing")]
pub fn with_cpu_block_bytes<R>(bytes: usize, f: impl FnOnce() -> R) -> R {
    assert!(
        crate::policy::is_cpu_block_bytes(bytes),
        "with_cpu_block_bytes: {bytes} is not a power of two from 4 KiB to 64 KiB"
    );
    scoped(&CPU_BLOCK, bytes, f)
}

/// CPU units for plans made inside `f`, in place of `VectorPolicy::cpu_units` for every kind
/// on CPU devices (ADR-0017); no effect on GPUs. `Performance` still uses the
/// performance-core count only where it is smaller than `max_units_per_cube`.
///
/// # Example
///
/// ```
/// use gridcl_device::testing::{with_cpu_units, with_performance_cores};
/// use gridcl_device::{CpuUnits, DeviceClass, KernelKind, LaunchPlan};
///
/// let client = gridcl_testing::test_client();
/// let all = LaunchPlan::elementwise(&client, KernelKind::Reduce, 1 << 20, 4, 16, 1).unwrap();
/// let plan = with_performance_cores(Some(1), || {
///     with_cpu_units(CpuUnits::Performance, || {
///         LaunchPlan::elementwise(&client, KernelKind::Reduce, 1 << 20, 4, 16, 1)
///     })
/// })
/// .unwrap();
/// if DeviceClass::for_client(&client) != DeviceClass::Gpu {
///     assert_eq!(plan.cube_dim.num_elems(), 1.min(all.cube_dim.num_elems()));
/// }
/// ```
#[cfg(feature = "testing")]
pub fn with_cpu_units<R>(units: CpuUnits, f: impl FnOnce() -> R) -> R {
    scoped(&CPU_UNITS, units, f)
}

/// Replaces the detected performance-core count inside `f`; `None` means the CPU has no
/// separate performance cores, so `CpuUnits::Performance` keeps all hardware threads.
///
/// # Example
///
/// ```
/// use gridcl_device::testing::{with_cpu_units, with_performance_cores};
/// use gridcl_device::{CpuUnits, KernelKind, LaunchPlan};
///
/// let client = gridcl_testing::test_client();
/// let units = |cores| {
///     with_performance_cores(cores, || {
///         with_cpu_units(CpuUnits::Performance, || {
///             LaunchPlan::elementwise(&client, KernelKind::Fill, 1000, 4, 16, 1)
///         })
///     })
///     .unwrap()
///     .cube_dim
///     .num_elems()
/// };
/// let all = with_cpu_units(CpuUnits::All, || {
///     LaunchPlan::elementwise(&client, KernelKind::Fill, 1000, 4, 16, 1)
/// })
/// .unwrap();
/// assert_eq!(units(None), all.cube_dim.num_elems());
/// ```
#[cfg(feature = "testing")]
pub fn with_performance_cores<R>(cores: Option<u32>, f: impl FnOnce() -> R) -> R {
    scoped(&PERFORMANCE_CORES, cores, f)
}

/// The plans made inside `f`, in order, so a test can assert the vector size it ran at. An
/// enclosing `record_plans` sees them too.
///
/// # Example
///
/// ```
/// use gridcl_device::testing::record_plans;
/// use gridcl_device::{KernelKind, LaunchPlan};
///
/// let client = gridcl_testing::test_client();
/// let (_, plans) = record_plans(|| LaunchPlan::elementwise(&client, KernelKind::Reduce, 64, 4, 16, 2));
/// assert_eq!(plans.len(), 1);
/// assert_eq!(plans[0].0, KernelKind::Reduce);
/// ```
#[cfg(feature = "testing")]
pub fn record_plans<R>(f: impl FnOnce() -> R) -> (R, Vec<(KernelKind, LaunchPlan)>) {
    struct Restore {
        previous: Option<Vec<(KernelKind, LaunchPlan)>>,
    }
    impl Drop for Restore {
        fn drop(&mut self) {
            let mut previous = self.previous.take();
            PLANS.with(|p| {
                let mut p = p.borrow_mut();
                // On a panic inside `f` the inner plans are still passed on to the outer scope.
                if let (Some(outer), Some(inner)) = (previous.as_mut(), p.take()) {
                    outer.extend(inner);
                }
                *p = previous;
            });
        }
    }
    let previous = PLANS.with(|p| p.borrow_mut().replace(Vec::new()));
    let restore = Restore { previous };
    let result = f();
    let plans = PLANS.with(|p| p.borrow().clone().unwrap_or_default());
    drop(restore);
    (result, plans)
}
