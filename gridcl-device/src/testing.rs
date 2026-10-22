//! Test hooks (ADR-0009). Only with the Cargo feature `testing`, which is off by default and not
//! part of the stable API; gridcl-testing turns it on for every workspace test.
//!
//! The overrides are scoped and thread-local, because gridcl runs one host thread per mesh.
//! Precedence: the scoped override, then the environment variable (`GRIDCL_VECTOR_SIZE`,
//! `GRIDCL_QUAD_TILE`), then vector-policy.toml.
//!
//! Without the feature this module is private and holds only the accessors the rest of the
//! crate reads its properties through.

use std::sync::Arc;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;

#[cfg(feature = "testing")]
use std::cell::RefCell;

use crate::{KernelKind, LaunchPlan, VectorSizeOverride};

#[cfg(feature = "testing")]
thread_local! {
    static PROPERTIES: RefCell<Option<Arc<DeviceProperties>>> = const { RefCell::new(None) };
    static VECTOR_SIZE: RefCell<Option<VectorSizeOverride>> = const { RefCell::new(None) };
    static QUAD_TILE: RefCell<Option<usize>> = const { RefCell::new(None) };
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
/// assert_eq!(with_quad_tile(4, || VectorPolicy::for_client(&client).quad_tile()), 4);
/// ```
#[cfg(feature = "testing")]
pub fn with_quad_tile<R>(tile: usize, f: impl FnOnce() -> R) -> R {
    assert!(
        crate::policy::is_vector_size(tile),
        "with_quad_tile: {tile} is not 1, 2, 4, 8 or 16"
    );
    scoped(&QUAD_TILE, tile, f)
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
