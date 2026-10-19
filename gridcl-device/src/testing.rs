//! Test hooks (ADR-0007). Only with the Cargo feature `testing`, which is off by default and not
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

#[cfg(feature = "testing")]
use crate::{KernelKind, LaunchPlan, VectorSizeOverride};

#[cfg(feature = "testing")]
thread_local! {
    static PROPERTIES: RefCell<Option<Arc<DeviceProperties>>> = const { RefCell::new(None) };
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

/// Forces the vector size of plans made inside `f`.
#[cfg(feature = "testing")]
pub fn with_vector_size<R>(_v: VectorSizeOverride, _f: impl FnOnce() -> R) -> R {
    todo!()
}

/// Forces `VectorPolicy::quad_tile` inside `f`.
#[cfg(feature = "testing")]
pub fn with_quad_tile<R>(_tile: usize, _f: impl FnOnce() -> R) -> R {
    todo!()
}

/// The plans made inside `f`, so a test can assert the vector size it ran at.
#[cfg(feature = "testing")]
pub fn record_plans<R>(_f: impl FnOnce() -> R) -> (R, Vec<(KernelKind, LaunchPlan)>) {
    todo!()
}
