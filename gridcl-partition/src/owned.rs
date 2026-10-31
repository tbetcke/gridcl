//! Scotch and PT-Scotch objects (graphs, architectures, strategies, contexts) owned on the Rust
//! side and released on drop, and the conversion of a Scotch status to an error.

use std::mem::MaybeUninit;
use std::os::raw::c_int;

use gridcl_parallel::{GridError, Result};

/// `PartitionerFailed { name, code }` with a Scotch status other than 0.
pub(crate) fn status(name: &'static str, code: c_int) -> Result<()> {
    if code == 0 {
        Ok(())
    } else {
        Err(GridError::PartitionerFailed { name, code })
    }
}

/// A Scotch object initialised in place on the heap, so that it never moves, and released by
/// its exit function on drop.
pub(crate) struct Owned<T> {
    inner: Box<MaybeUninit<T>>,
    exit: unsafe extern "C" fn(*mut T),
}

impl<T> Owned<T> {
    /// The object initialised by `init`, released by `exit`; `name` is the partitioner's, for
    /// the error if `init` fails.
    ///
    /// # Safety
    ///
    /// `init` initialises the `T` at the pointer it is given when it returns 0, and `exit`
    /// releases such an object.
    pub(crate) unsafe fn new(
        name: &'static str,
        init: impl FnOnce(*mut T) -> c_int,
        exit: unsafe extern "C" fn(*mut T),
    ) -> Result<Self> {
        // The storage has the size and alignment of T, the header's opaque struct.
        let mut inner = Box::new(MaybeUninit::uninit());
        status(name, init(inner.as_mut_ptr()))?;
        Ok(Owned { inner, exit })
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
        self.inner.as_mut_ptr()
    }
}

impl<T> Drop for Owned<T> {
    fn drop(&mut self) {
        // SAFETY: the object was initialised in `new`, exit is its matching `*Exit` function,
        // and the object is not used afterwards.
        unsafe { (self.exit)(self.inner.as_mut_ptr()) }
    }
}
