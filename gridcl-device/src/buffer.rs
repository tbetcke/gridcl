//! The `DeviceBuffer` family (Listing S1).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::Result;

/// Every allocation is padded to a multiple of `VECTOR_PAD` elements, the widest vector size
/// any device class uses, so vectorized kernels never need a scalar tail.
pub const VECTOR_PAD: usize = 16;

/// Typed, length-carrying view of one CubeCL allocation. Not `Clone`: a second view of the same
/// memory is made explicitly with [`share`](Self::share), an independent copy with
/// [`deep_clone`](Self::deep_clone).
pub struct DeviceBuffer<T: CubeElement> {
    handle: Handle,
    len: usize,
    padded: usize,
    client: Client,
    _t: PhantomData<T>,
}

// Listing S1 has no `is_empty`; the public API follows it.
#[allow(clippy::len_without_is_empty)]
impl<T: CubeElement> DeviceBuffer<T> {
    /// Uploads `data` and zeroes the padding.
    pub fn from_host(_client: &Client, _data: &[T]) -> Result<Self> {
        todo!()
    }

    /// An allocation of `len` elements whose contents, padding included, are unspecified.
    pub fn uninit(_client: &Client, _len: usize) -> Result<Self> {
        todo!()
    }

    /// `len` zeros, padding included (uninit and the fill kernel).
    pub fn zeros(_client: &Client, _len: usize) -> Result<Self> {
        todo!()
    }

    /// A second view of the same allocation; aliasing is explicit.
    pub fn share(&self) -> Self {
        todo!()
    }

    /// An independent copy, made by the copy kernel.
    pub fn deep_clone(&self) -> Result<Self> {
        todo!()
    }

    /// The logical length in elements.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The allocated length in elements: `len` rounded up to `VECTOR_PAD`, at least one block.
    pub fn padded_len(&self) -> usize {
        self.padded
    }

    /// The CubeCL handle.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The client the buffer lives on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Explicit, blocking device-to-host copy of the logical length.
    pub fn to_host(&self) -> Result<Vec<T>> {
        todo!()
    }

    /// Explicit device-to-host copy of the logical length.
    pub async fn to_host_async(&self) -> Result<Vec<T>> {
        todo!()
    }

    /// Scalar kernel argument.
    ///
    /// # Safety
    ///
    /// In CubeCL 0.11 the handle's size, not the logical length, bounds the kernel's slice. The
    /// kernel must stay within `len` and the padding, and takes its lengths as explicit
    /// arguments.
    pub unsafe fn arg(&self) -> BufferArg {
        todo!()
    }

    /// Argument for a kernel over `Vector<T, n>`: the padded length, counted in vectors.
    ///
    /// # Safety
    ///
    /// As for [`arg`](Self::arg); kernels take `num_vectors` explicitly, as in Listing B5.
    pub unsafe fn vector_arg(&self, _n: usize) -> BufferArg {
        todo!()
    }
}

/// Fixed number of entries per row (cell -> vertices, etc.).
pub struct DeviceStrided<T: CubeElement> {
    pub data: DeviceBuffer<T>,
    pub stride: u32,
}

/// Variable-length rows (facet -> cells, vertex -> cells).
pub struct DeviceCsr {
    pub offsets: DeviceBuffer<u32>,
    pub indices: DeviceBuffer<u32>,
}
