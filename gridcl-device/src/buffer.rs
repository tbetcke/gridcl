//! The `DeviceBuffer` family (Listing S1).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::primitives::launch_fill;
use crate::{GridError, Result, testing};

/// Every allocation is padded to a multiple of `VECTOR_PAD` elements, the widest vector size
/// any device class uses, so vectorized kernels never need a scalar tail.
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, VECTOR_PAD};
///
/// let client = gridcl_testing::test_client();
/// let b = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
/// assert_eq!(b.padded_len(), VECTOR_PAD);
/// ```
pub const VECTOR_PAD: usize = 16;

/// Typed, length-carrying view of one CubeCL allocation. Not `Clone`: a second view of the same
/// memory is made explicitly with [`share`](Self::share), an independent copy with
/// [`deep_clone`](Self::deep_clone).
///
/// Every constructor checks the allocation against the device's `max_page_size` and returns
/// `GridError::AllocationTooLarge` above it.
///
/// # Example
///
/// ```
/// use gridcl_device::DeviceBuffer;
///
/// let client = gridcl_testing::test_client();
/// let b = DeviceBuffer::from_host(&client, &[1.5f32, 2.5]).unwrap();
/// assert_eq!(b.len(), 2);
/// assert_eq!(b.to_host().unwrap(), vec![1.5, 2.5]);
/// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[7i32; 20]).unwrap();
    /// assert_eq!((b.len(), b.padded_len()), (20, 32));
    /// ```
    pub fn from_host(client: &Client, data: &[T]) -> Result<Self> {
        let padded = pad_len(data.len())?;
        let bytes = check_allocation::<T>(client, padded)?;
        let mut host = T::as_bytes(data).to_vec();
        host.resize(bytes, 0);
        let handle = client.create_from_slice(&host);
        Ok(Self::new(handle, data.len(), padded, client))
    }

    /// An allocation of `len` elements whose contents, padding included, are unspecified: only
    /// for outputs that a kernel fully writes. An empty buffer still allocates one
    /// `VECTOR_PAD` block, so no handle has zero size.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{DeviceBuffer, VECTOR_PAD};
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::<f32>::uninit(&client, 0).unwrap();
    /// assert_eq!((b.len(), b.padded_len()), (0, VECTOR_PAD));
    /// ```
    pub fn uninit(client: &Client, len: usize) -> Result<Self> {
        let padded = pad_len(len)?;
        let bytes = check_allocation::<T>(client, padded)?;
        Ok(Self::new(client.empty(bytes), len, padded, client))
    }

    /// `len` zeros, padding included (uninit and the fill kernel, over u32 words).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let z = DeviceBuffer::<f32>::zeros(&client, 3).unwrap();
    /// assert_eq!(z.to_host().unwrap(), vec![0.0; 3]);
    /// ```
    pub fn zeros(client: &Client, len: usize) -> Result<Self> {
        let buffer = Self::uninit(client, len)?;
        // `padded` is a multiple of 16, so the allocation is a whole number of u32 words.
        let words = buffer.padded * size_of::<T>() / size_of::<u32>();
        launch_fill::<u32>(client, &buffer.handle, words, 0)?;
        Ok(buffer)
    }

    /// A second view of the same allocation; aliasing is explicit.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let a = DeviceBuffer::from_host(&client, &[1u32, 2]).unwrap();
    /// let b = a.share();
    /// assert_eq!(b.to_host().unwrap(), a.to_host().unwrap());
    /// ```
    pub fn share(&self) -> Self {
        Self::new(self.handle.clone(), self.len, self.padded, &self.client)
    }

    /// An independent copy, made by the copy kernel (CubeCL 0.11 has no same-device copy). As
    /// for [`copy`](crate::copy), `size_of::<T>()` must be a multiple of 4.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let a = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
    /// let b = a.deep_clone().unwrap();
    /// drop(a);
    /// assert_eq!(b.to_host().unwrap(), vec![1, 2, 3]);
    /// ```
    pub fn deep_clone(&self) -> Result<Self> {
        let mut clone = Self::uninit(&self.client, self.len)?;
        crate::copy(self, &mut clone)?;
        Ok(clone)
    }

    /// The logical length in elements, not bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The allocated length in elements: `len` rounded up to `VECTOR_PAD`, at least one block.
    pub fn padded_len(&self) -> usize {
        self.padded
    }

    /// The CubeCL handle, which keeps the allocation alive.
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The client the buffer lives on.
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// Explicit, blocking device-to-host copy of the logical length.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[3u32, 1, 4]).unwrap();
    /// assert_eq!(b.to_host().unwrap(), vec![3, 1, 4]);
    /// ```
    pub fn to_host(&self) -> Result<Vec<T>> {
        let bytes = self.client.read_one(self.handle.clone())?;
        decode(&bytes, self.len)
    }

    /// Explicit device-to-host copy of the logical length.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[2.0f32, 4.0]).unwrap();
    /// let host = cubecl::future::block_on(b.to_host_async()).unwrap();
    /// assert_eq!(host, vec![2.0, 4.0]);
    /// ```
    pub async fn to_host_async(&self) -> Result<Vec<T>> {
        let mut out = self.client.read_async(vec![self.handle.clone()]).await?;
        decode(&out.remove(0), self.len)
    }

    /// Scalar kernel argument.
    ///
    /// # Safety
    ///
    /// In CubeCL 0.11 the handle's size, not the logical length, bounds the kernel's slice. The
    /// kernel must stay within `len` and the padding, and takes its lengths as explicit
    /// arguments.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
    /// // SAFETY: the argument is not passed to a kernel.
    /// let _arg = unsafe { b.arg() };
    /// ```
    pub unsafe fn arg(&self) -> BufferArg {
        // SAFETY: the handle holds `padded >= len` elements of `T`; the caller keeps the kernel
        // within them.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len) }
    }

    /// Argument for a kernel over `Vector<T, n>`: the padded length, counted in vectors.
    ///
    /// # Safety
    ///
    /// As for [`arg`](Self::arg); kernels take `num_vectors` explicitly, as in Listing B5.
    /// `n` must be a power of two that divides `VECTOR_PAD`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::DeviceBuffer;
    ///
    /// let client = gridcl_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[0.5f32; 40]).unwrap();
    /// // SAFETY: the argument is not passed to a kernel.
    /// let _arg = unsafe { b.vector_arg(4) };
    /// ```
    pub unsafe fn vector_arg(&self, n: usize) -> BufferArg {
        debug_assert!(n.is_power_of_two() && VECTOR_PAD.is_multiple_of(n));
        // SAFETY: the handle holds `padded` elements of `T`, a whole number of vectors of `n`
        // lanes since `n` divides VECTOR_PAD; the caller keeps the kernel within them.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.padded / n) }
    }

    fn new(handle: Handle, len: usize, padded: usize, client: &Client) -> Self {
        Self {
            handle,
            len,
            padded,
            client: client.clone(),
            _t: PhantomData,
        }
    }
}

/// `len` rounded up to `VECTOR_PAD`. An empty buffer still gets one block, so no handle has
/// zero size (wgpu rejects zero-size bindings); `len()` stays 0.
pub(crate) fn pad_len(len: usize) -> Result<usize> {
    len.max(1)
        .checked_next_multiple_of(VECTOR_PAD)
        .ok_or(GridError::TooLarge)
}

/// Bytes for `len` elements, or an error if the device cannot hold them in one allocation.
fn check_allocation<T: CubeElement>(client: &Client, len: usize) -> Result<usize> {
    check_bytes::<T>(len, testing::properties(client).memory.max_page_size)
}

/// Bytes for `len` elements of `T`, checked against overflow and `max` bytes.
fn check_bytes<T: CubeElement>(len: usize, max: u64) -> Result<usize> {
    let bytes = len.checked_mul(size_of::<T>()).ok_or(GridError::TooLarge)?;
    if u64::try_from(bytes).map_or(true, |b| b > max) {
        return Err(GridError::AllocationTooLarge { bytes, max });
    }
    Ok(bytes)
}

/// The first `len` elements of `bytes`. Copies through bytes when `bytes` is not aligned for
/// `T`, so a read-back view at any offset decodes.
fn decode<T: CubeElement>(bytes: &[u8], len: usize) -> Result<Vec<T>> {
    let need = len.checked_mul(size_of::<T>()).ok_or(GridError::TooLarge)?;
    let Some(bytes) = bytes.get(..need) else {
        return Err(GridError::LengthMismatch {
            expected: need,
            found: bytes.len(),
        });
    };
    if bytes.as_ptr().align_offset(align_of::<T>()) == 0 {
        return Ok(T::from_bytes(bytes).to_vec());
    }
    let mut out = Vec::<T>::with_capacity(len);
    // SAFETY: `out` has capacity for `len` elements, `need = len * size_of::<T>()` bytes, and
    // `bytes` holds exactly `need` bytes; the regions do not overlap. `T: CubeElement` is
    // `bytemuck::Pod`, so every bit pattern is a valid `T` and the `len` elements are
    // initialised after the copy.
    unsafe {
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr().cast::<u8>(), need);
        out.set_len(len);
    }
    Ok(out)
}

/// Fixed number of entries per row (cell -> vertices, etc.).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceStrided};
///
/// let client = gridcl_testing::test_client();
/// let triangles = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 1, 3, 2]).unwrap(), stride: 3 };
/// assert_eq!(triangles.data.len() / triangles.stride as usize, 2);
/// ```
pub struct DeviceStrided<T: CubeElement> {
    pub data: DeviceBuffer<T>,
    pub stride: u32,
}

/// Variable-length rows (facet -> cells, vertex -> cells).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceCsr};
///
/// let client = gridcl_testing::test_client();
/// let csr = DeviceCsr {
///     offsets: DeviceBuffer::from_host(&client, &[0u32, 2, 3]).unwrap(),
///     indices: DeviceBuffer::from_host(&client, &[4u32, 5, 6]).unwrap(),
/// };
/// assert_eq!(csr.offsets.len() - 1, 2);
/// ```
pub struct DeviceCsr {
    pub offsets: DeviceBuffer<u32>,
    pub indices: DeviceBuffer<u32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pad_len_rounds_up_and_gives_empty_one_block() {
        assert_eq!(pad_len(0).unwrap(), VECTOR_PAD);
        assert_eq!(pad_len(1).unwrap(), VECTOR_PAD);
        assert_eq!(pad_len(16).unwrap(), 16);
        assert_eq!(pad_len(17).unwrap(), 32);
    }

    #[test]
    fn pad_len_overflow_is_too_large() {
        assert!(matches!(pad_len(usize::MAX), Err(GridError::TooLarge)));
        assert!(matches!(pad_len(usize::MAX - 14), Err(GridError::TooLarge)));
        let largest = usize::MAX - usize::MAX % VECTOR_PAD;
        assert_eq!(pad_len(largest).unwrap(), largest);
    }

    #[test]
    fn byte_checks() {
        assert_eq!(check_bytes::<u32>(256, 1024).unwrap(), 1024);
        assert!(matches!(
            check_bytes::<u32>(272, 1024),
            Err(GridError::AllocationTooLarge {
                bytes: 1088,
                max: 1024
            })
        ));
        assert!(matches!(
            check_bytes::<f64>(usize::MAX / 4, u64::MAX),
            Err(GridError::TooLarge)
        ));
        assert_eq!(check_bytes::<f64>(0, 0).unwrap(), 0);
    }

    #[test]
    fn decode_reads_unaligned_bytes() {
        let words = [1u32, 2, 3, 0xdead_beef];
        let mut bytes = vec![0u8];
        bytes.extend_from_slice(u32::as_bytes(&words));
        let shifted = &bytes[1..];
        assert_eq!(decode::<u32>(shifted, 4).unwrap(), words);
        assert_eq!(decode::<u32>(shifted, 2).unwrap(), &words[..2]);
        assert!(matches!(
            decode::<u32>(shifted, 5),
            Err(GridError::LengthMismatch {
                expected: 20,
                found: 16
            })
        ));
    }

    #[test]
    fn empty_buffer_allocates_one_block() {
        let client = gridcl_testing::test_client();
        let b = DeviceBuffer::<f32>::uninit(&client, 0).unwrap();
        assert_eq!(b.len(), 0);
        assert_eq!(b.padded_len(), VECTOR_PAD);
        assert_eq!(b.handle().size_in_used(), (VECTOR_PAD * 4) as u64);
        let b = DeviceBuffer::<u32>::from_host(&client, &[]).unwrap();
        assert_eq!(b.to_host().unwrap(), Vec::<u32>::new());
        assert_eq!(b.handle().size_in_used(), (VECTOR_PAD * 4) as u64);
    }
}
