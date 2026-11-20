//! The `DeviceBuffer` family (Listing S1).

use core::marker::PhantomData;

use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::primitives::launch_fill;
use crate::{GridError, Result, testing};

/// The padding of every device buffer: its allocation holds a multiple of `VECTOR_PAD` = 16
/// elements, at least one block, so vectorised kernels need no scalar tail.
///
/// 16 is the widest vector size of any device class. A kernel of your own may read up to the end of
/// the padding ([Writing
/// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, VECTOR_PAD};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let b = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
/// assert_eq!(b.padded_len(), VECTOR_PAD);
/// ```
pub const VECTOR_PAD: usize = 16;

// Listing S1.
/// A typed array of `len()` elements of `T` in device memory.
///
/// Meshes hand out their data as device buffers, and kernels read and write them. The
/// allocation is padded to a multiple of [`VECTOR_PAD`] elements. Copies between host and
/// device are explicit and blocking: [`from_host`](Self::from_host) and
/// [`write_host`](Self::write_host) upload, [`to_host`](Self::to_host) downloads ([Host–device
/// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
///
/// A buffer is not `Clone`: [`share`](Self::share) makes a second view of the same memory, and
/// [`deep_clone`](Self::deep_clone) an independent copy. Every constructor checks the
/// allocation against the largest allocation the device allows, its `max_page_size`, and
/// returns `GridError::AllocationTooLarge` above it.
///
/// # Example
///
/// ```
/// use tetraloma_device::DeviceBuffer;
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
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
    /// A buffer holding a copy of `data`, uploaded to `client`'s device; the padding is zero. The
    /// copy blocks until it is done ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Errors
    ///
    /// `AllocationTooLarge` if the padded buffer exceeds the device's largest allocation;
    /// `TooLarge` if its length overflows `usize`.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
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

    /// A buffer of `len` elements whose contents, padding included, are unspecified: only for
    /// outputs that a kernel writes completely. An empty buffer still allocates one `VECTOR_PAD`
    /// block, so no allocation has zero size ([Writing
    /// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
    ///
    /// # Errors
    ///
    /// As for [`from_host`](Self::from_host).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, VECTOR_PAD};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let b = DeviceBuffer::<f32>::uninit(&client, 0).unwrap();
    /// assert_eq!((b.len(), b.padded_len()), (0, VECTOR_PAD));
    /// ```
    pub fn uninit(client: &Client, len: usize) -> Result<Self> {
        let padded = pad_len(len)?;
        let bytes = check_allocation::<T>(client, padded)?;
        Ok(Self::new(client.empty(bytes), len, padded, client))
    }

    /// A buffer of `len` zeros, padding included, written on the device by a kernel: nothing is
    /// uploaded ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Errors
    ///
    /// As for [`from_host`](Self::from_host).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
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

    /// A second view of the same allocation: a write through either view changes both. Nothing is
    /// copied ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let a = DeviceBuffer::from_host(&client, &[1u32, 2]).unwrap();
    /// let b = a.share();
    /// assert_eq!(b.to_host().unwrap(), a.to_host().unwrap());
    /// ```
    pub fn share(&self) -> Self {
        Self::new(self.handle.clone(), self.len, self.padded, &self.client)
    }

    /// An independent copy on the same device, made by a copy kernel without a round trip through
    /// the host ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)). As for
    /// [`copy`](crate::copy), `size_of::<T>()` must be a multiple of 4, which the compiler checks.
    ///
    /// # Errors
    ///
    /// As for [`uninit`](Self::uninit).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
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

    /// The length in elements, not bytes, without the padding ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// assert_eq!(DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap().len(), 3);
    /// assert_eq!(DeviceBuffer::<u32>::uninit(&client, 0).unwrap().len(), 0);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// The allocated length in elements: `len` rounded up to a multiple of [`VECTOR_PAD`], at least
    /// one block ([Writing
    /// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, VECTOR_PAD};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// assert_eq!(DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap().padded_len(), VECTOR_PAD);
    /// assert_eq!(DeviceBuffer::<u32>::uninit(&client, 0).unwrap().padded_len(), VECTOR_PAD);
    /// assert_eq!(DeviceBuffer::<u32>::uninit(&client, 17).unwrap().padded_len(), 2 * VECTOR_PAD);
    /// ```
    pub fn padded_len(&self) -> usize {
        self.padded
    }

    /// The allocated bytes, `padded_len` elements. For memory accounting in tests
    /// (`Mesh::footprint_bytes`, ADR-0035); not part of the stable API.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, VECTOR_PAD};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let b = DeviceBuffer::<f64>::uninit(&client, 17).unwrap();
    /// assert_eq!(b.footprint_bytes(), 2 * VECTOR_PAD * 8);
    /// ```
    #[doc(hidden)]
    pub fn footprint_bytes(&self) -> usize {
        // `check_allocation` bounded padded × size_of::<T>() at construction.
        self.padded * size_of::<T>()
    }

    /// The CubeCL handle of the allocation, for CubeCL calls of your own; it keeps the allocation
    /// alive ([Writing
    /// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let x = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
    /// let bytes = client.read_one(x.handle().clone()).unwrap();
    /// assert_eq!(bytes.len(), x.padded_len() * size_of::<u32>());
    /// ```
    pub fn handle(&self) -> &Handle {
        &self.handle
    }

    /// The client of the device the buffer lives on ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let x = DeviceBuffer::from_host(&client, &[1.0f32, 2.0]).unwrap();
    /// let y = DeviceBuffer::from_host(x.client(), &[3.0f32, 4.0]).unwrap();
    /// assert_eq!(y.client().name(), client.name());
    /// ```
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// A copy of the `len()` elements on the host. It blocks until the device has finished the work
    /// queued before it and the copy is done, so use it to inspect results, not inside loops
    /// ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Errors
    ///
    /// `GridError::Server` if the device reports an error.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[3u32, 1, 4]).unwrap();
    /// assert_eq!(b.to_host().unwrap(), vec![3, 1, 4]);
    /// ```
    pub fn to_host(&self) -> Result<Vec<T>> {
        let bytes = self.client.read_one(self.handle.clone())?;
        decode(&bytes, self.len)
    }

    // ADR-0026: the write covers the whole allocation, which the CPU runtime requires; the
    // host-staged halo exchange uploads its receive buffer with it. Pinned staging through
    // `Client::staging` on CUDA and HIP uploads 2.4 MB about six times faster on CUDA (C23); on
    // the other runtimes it would only block until the device's queue reaches it.
    /// Overwrites the buffer with `data` from the host and zeroes the padding, as
    /// [`from_host`](DeviceBuffer::from_host) does, without a new allocation ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// `data` is copied once, into host memory of the padded length. On CUDA and HIP that copy
    /// is moved into pinned (page-locked) host memory before the upload, which makes large
    /// uploads several times faster. The CPU runtime and the wgpu runtimes (Metal, Vulkan) have
    /// no pinned memory and upload directly.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `data.len()` is not [`len`](DeviceBuffer::len).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, GridError};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let mut b = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
    /// b.write_host(&[7, 8, 9]).unwrap();
    /// assert_eq!(b.to_host().unwrap(), vec![7, 8, 9]);
    /// assert!(matches!(b.write_host(&[1]), Err(GridError::LengthMismatch { expected: 3, found: 1 })));
    /// ```
    pub fn write_host(&mut self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(GridError::LengthMismatch {
                expected: self.len,
                found: data.len(),
            });
        }
        if data.is_empty() {
            return Ok(());
        }
        let bytes = T::as_bytes(data);
        let padded = self.padded * size_of::<T>();
        let mut host = Vec::with_capacity(padded);
        host.extend_from_slice(bytes);
        host.resize(padded, 0);
        let mut host = cubecl::bytes::Bytes::from_bytes_vec(host);
        if stages_in_pinned_memory(&self.client) {
            self.client.staging(core::iter::once(&mut host), false);
        }
        self.client.write(&self.handle, host);
        Ok(())
    }

    /// A copy of the `len()` elements on the host, as a future: the asynchronous form of
    /// [`to_host`](Self::to_host) ([Host–device
    /// transfers](https://tetracl.github.io/tetraloma/devices.html#hostdevice-transfers)).
    ///
    /// # Errors
    ///
    /// As for [`to_host`](Self::to_host).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[2.0f32, 4.0]).unwrap();
    /// let host = cubecl::future::block_on(b.to_host_async()).unwrap();
    /// assert_eq!(host, vec![2.0, 4.0]);
    /// ```
    pub async fn to_host_async(&self) -> Result<Vec<T>> {
        let mut out = self.client.read_async(vec![self.handle.clone()]).await?;
        decode(&out.remove(0), self.len)
    }

    /// The buffer as a kernel argument of scalars, for a kernel of your own that takes it as `&[T]`
    /// or `&mut [T]` ([Writing
    /// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
    ///
    /// # Safety
    ///
    /// The kernel does not check its indices: the size of the allocation, not `len()`, bounds
    /// its slice. The kernel must stay within `len()` and the padding, and should take the
    /// lengths it needs as explicit arguments.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let b = DeviceBuffer::from_host(&client, &[1u32, 2, 3]).unwrap();
    /// // SAFETY: the argument is not passed to a kernel.
    /// let _arg = unsafe { b.arg() };
    /// ```
    pub unsafe fn arg(&self) -> BufferArg {
        // SAFETY: the handle holds `padded >= len` elements of `T`; the caller keeps the kernel
        // within them.
        unsafe { BufferArg::from_raw_parts(self.handle.clone(), self.len) }
    }

    /// The buffer as a kernel argument of vectors of `n` lanes, for a kernel of your own that
    /// takes it as `&[Vector<T, N>]`: the padded length, counted in vectors ([Writing
    /// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
    ///
    /// # Safety
    ///
    /// As for [`arg`](Self::arg); the kernel should take its number of vectors as an explicit
    /// argument. `n` must be a power of two that divides [`VECTOR_PAD`].
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::DeviceBuffer;
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
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

/// `len` rounded up to a multiple of [`VECTOR_PAD`], at least one block: the allocated length of
/// a buffer of `len` elements, and the plane stride of [`SortKeys`](crate::SortKeys).
///
/// An empty buffer still gets one block, because some backends reject bindings of size zero; its
/// `len()` stays 0 ([Writing
/// kernels](https://tetracl.github.io/tetraloma/tutorials/kernels.html#the-launch)).
///
/// # Errors
///
/// `TooLarge` if the rounded length overflows `usize`.
///
/// # Example
///
/// ```
/// use tetraloma_device::{VECTOR_PAD, pad_len};
///
/// assert_eq!(pad_len(0).unwrap(), VECTOR_PAD);
/// assert_eq!(pad_len(17).unwrap(), 2 * VECTOR_PAD);
/// ```
pub fn pad_len(len: usize) -> Result<usize> {
    len.max(1)
        .checked_next_multiple_of(VECTOR_PAD)
        .ok_or(GridError::TooLarge)
}

/// Whether `Client::staging` moves host data into pinned memory on `client`'s runtime: CUDA
/// and HIP. The CPU runtime and the wgpu runtimes return `UnsupportedIoOperation` after a
/// blocking submit (C23).
fn stages_in_pinned_memory(client: &Client) -> bool {
    matches!(client.name(), "cuda" | "hip")
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

/// Rows of a fixed length in one device buffer: row `i` is `data[i * stride..(i + 1) * stride]`.
///
/// A cell block's vertices and its downward connectivity, such as cell → edges, are strided:
/// every cell has the same number of them ([Meshes and
/// topology](https://tetracl.github.io/tetraloma/concepts/meshes.html#connectivity)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceStrided};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let triangles = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 1, 3, 2]).unwrap(), stride: 3 };
/// assert_eq!(triangles.data.len() / triangles.stride as usize, 2);
/// ```
pub struct DeviceStrided<T: CubeElement> {
    /// The rows, one after another.
    pub data: DeviceBuffer<T>,
    /// The number of entries in each row.
    pub stride: u32,
}

/// Rows of different lengths in compressed sparse row (CSR) form: row `i` is
/// `indices[offsets[i]..offsets[i + 1]]`.
///
/// Upward connectivity, such as facet → cells or vertex → cells, is stored this way, because a
/// boundary facet has one cell and an interior facet two ([Meshes and
/// topology](https://tetracl.github.io/tetraloma/concepts/meshes.html#connectivity)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceCsr};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// let csr = DeviceCsr {
///     offsets: DeviceBuffer::from_host(&client, &[0u32, 2, 3]).unwrap(),
///     indices: DeviceBuffer::from_host(&client, &[4u32, 5, 6]).unwrap(),
/// };
/// assert_eq!(csr.offsets.len() - 1, 2);
/// ```
pub struct DeviceCsr {
    /// Row `i` holds `indices[offsets[i]..offsets[i + 1]]`; one more entry than rows.
    pub offsets: DeviceBuffer<u32>,
    /// The entries of all rows, one row after another.
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

    /// `write_host` writes the padded allocation: the padding reads back as zeros, also where
    /// the memory held other values before (a released buffer of the same size).
    #[test]
    fn write_host_zeroes_the_padding() {
        let client = tetraloma_testing::test_client();
        for len in [1, 5, VECTOR_PAD, VECTOR_PAD + 3] {
            let padded = pad_len(len).unwrap();
            drop(DeviceBuffer::from_host(&client, &vec![u32::MAX; padded]).unwrap());
            let mut b = DeviceBuffer::<u32>::uninit(&client, len).unwrap();
            let data: Vec<u32> = (1..=len as u32).collect();
            b.write_host(&data).unwrap();
            b.write_host(&data).unwrap();
            let bytes = client.read_one(b.handle().clone()).unwrap();
            let words = decode::<u32>(&bytes, padded).unwrap();
            assert_eq!(&words[..len], &data[..], "len {len}");
            assert!(words[len..].iter().all(|&w| w == 0), "len {len}: {words:?}");
        }
    }

    /// `write_host` stages through pinned memory exactly on the runtimes that have it (C23):
    /// CUDA and HIP, not the CPU runtime, Metal or Vulkan.
    #[test]
    fn write_host_stages_only_where_staging_pins() {
        let client = tetraloma_testing::test_client();
        let device = std::env::var(tetraloma_testing::TEST_DEVICE_VAR).unwrap_or_default();
        assert_eq!(
            stages_in_pinned_memory(&client),
            matches!(device.as_str(), "cuda" | "hip"),
            "TETRALOMA_TEST_DEVICE={device:?}, runtime {}",
            client.name()
        );
    }

    #[test]
    fn empty_buffer_allocates_one_block() {
        let client = tetraloma_testing::test_client();
        let b = DeviceBuffer::<f32>::uninit(&client, 0).unwrap();
        assert_eq!(b.len(), 0);
        assert_eq!(b.padded_len(), VECTOR_PAD);
        assert_eq!(b.handle().size_in_used(), (VECTOR_PAD * 4) as u64);
        let b = DeviceBuffer::<u32>::from_host(&client, &[]).unwrap();
        assert_eq!(b.to_host().unwrap(), Vec::<u32>::new());
        assert_eq!(b.handle().size_in_used(), (VECTOR_PAD * 4) as u64);
    }
}
