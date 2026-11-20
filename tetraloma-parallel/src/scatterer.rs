//! `Scatterer`: the host-staged halo exchange with a deterministic reverse (docs/design.md,
//! "Transports"; ADR-0009, ADR-0026).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use tetraloma_device::{
    Client, CubeElement, DeviceBuffer, DeviceClass, DeviceScalar, ReduceOp, device_properties,
    gather, pack, segmented_reduce, unpack,
};

use crate::index_map::Exchange;
use crate::{DType, GridError, IndexMap, Neighbourhood, Result};

/// The device copies of an [`IndexMap`]'s index lists that do not depend on the block size,
/// in block indices. Cached by the map for the first client it is asked for (ADR-0026, item 14).
pub(crate) struct SharedIndices {
    client: Client,
    /// The owned local indices sent by forward, in send order.
    send_idx: DeviceBuffer<u32>,
    /// The ghost local indices received by forward, in receive order.
    recv_idx: DeviceBuffer<u32>,
    /// The distinct owned local indices that some rank ghosts, ascending: the reverse targets.
    targets: DeviceBuffer<u32>,
}

impl SharedIndices {
    fn new(client: &Client, e: &Exchange, targets: &[u32]) -> Result<Self> {
        Ok(SharedIndices {
            client: client.clone(),
            send_idx: DeviceBuffer::from_host(client, &e.send_idx)?,
            recv_idx: DeviceBuffer::from_host(client, &e.recv_idx)?,
            targets: DeviceBuffer::from_host(client, targets)?,
        })
    }
    /// The allocated bytes of the three lists.
    pub(crate) fn footprint_bytes(&self) -> usize {
        self.send_idx.footprint_bytes()
            + self.recv_idx.footprint_bytes()
            + self.targets.footprint_bytes()
    }
    /// Shares of the device buffers.
    fn share(&self) -> Self {
        SharedIndices {
            client: self.client.clone(),
            send_idx: self.send_idx.share(),
            recv_idx: self.recv_idx.share(),
            targets: self.targets.share(),
        }
    }
}

/// Waits for the work queued on `client`.
fn sync(client: &Client) -> Result<()> {
    let mut done = client.sync();
    let mut cx = std::task::Context::from_waker(std::task::Waker::noop());
    loop {
        if let std::task::Poll::Ready(r) = done.as_mut().poll(&mut cx) {
            return r.map_err(GridError::from);
        }
        std::thread::yield_now();
    }
}

/// Whether two clients drive the same device server (they share its properties).
fn same_device(a: &Client, b: &Client) -> bool {
    Arc::ptr_eq(&a.properties_shared(), &b.properties_shared())
}

// Design: "Transports"; ADR-0009, ADR-0026.
/// The halo exchange of one [`IndexMap`] on device arrays, made by [`IndexMap::scatterer`]:
/// [`forward`](Self::forward) sends the owners' values to their ghosts, and
/// [`reverse`](Self::reverse) combines the ghosts' values into their owners'.
///
/// An array holds `block` values per local index, the owned blocks first and the ghost blocks after
/// them. The exchange stays on the device except for the values that cross between ranks: a kernel
/// packs the values to send, they are copied to the host and sent by MPI to the neighbouring ranks
/// only, and the received values are uploaded and unpacked by another kernel. A scatterer keeps its
/// buffers for reuse, so make it once and use it for every exchange ([Halo
/// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#scatterers)).
///
/// # Example
///
/// ```
/// use tetraloma_device::{Client, DeviceBuffer};
/// use tetraloma_parallel::{CommDyn, IndexMap, ReverseOp, Result};
///
/// /// Each rank owns two indices and holds the first index of the next rank, round the ring,
/// /// as a ghost. Returns the values after the forward and after the reverse exchange.
/// fn ring(comm: &dyn CommDyn, client: &Client) -> Result<(Vec<f32>, Vec<f32>)> {
///     let next = (comm.rank() + 1) % comm.size();
///     let map = IndexMap::with_ghosts(comm, 2, vec![2 * next as i64], vec![next as i32])?;
///     let scatterer = map.scatterer(client, 1)?;
///     // The owned values are the global numbers plus 1; the ghost starts at 0.
///     let first = map.owned_range().0 as f32;
///     let mut x = DeviceBuffer::from_host(client, &[first + 1.0, first + 2.0, 0.0])?;
///     // Owners to ghosts: the ghost receives its owner's value.
///     scatterer.forward(&mut x)?;
///     let forward = x.to_host()?;
///     // Ghosts to owners: each owner adds the values of its ghosts.
///     scatterer.reverse(&mut x, ReverseOp::Add)?;
///     Ok((forward, x.to_host()?))
/// }
///
/// // `results`: `ring` called on each of two ranks, run as two threads of this process by a
/// // test communicator, each with its own `client`. Under MPI, each process calls `ring` with
/// // its `Mpi`.
/// # let results = tetraloma_testing::ThreadComm::run(2, |comm| {
/// #     ring(&comm, &tetraloma_testing::test_client()).unwrap()
/// # });
/// // Rank 0 owns global indices 0 and 1 and ghosts index 2; rank 1 owns 2 and 3 and ghosts 0.
/// assert_eq!(results[0], (vec![1.0, 2.0, 3.0], vec![2.0, 2.0, 3.0]));
/// assert_eq!(results[1], (vec![3.0, 4.0, 1.0], vec![6.0, 4.0, 1.0]));
/// ```
pub struct Scatterer {
    transport: Transport,
    exchange: Rc<Exchange>,
    block: usize,
    /// Values per rank: `num_local · block`.
    len: usize,
    indices: SharedIndices,
    /// Segment offsets of the reverse buffer, one segment per target component: its own
    /// value, then its contributions in receive order.
    rev_segments: DeviceBuffer<u32>,
    /// Element index in the reverse buffer of each target component's own value, in the
    /// order of `pack(x, targets)`.
    rev_own: DeviceBuffer<u32>,
    /// Element index in the reverse buffer of each received value.
    rev_received: DeviceBuffer<u32>,
    /// Element index in the received values of each target component's first contribution
    /// (`Insert`).
    rev_first: DeviceBuffer<u32>,
    /// The device and host buffers of each scalar type, reused across calls ([`Buffers`]).
    buffers: RefCell<Vec<Box<dyn Any>>>,
}

/// The buffers of one scalar type that a [`Scatterer`] reuses across calls, all on the device
/// of `owned`. The lengths are fixed by the map and the block: `owned` and `host_owned` hold
/// the blocks of the send list (forward's packed send, reverse's received values), `ghost` and
/// `host_ghost` those of the ghosts (forward's received values, reverse's packed send);
/// `own` the reverse targets' own blocks and `all` the reverse buffer, the targets' own and
/// received values in segment order.
struct Buffers<T: CubeElement> {
    owned: DeviceBuffer<T>,
    ghost: DeviceBuffer<T>,
    host_owned: Vec<T>,
    host_ghost: Vec<T>,
    own: DeviceBuffer<T>,
    all: DeviceBuffer<T>,
}

impl Scatterer {
    /// The scatterer of `map` for `block` values per index on `client` (`IndexMap::scatterer`).
    pub(crate) fn new(map: &IndexMap, client: &Client, block: usize) -> Result<Scatterer> {
        if block == 0 {
            return Err(GridError::LengthMismatch {
                expected: 1,
                found: 0,
            });
        }
        let e = map.exchange_rc();
        let words = |n: usize| -> Result<usize> {
            let w = n.checked_mul(block).ok_or(GridError::TooLarge)?;
            u32::try_from(w).map_err(|_| GridError::TooLarge)?;
            Ok(w)
        };
        let len = words(map.num_local())?;

        let mut targets = e.send_idx.clone();
        targets.sort_unstable();
        targets.dedup();
        let indices = match map.device_cache().get() {
            Some(cached) if same_device(&cached.client, client) => cached.share(),
            Some(_) => SharedIndices::new(client, &e, &targets)?,
            None => {
                let fresh = SharedIndices::new(client, &e, &targets)?;
                map.device_cache().get_or_init(|| fresh).share()
            }
        };

        // The reverse buffer holds the T target blocks, then the R received blocks.
        let t = targets.len();
        let r = e.send_idx.len();
        words(t.checked_add(r).ok_or(GridError::TooLarge)?)?;
        let mut contributions: Vec<Vec<usize>> = vec![Vec::new(); t];
        for (j, s) in e.send_idx.iter().enumerate() {
            let k = targets
                .binary_search(s)
                .expect("every send index is a target");
            contributions[k].push(j);
        }
        // Segment (k, c) holds target k's own component c, then component c of each block j
        // that k receives, in receive order. All indices are below (t + r) · block, which fits
        // u32 (checked above).
        let mut segments = Vec::with_capacity(t * block + 1);
        let mut own = Vec::with_capacity(t * block);
        let mut received = vec![0_u32; r * block];
        let mut first = Vec::with_capacity(t * block);
        let mut at = 0_usize;
        segments.push(0_u32);
        for js in &contributions {
            for c in 0..block {
                own.push(at as u32);
                for (i, &j) in js.iter().enumerate() {
                    received[j * block + c] = (at + 1 + i) as u32;
                }
                at += 1 + js.len();
                segments.push(at as u32);
                first.push((js[0] * block + c) as u32);
            }
        }
        Ok(Scatterer {
            transport: Transport::HostStaged,
            exchange: e,
            block,
            len,
            indices,
            rev_segments: DeviceBuffer::from_host(client, &segments)?,
            rev_own: DeviceBuffer::from_host(client, &own)?,
            rev_received: DeviceBuffer::from_host(client, &received)?,
            rev_first: DeviceBuffer::from_host(client, &first)?,
            buffers: RefCell::new(Vec::new()),
        })
    }

    /// This scatterer's buffers of T on `client`'s device, taken out of the cache (and put
    /// back by [`put_buffers`](Self::put_buffers)), or new ones.
    fn take_buffers<T: CubeElement>(&self, client: &Client) -> Result<Buffers<T>> {
        let mut cache = self.buffers.borrow_mut();
        if let Some(i) = cache.iter().position(|b| b.is::<Buffers<T>>()) {
            let b = cache.swap_remove(i);
            let b = *b.downcast::<Buffers<T>>().expect("checked by `is`");
            if same_device(b.owned.client(), client) {
                return Ok(b);
            }
        }
        drop(cache);
        let blocks = |n: usize| n * self.block;
        let (sent, ghosts) = (self.indices.send_idx.len(), self.indices.recv_idx.len());
        let targets = self.indices.targets.len();
        let zero = <T as bytemuck::Zeroable>::zeroed();
        Ok(Buffers {
            owned: DeviceBuffer::uninit(client, blocks(sent))?,
            ghost: DeviceBuffer::uninit(client, blocks(ghosts))?,
            host_owned: vec![zero; blocks(sent)],
            host_ghost: vec![zero; blocks(ghosts)],
            own: DeviceBuffer::uninit(client, blocks(targets))?,
            all: DeviceBuffer::uninit(client, blocks(targets + sent))?,
        })
    }

    /// Returns `b` to the cache.
    fn put_buffers<T: CubeElement>(&self, b: Buffers<T>) {
        self.buffers.borrow_mut().push(Box::new(b));
    }

    /// Owners to ghosts: every ghost block of `x` receives its owner's block, and the owned blocks
    /// are unchanged. Collective over the map's communicator ([Halo
    /// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#forward)).
    ///
    /// The values move as 32-bit words, so `size_of::<T>()` must be a multiple of 4, which the
    /// compiler checks, and any such `T` works on every device.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `x` does not hold `block` values per local index;
    /// `MessageTooLarge` and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{Client, DeviceBuffer};
    /// use tetraloma_parallel::{CommDyn, IndexMap, Result};
    ///
    /// /// Each rank owns one index, with two values, and holds the other rank's as a ghost.
    /// fn swap(comm: &dyn CommDyn, client: &Client) -> Result<Vec<u32>> {
    ///     let (rank, other) = (comm.rank(), 1 - comm.rank());
    ///     let map = IndexMap::with_ghosts(comm, 1, vec![other as i64], vec![other as i32])?;
    ///     let scatterer = map.scatterer(client, 2)?;
    ///     let r = rank as u32;
    ///     let mut x = DeviceBuffer::from_host(client, &[10 * r, 10 * r + 1, 0, 0])?;
    ///     scatterer.forward(&mut x)?;
    ///     x.to_host()
    /// }
    ///
    /// // `results`: `swap` called on each of two ranks, run as two threads of this process
    /// // by a test communicator. Under MPI, each process calls `swap` with its `Mpi`.
    /// # let results = tetraloma_testing::ThreadComm::run(2, |comm| {
    /// #     swap(&comm, &tetraloma_testing::test_client()).unwrap()
    /// # });
    /// assert_eq!(results, [vec![0u32, 1, 10, 11], vec![10, 11, 0, 1]]);
    /// ```
    pub fn forward<T: CubeElement>(&self, x: &mut DeviceBuffer<T>) -> Result<()> {
        const { assert!(size_of::<T>().is_multiple_of(4), "forward moves u32 words") };
        self.check_len(x.len())?;
        let e = &self.exchange;
        if e.forward.comm().is_none() {
            return Ok(());
        }
        let mut b = self.take_buffers::<T>(x.client())?;
        let result = (|| {
            let Buffers {
                owned,
                ghost,
                host_ghost,
                ..
            } = &mut b;
            self.exchange_blocks(
                x,
                &self.indices.send_idx,
                &e.forward,
                &e.send_counts,
                &e.recv_counts,
                owned,
                host_ghost,
                || Ok(()),
            )?;
            ghost.write_host(host_ghost)?;
            unpack(ghost, &self.indices.recv_idx, self.block, x)
        })();
        self.put_buffers(b);
        result
    }

    // ADR-0009, ADR-0026 (segmented reduce, order of the contributions); ADR-0059 (only the
    // received values are uploaded; the targets' own values are packed on the device, during
    // the exchange on GPUs).
    /// Ghosts to owners: each owned value that some rank holds as a ghost is combined by `op`
    /// with the values of all its ghost copies. Ghost values, and owned values that no rank
    /// ghosts, are unchanged. Collective over the map's communicator ([Halo
    /// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#reverse)).
    ///
    /// The values are combined on the device in a fixed order, by neighbouring rank and then by
    /// the ghost's position on that rank, without atomic operations, so a floating-point sum
    /// gives the same bits on every run. Only the received values are uploaded.
    ///
    /// # Errors
    ///
    /// `UnsupportedPrecision` if the device does not support `T`; `LengthMismatch` if `x` does
    /// not hold `block` values per local index; `MessageTooLarge` and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{Client, DeviceBuffer};
    /// use tetraloma_parallel::{CommDyn, IndexMap, Result, ReverseOp};
    ///
    /// /// Each rank owns one index and holds the other rank's as a ghost; every value is its
    /// /// rank plus 1. Returns the values after a reverse maximum.
    /// fn largest(comm: &dyn CommDyn, client: &Client) -> Result<Vec<i32>> {
    ///     let other = 1 - comm.rank();
    ///     let map = IndexMap::with_ghosts(comm, 1, vec![other as i64], vec![other as i32])?;
    ///     let scatterer = map.scatterer(client, 1)?;
    ///     let v = comm.rank() as i32 + 1;
    ///     let mut x = DeviceBuffer::from_host(client, &[v, v])?;
    ///     scatterer.reverse(&mut x, ReverseOp::Max)?;
    ///     x.to_host()
    /// }
    ///
    /// // `results`: `largest` called on each of two ranks, run as two threads of this process
    /// // by a test communicator. Under MPI, each process calls `largest` with its `Mpi`.
    /// # let results = tetraloma_testing::ThreadComm::run(2, |comm| {
    /// #     largest(&comm, &tetraloma_testing::test_client()).unwrap()
    /// # });
    /// // Each owner takes the larger of its value and its ghost's; ghosts are unchanged.
    /// assert_eq!(results, [vec![2, 1], vec![2, 2]]);
    /// ```
    pub fn reverse<T: ScatterNumeric>(&self, x: &mut DeviceBuffer<T>, op: ReverseOp) -> Result<()> {
        let client = x.client().clone();
        if !device_properties(&client).supports_type(T::cube_type()) {
            return Err(GridError::UnsupportedPrecision {
                scalar: T::type_name(),
                device: client.name(),
            });
        }
        self.check_len(x.len())?;
        let e = &self.exchange;
        if e.reverse.comm().is_none() {
            return Ok(());
        }
        let mut b = self.take_buffers::<T>(&client)?;
        let result = (|| {
            let Buffers {
                owned,
                ghost,
                host_owned,
                own,
                all,
                ..
            } = &mut b;
            // Without targets nobody ghosts this rank's indices: it only sends.
            let targets = self.indices.targets.len() > 0;
            let combine = targets && op != ReverseOp::Insert;
            // The CPU runtime's workers spin on queued kernels whose predecessors are still
            // running, which stalls for whole time slices when ranks share the cores (C24):
            // there each kernel is waited for, and nothing is queued during the exchange.
            let cpu = DeviceClass::for_client(&client) == DeviceClass::Cpu;
            let step = || if cpu { sync(&client) } else { Ok(()) };
            let pack_own = |own: &mut DeviceBuffer<T>, all: &mut DeviceBuffer<T>| -> Result<()> {
                pack(x, &self.indices.targets, self.block, own)?;
                step()?;
                unpack(own, &self.rev_own, 1, all)?;
                step()
            };
            self.exchange_blocks(
                x,
                &self.indices.recv_idx,
                &e.reverse,
                &e.recv_counts,
                &e.send_counts,
                ghost,
                host_owned,
                // On GPUs the targets' own values are packed during the exchange: `x` is
                // read-only until the final unpack.
                || {
                    if combine && !cpu {
                        pack_own(own, all)?;
                    }
                    Ok(())
                },
            )?;
            if !targets {
                return Ok(());
            }
            if combine && cpu {
                pack_own(own, all)?;
            }
            owned.write_host(host_owned)?;
            step()?;
            let values = match op {
                ReverseOp::Insert => gather(owned, &self.rev_first)?,
                ReverseOp::Add | ReverseOp::Min | ReverseOp::Max => {
                    let reduce = match op {
                        ReverseOp::Add => ReduceOp::Sum,
                        ReverseOp::Min => ReduceOp::Min,
                        _ => ReduceOp::Max,
                    };
                    unpack(owned, &self.rev_received, 1, all)?;
                    step()?;
                    segmented_reduce(all, &self.rev_segments, reduce)?
                }
            };
            step()?;
            // The targets are distinct: `unpack` writes each target block once.
            unpack(&values, &self.indices.targets, self.block, x)
        })();
        self.put_buffers(b);
        result
    }

    /// How this scatterer moves values between ranks: always [`Transport::HostStaged`] ([Halo
    /// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#scatterers)).
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_parallel::{GridError, IndexMap, Serial, Transport};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let scatterer = IndexMap::new(&Serial, 5)?.scatterer(&client, 1)?;
    /// assert_eq!(scatterer.transport(), Transport::HostStaged);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn transport(&self) -> Transport {
        self.transport
    }

    fn check_len(&self, found: usize) -> Result<()> {
        if found != self.len {
            return Err(GridError::LengthMismatch {
                expected: self.len,
                found,
            });
        }
        Ok(())
    }

    /// Packs the blocks `idx` of `x` into `send`, copies them to the host, runs `meanwhile`
    /// (device work queued behind the copy, which runs during the exchange), sends
    /// `send_counts` blocks to each destination of `nb` and receives `recv_counts` blocks from
    /// each source into `recv`, in receive order. The map has an exchange (`nb.comm()` is
    /// `Some`), and `send` and `recv` hold `idx.len()` and the received blocks.
    #[allow(clippy::too_many_arguments)]
    fn exchange_blocks<T: CubeElement>(
        &self,
        x: &DeviceBuffer<T>,
        idx: &DeviceBuffer<u32>,
        nb: &Neighbourhood,
        send_counts: &[usize],
        recv_counts: &[usize],
        send: &mut DeviceBuffer<T>,
        recv: &mut [T],
        meanwhile: impl FnOnce() -> Result<()>,
    ) -> Result<()> {
        let comm = nb.comm().expect("the map has an exchange");
        pack(x, idx, self.block, send)?;
        let host = send.to_host()?;
        meanwhile()?;
        // Counts in u32 words.
        let per_block = self
            .block
            .checked_mul(size_of::<T>() / 4)
            .ok_or(GridError::TooLarge)?;
        let words = |counts: &[usize]| -> Result<Vec<usize>> {
            counts
                .iter()
                .map(|&c| c.checked_mul(per_block).ok_or(GridError::TooLarge))
                .collect()
        };
        comm.neighbour_all_to_all_v_bytes(
            nb,
            bytemuck::cast_slice(&host),
            &words(send_counts)?,
            bytemuck::cast_slice_mut(recv),
            &words(recv_counts)?,
            DType::U32,
        )
    }
}

// ADR-0011, ADR-0026.
/// How [`Scatterer::reverse`] combines the values of an index's ghost copies with its owner's
/// value.
///
/// `Add`, `Min` and `Max` combine the owner's value with the value of every ghost copy in a
/// fixed order; `Insert` replaces it by the value from the lowest neighbouring rank, and leaves
/// it as it is if no rank holds a ghost copy. Ghost values are unchanged ([Halo
/// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#reverse)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReverseOp {
    /// The sum.
    Add,
    /// The minimum.
    Min,
    /// The maximum.
    Max,
    /// The contribution of the lowest neighbour rank.
    Insert,
}

/// The scalar types of [`Scatterer::reverse`]: `f32`, `f64`, `i32` and `u32`. `f64` works only on
/// devices that support it ([Halo
/// exchange](https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#reverse)).
pub trait ScatterNumeric: DeviceScalar {}

impl ScatterNumeric for f32 {}
impl ScatterNumeric for f64 {}
impl ScatterNumeric for i32 {}
impl ScatterNumeric for u32 {}

// ADR-0026, ADR-0059.
/// How a [`Scatterer`] moves values between ranks. `HostStaged` is the only transport in use;
/// `GpuAware` is reserved for a later version ([Reserved
/// features](https://tetracl.github.io/tetraloma/features.html#reserved-features)).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Pack on the device, exchange on the host, unpack on the device.
    HostStaged,
    /// Reserved: device memory handed directly to a CUDA-aware MPI. No scatterer uses it yet.
    GpuAware,
}
