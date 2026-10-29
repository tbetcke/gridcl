//! `Scatterer`: the host-staged halo exchange with a deterministic reverse (docs/design.md,
//! "Transports"; ADR-0009, ADR-0026).

use std::rc::Rc;
use std::sync::Arc;

use gridcl_device::{
    Client, CubeElement, DeviceBuffer, DeviceScalar, ReduceOp, device_properties, gather, pack,
    scatter, segmented_reduce, unpack,
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
    /// 0..targets.len(): where the targets' own values go in the reverse buffer.
    target_slots: DeviceBuffer<u32>,
}

impl SharedIndices {
    fn new(client: &Client, e: &Exchange, targets: &[u32]) -> Result<Self> {
        let slots: Vec<u32> = (0..targets.len() as u32).collect();
        Ok(SharedIndices {
            client: client.clone(),
            send_idx: DeviceBuffer::from_host(client, &e.send_idx)?,
            recv_idx: DeviceBuffer::from_host(client, &e.recv_idx)?,
            targets: DeviceBuffer::from_host(client, targets)?,
            target_slots: DeviceBuffer::from_host(client, &slots)?,
        })
    }
    /// Shares of the device buffers.
    fn share(&self) -> Self {
        SharedIndices {
            client: self.client.clone(),
            send_idx: self.send_idx.share(),
            recv_idx: self.recv_idx.share(),
            targets: self.targets.share(),
            target_slots: self.target_slots.share(),
        }
    }
}

/// Whether two clients drive the same device server (they share its properties).
fn same_device(a: &Client, b: &Client) -> bool {
    Arc::ptr_eq(&a.properties_shared(), &b.properties_shared())
}

/// The forward (owner → ghosts) and reverse (ghosts → owner) exchange of one `IndexMap`,
/// built by [`IndexMap::scatterer`].
///
/// Values are stored `block` per local index, `[owned | ghost]`.
///
/// # Example
///
/// ```
/// use gridcl_device::{Client, DeviceBuffer};
/// use gridcl_parallel::{IndexMap, ReverseOp, Result, Serial, Transport};
///
/// fn round_trip(client: &Client) -> Result<()> {
///     let map = IndexMap::new(&Serial, 3)?;
///     let scatterer = map.scatterer(client, 2)?;
///     assert_eq!(scatterer.transport(), Transport::HostStaged);
///     let mut x = DeviceBuffer::from_host(client, &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0])?;
///     scatterer.forward(&mut x)?;
///     scatterer.reverse(&mut x, ReverseOp::Add)?;
///     // One rank: nothing to exchange.
///     assert_eq!(x.to_host()?, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
///     Ok(())
/// }
/// ```
pub struct Scatterer {
    transport: Transport,
    exchange: Rc<Exchange>,
    block: usize,
    /// Values per rank: `num_local · block`.
    len: usize,
    indices: SharedIndices,
    /// Element indices into the reverse buffer `[target values | received values]`: each
    /// target component's own value, then its contributions in receive order.
    rev_order: DeviceBuffer<u32>,
    /// Segment offsets of `rev_order`, one segment per target component.
    rev_segments: DeviceBuffer<u32>,
    /// Element index of each target component's first contribution (`Insert`).
    rev_first: DeviceBuffer<u32>,
    /// Element index in `x` of each target component.
    rev_targets: DeviceBuffer<u32>,
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
        let mut order = Vec::with_capacity((t + r) * block);
        let mut segments = Vec::with_capacity(t * block + 1);
        let mut first = Vec::with_capacity(t * block);
        let mut target_elems = Vec::with_capacity(t * block);
        segments.push(0_u32);
        for (k, js) in contributions.iter().enumerate() {
            for c in 0..block {
                // All below (t + r) · block, which fits u32 (checked above).
                order.push((k * block + c) as u32);
                order.extend(js.iter().map(|&j| ((t + j) * block + c) as u32));
                segments.push(order.len() as u32);
                first.push(((t + js[0]) * block + c) as u32);
                target_elems.push((targets[k] as usize * block + c) as u32);
            }
        }
        Ok(Scatterer {
            transport: Transport::HostStaged,
            exchange: e,
            block,
            len,
            indices,
            rev_order: DeviceBuffer::from_host(client, &order)?,
            rev_segments: DeviceBuffer::from_host(client, &segments)?,
            rev_first: DeviceBuffer::from_host(client, &first)?,
            rev_targets: DeviceBuffer::from_host(client, &target_elems)?,
        })
    }

    /// Owner → ghosts: every ghost block receives its owner's block. Owned values are
    /// unchanged. T moves as u32 words, so `size_of::<T>()` must be a multiple of 4 (checked at
    /// compile time) and any such T works on every device. Collective over the map's
    /// communicator.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `x` does not hold `block` values per local index;
    /// `MessageTooLarge` and device errors.
    pub fn forward<T: CubeElement>(&self, x: &mut DeviceBuffer<T>) -> Result<()> {
        const { assert!(size_of::<T>().is_multiple_of(4), "forward moves u32 words") };
        self.check_len(x.len())?;
        let e = &self.exchange;
        let ghosts = self.exchange_blocks(
            x,
            &self.indices.send_idx,
            &e.forward,
            &e.send_counts,
            &e.recv_counts,
        )?;
        if let Some(ghosts) = ghosts {
            let mut received = DeviceBuffer::<T>::uninit(x.client(), ghosts.len())?;
            received.write_host(&ghosts)?;
            unpack(&received, &self.indices.recv_idx, self.block, x)?;
        }
        Ok(())
    }

    /// Ghosts → owner by a segmented reduce: deterministic and free of atomics. Each owned
    /// value that some rank ghosts is combined by `op` with every ghost contribution, ordered
    /// by neighbour rank and then by ghost position on that rank (ADR-0009, ADR-0026). Ghost
    /// values and owned values nobody ghosts are unchanged. Collective over the map's
    /// communicator.
    ///
    /// # Errors
    ///
    /// `UnsupportedPrecision` if the device lacks T; `LengthMismatch` if `x` does not hold
    /// `block` values per local index; `MessageTooLarge` and device errors.
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
        let received = self.exchange_blocks(
            x,
            &self.indices.recv_idx,
            &e.reverse,
            &e.recv_counts,
            &e.send_counts,
        )?;
        let Some(received) = received else {
            return Ok(());
        };
        let own_len = self.indices.targets.len() * self.block;
        if own_len == 0 {
            return Ok(());
        }
        // [target values | received values]: the received values are uploaded behind room
        // for the targets' own values, which are then packed and unpacked into place.
        let mut host = vec![<T as bytemuck::Zeroable>::zeroed(); own_len];
        host.extend_from_slice(&received);
        let mut all = DeviceBuffer::<T>::uninit(&client, host.len())?;
        all.write_host(&host)?;
        let mut own = DeviceBuffer::<T>::uninit(&client, own_len)?;
        pack(x, &self.indices.targets, self.block, &mut own)?;
        unpack(&own, &self.indices.target_slots, self.block, &mut all)?;
        let values = match op {
            ReverseOp::Insert => gather(&all, &self.rev_first)?,
            ReverseOp::Add | ReverseOp::Min | ReverseOp::Max => {
                let reduce = match op {
                    ReverseOp::Add => ReduceOp::Sum,
                    ReverseOp::Min => ReduceOp::Min,
                    _ => ReduceOp::Max,
                };
                segmented_reduce(&gather(&all, &self.rev_order)?, &self.rev_segments, reduce)?
            }
        };
        scatter(&values, &self.rev_targets, x)
    }

    /// The transport this scatterer exchanges through.
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

    /// Packs the blocks `idx` of `x`, sends `send_counts` blocks to each destination of `nb` and
    /// receives `recv_counts` blocks from each source, in receive order. `None` when the map
    /// has no exchange (a map made by `IndexMap::new`).
    fn exchange_blocks<T: CubeElement>(
        &self,
        x: &DeviceBuffer<T>,
        idx: &DeviceBuffer<u32>,
        nb: &Neighbourhood,
        send_counts: &[usize],
        recv_counts: &[usize],
    ) -> Result<Option<Vec<T>>> {
        let Some(comm) = nb.comm() else {
            return Ok(None);
        };
        let client = x.client();
        let mut packed = DeviceBuffer::<T>::uninit(client, idx.len() * self.block)?;
        pack(x, idx, self.block, &mut packed)?;
        let send = packed.to_host()?;
        // Counts in u32 words.
        let per_block = self.block * size_of::<T>() / 4;
        let words = |counts: &[usize]| -> Result<Vec<usize>> {
            counts
                .iter()
                .map(|&c| c.checked_mul(per_block).ok_or(GridError::TooLarge))
                .collect()
        };
        let total: usize = recv_counts.iter().sum::<usize>() * self.block;
        let mut recv = vec![<T as bytemuck::Zeroable>::zeroed(); total];
        comm.neighbour_all_to_all_v_bytes(
            nb,
            bytemuck::cast_slice(&send),
            &words(send_counts)?,
            bytemuck::cast_slice_mut(&mut recv),
            &words(recv_counts)?,
            DType::U32,
        )?;
        Ok(Some(recv))
    }
}

/// How [`Scatterer::reverse`] combines the ghost contributions with the owner's value.
///
/// `Add`, `Min` and `Max` combine the owner's value with every ghost contribution in a fixed
/// order; `Insert` replaces it by the lowest neighbour rank's contribution (unchanged without
/// any). Ghost values are unchanged by reverse; empty segments use the ADR-0011 identities
/// (ADR-0026).
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

/// The scalars of [`Scatterer::reverse`]: f32, f64, i32 and u32.
pub trait ScatterNumeric: DeviceScalar {}

impl ScatterNumeric for f32 {}
impl ScatterNumeric for f64 {}
impl ScatterNumeric for i32 {}
impl ScatterNumeric for u32 {}

/// How a [`Scatterer`] moves data between ranks. M6 always chooses `HostStaged`; M10 adds
/// `GpuAware` (ADR-0026).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transport {
    /// Pack on the device, exchange on the host, unpack on the device.
    HostStaged,
    /// Device pointers handed to a CUDA-aware MPI (M10).
    GpuAware,
}
