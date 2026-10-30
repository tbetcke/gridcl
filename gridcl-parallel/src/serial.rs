//! `Serial`: the one-rank communicator with no-op collectives.

use crate::comm::{dtype_size, identity_bytes};
use crate::{CommDyn, DType, GridError, Neighbourhood, ReduceOp, Result};

/// One rank; collectives are no-ops.
///
/// - `rank()` is 0, `size()` is 1 and `barrier()` returns at once.
/// - `all_reduce_bytes` leaves `buf` unchanged.
/// - `exscan_bytes` overwrites every element with the identity of `op` for `dtype`: 0 for
///   `Sum`, the largest finite value for `Min` and the smallest finite value for `Max` (the
///   convention of gridcl-device's reductions; for floats `MAX`/`MIN`, not infinities).
/// - `all_to_all_v_bytes` copies `send` to `recv`. It returns `LengthMismatch` unless both
///   count slices have one entry, the two counts agree, and `send` and `recv` hold exactly
///   that many elements; `TooLarge` if a byte length overflows `usize`.
/// - `neighbour_all_to_all_v_bytes` accepts only [`Neighbourhood::empty`] with empty counts
///   and buffers. A neighbourhood with `k` sources or destinations returns
///   `LengthMismatch { expected: 0, found: k }`; a non-empty count slice or buffer returns
///   `LengthMismatch { expected: 0, found: its length }`.
/// - `raw()` (feature `mpi`) is `None`, and `duplicate()` is another `Serial`.
///
/// In debug builds the reduce and scan assert that `buf.len()` is a multiple of the size of
/// `dtype`.
///
/// # Example
///
/// ```
/// use gridcl_parallel::{CommDyn, CommExt, DType, GridError, Serial};
///
/// let comm = Serial;
/// assert_eq!(comm.sum(4_i64), 4);
///
/// let send = 1.0_f64.to_ne_bytes();
/// let mut recv = [0_u8; 8];
/// comm.all_to_all_v_bytes(&send, &[1], &mut recv, &[1], DType::F64).unwrap();
/// assert_eq!(f64::from_ne_bytes(recv), 1.0);
///
/// let err = comm.all_to_all_v_bytes(&send, &[1], &mut recv, &[2], DType::F64);
/// assert!(matches!(err, Err(GridError::LengthMismatch { .. })));
/// ```
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;

/// `Ok` if `found == expected`, else `LengthMismatch`.
fn check_len(expected: usize, found: usize) -> Result<()> {
    if expected == found {
        Ok(())
    } else {
        Err(GridError::LengthMismatch { expected, found })
    }
}

/// The byte length of `count` elements of `dtype`; `TooLarge` on overflow.
fn byte_len(count: usize, dtype: DType) -> Result<usize> {
    count
        .checked_mul(dtype_size(dtype))
        .ok_or(GridError::TooLarge)
}

impl CommDyn for Serial {
    fn rank(&self) -> usize {
        0
    }
    fn size(&self) -> usize {
        1
    }
    fn barrier(&self) {}
    fn all_reduce_bytes(&self, buf: &mut [u8], dtype: DType, _op: ReduceOp) {
        debug_assert!(
            buf.len().is_multiple_of(dtype_size(dtype)),
            "all_reduce_bytes: {} bytes are not a whole number of {dtype:?} elements",
            buf.len()
        );
    }
    fn exscan_bytes(&self, buf: &mut [u8], dtype: DType, op: ReduceOp) {
        let size = dtype_size(dtype);
        debug_assert!(
            buf.len().is_multiple_of(size),
            "exscan_bytes: {} bytes are not a whole number of {dtype:?} elements",
            buf.len()
        );
        let identity = identity_bytes(dtype, op);
        for element in buf.chunks_exact_mut(size) {
            element.copy_from_slice(&identity[..size]);
        }
    }
    fn all_to_all_v_bytes(
        &self,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        dtype: DType,
    ) -> Result<()> {
        check_len(1, send_counts.len())?;
        check_len(1, recv_counts.len())?;
        check_len(send_counts[0], recv_counts[0])?;
        let bytes = byte_len(send_counts[0], dtype)?;
        check_len(bytes, send.len())?;
        check_len(bytes, recv.len())?;
        recv.copy_from_slice(send);
        Ok(())
    }
    fn neighbour_all_to_all_v_bytes(
        &self,
        nb: &Neighbourhood,
        send: &[u8],
        send_counts: &[usize],
        recv: &mut [u8],
        recv_counts: &[usize],
        _dtype: DType,
    ) -> Result<()> {
        check_len(0, nb.sources().len())?;
        check_len(0, nb.destinations().len())?;
        check_len(0, send_counts.len())?;
        check_len(0, recv_counts.len())?;
        check_len(0, send.len())?;
        check_len(0, recv.len())?;
        Ok(())
    }
    #[cfg(feature = "mpi")]
    fn raw(&self) -> Option<mpi::ffi::MPI_Comm> {
        None
    }
    fn duplicate(&self) -> std::rc::Rc<dyn CommDyn> {
        std::rc::Rc::new(Serial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CommExt, CommScalar};
    use proptest::prelude::*;

    const DTYPES: [DType; 6] = [
        DType::I32,
        DType::I64,
        DType::U32,
        DType::U64,
        DType::F32,
        DType::F64,
    ];
    const OPS: [ReduceOp; 3] = [ReduceOp::Sum, ReduceOp::Min, ReduceOp::Max];

    fn config() -> ProptestConfig {
        ProptestConfig::with_cases(256)
    }

    /// The bytes of `n` copies of the identity, built from the Rust constants.
    fn expected_identity(dtype: DType, op: ReduceOp, n: usize) -> Vec<u8> {
        fn rep<T: bytemuck::Pod>(x: T, n: usize) -> Vec<u8> {
            bytemuck::cast_slice(&vec![x; n]).to_vec()
        }
        match (dtype, op) {
            (DType::I32, ReduceOp::Sum) => rep(0_i32, n),
            (DType::I32, ReduceOp::Min) => rep(i32::MAX, n),
            (DType::I32, ReduceOp::Max) => rep(i32::MIN, n),
            (DType::I64, ReduceOp::Sum) => rep(0_i64, n),
            (DType::I64, ReduceOp::Min) => rep(i64::MAX, n),
            (DType::I64, ReduceOp::Max) => rep(i64::MIN, n),
            (DType::U32, ReduceOp::Sum) => rep(0_u32, n),
            (DType::U32, ReduceOp::Min) => rep(u32::MAX, n),
            (DType::U32, ReduceOp::Max) => rep(u32::MIN, n),
            (DType::U64, ReduceOp::Sum) => rep(0_u64, n),
            (DType::U64, ReduceOp::Min) => rep(u64::MAX, n),
            (DType::U64, ReduceOp::Max) => rep(u64::MIN, n),
            (DType::F32, ReduceOp::Sum) => rep(0_f32, n),
            (DType::F32, ReduceOp::Min) => rep(f32::MAX, n),
            (DType::F32, ReduceOp::Max) => rep(f32::MIN, n),
            (DType::F64, ReduceOp::Sum) => rep(0_f64, n),
            (DType::F64, ReduceOp::Min) => rep(f64::MAX, n),
            (DType::F64, ReduceOp::Max) => rep(f64::MIN, n),
        }
    }

    #[test]
    fn rank_size_barrier_raw() {
        let comm: &dyn CommDyn = &Serial;
        assert_eq!(comm.rank(), 0);
        assert_eq!(comm.size(), 1);
        comm.barrier();
        #[cfg(feature = "mpi")]
        assert!(comm.raw().is_none());
    }

    #[test]
    fn exscan_of_empty_buffer_is_empty() {
        let mut buf: [u8; 0] = [];
        for dtype in DTYPES {
            for op in OPS {
                Serial.exscan_bytes(&mut buf, dtype, op);
                Serial.all_reduce_bytes(&mut buf, dtype, op);
            }
        }
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not a whole number")]
    fn reduce_of_partial_element_panics_in_debug() {
        let mut buf = [0_u8; 6];
        Serial.all_reduce_bytes(&mut buf, DType::F64, ReduceOp::Sum);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "not a whole number")]
    fn exscan_of_partial_element_panics_in_debug() {
        let mut buf = [0_u8; 5];
        Serial.exscan_bytes(&mut buf, DType::I32, ReduceOp::Sum);
    }

    #[test]
    fn comm_ext_extremes() {
        let comm: &dyn CommDyn = &Serial;
        assert_eq!(comm.sum(i64::MIN), i64::MIN);
        assert_eq!(comm.max(u64::MAX), u64::MAX);
        assert_eq!(comm.max(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert!(comm.sum(f32::NAN).is_nan());
        assert_eq!(comm.exscan_sum(f64::MAX), 0.0);
        assert_eq!(comm.exscan_sum(-0.0_f32).to_bits(), 0.0_f32.to_bits());
    }

    #[test]
    fn all_to_all_v_rejects_wrong_counts() {
        let send = [1_u8, 2, 3, 4, 5, 6, 7, 8];
        let mut recv = [0_u8; 8];
        let mismatch = |r: Result<()>, e: usize, f: usize| {
            assert!(
                matches!(r, Err(GridError::LengthMismatch { expected, found }) if expected == e && found == f),
                "{r:?}"
            )
        };
        mismatch(
            Serial.all_to_all_v_bytes(&send, &[], &mut recv, &[2], DType::U32),
            1,
            0,
        );
        mismatch(
            Serial.all_to_all_v_bytes(&send, &[2], &mut recv, &[2, 0], DType::U32),
            1,
            2,
        );
        mismatch(
            Serial.all_to_all_v_bytes(&send, &[2], &mut recv, &[1], DType::U32),
            2,
            1,
        );
        mismatch(
            Serial.all_to_all_v_bytes(&send, &[1], &mut recv, &[1], DType::U32),
            4,
            8,
        );
        mismatch(
            Serial.all_to_all_v_bytes(&send[..4], &[1], &mut recv, &[1], DType::U32),
            4,
            8,
        );
        mismatch(
            Serial.all_to_all_v_bytes(&send, &[2], &mut recv[..4], &[2], DType::U32),
            8,
            4,
        );
        assert!(matches!(
            Serial.all_to_all_v_bytes(&send, &[usize::MAX], &mut recv, &[usize::MAX], DType::U64),
            Err(GridError::TooLarge)
        ));
        assert_eq!(recv, [0; 8], "recv is untouched on error");
        Serial
            .all_to_all_v_bytes(&send, &[1], &mut recv, &[1], DType::U64)
            .unwrap();
        assert_eq!(recv, send);
    }

    #[test]
    fn neighbour_all_to_all_v_accepts_only_the_empty_neighbourhood() {
        let nb = Neighbourhood::empty();
        let mut empty: [u8; 0] = [];
        Serial
            .neighbour_all_to_all_v_bytes(&nb, &[], &[], &mut empty, &[], DType::F64)
            .unwrap();
        let mut recv = [0_u8; 4];
        let err = |r: Result<()>| matches!(r, Err(GridError::LengthMismatch { expected: 0, .. }));
        assert!(err(Serial.neighbour_all_to_all_v_bytes(
            &nb,
            &[],
            &[1],
            &mut empty,
            &[],
            DType::I32
        )));
        assert!(err(Serial.neighbour_all_to_all_v_bytes(
            &nb,
            &[],
            &[],
            &mut empty,
            &[1],
            DType::I32
        )));
        assert!(err(Serial.neighbour_all_to_all_v_bytes(
            &nb,
            &[0; 4],
            &[],
            &mut empty,
            &[],
            DType::I32
        )));
        assert!(err(Serial.neighbour_all_to_all_v_bytes(
            &nb,
            &[],
            &[],
            &mut recv,
            &[],
            DType::I32
        )));
        let non_empty = Neighbourhood::from_ranks(vec![0], vec![0]);
        assert!(matches!(
            Serial.neighbour_all_to_all_v_bytes(
                &non_empty,
                &[0; 4],
                &[1],
                &mut recv,
                &[1],
                DType::I32
            ),
            Err(GridError::LengthMismatch {
                expected: 0,
                found: 1
            })
        ));
    }

    fn comm_ext_identity<T: CommScalar + PartialEq + core::fmt::Debug>(x: T, zero: T) {
        let comm: &dyn CommDyn = &Serial;
        assert_eq!(bytemuck::bytes_of(&comm.sum(x)), bytemuck::bytes_of(&x));
        assert_eq!(bytemuck::bytes_of(&comm.max(x)), bytemuck::bytes_of(&x));
        assert_eq!(
            bytemuck::bytes_of(&comm.exscan_sum(x)),
            bytemuck::bytes_of(&zero)
        );
        assert_eq!(Serial.sum(x), Serial.sum(x));
        assert_eq!(bytemuck::bytes_of(&comm.min(x)), bytemuck::bytes_of(&x));
        assert_eq!(comm.all_gather(x).len(), 1);
        assert_eq!(
            bytemuck::bytes_of(&comm.all_gather(x)[0]),
            bytemuck::bytes_of(&x)
        );
    }

    #[test]
    fn typed_all_to_all_v_counts_and_errors() {
        let comm: &dyn CommDyn = &Serial;
        let (recv, counts) = comm.all_to_all_v(&[3_i64, -4, 5], &[3]).unwrap();
        assert_eq!((recv, counts), (vec![3, -4, 5], vec![3]));
        let (recv, counts) = comm.all_to_all_v::<f32>(&[], &[0]).unwrap();
        assert!(recv.is_empty() && counts == [0]);
        // One count per rank, adding up to the values sent.
        assert!(matches!(
            comm.all_to_all_v(&[1_u32], &[1, 0]),
            Err(GridError::LengthMismatch {
                expected: 1,
                found: 2
            })
        ));
        assert!(matches!(
            comm.all_to_all_v(&[1_u32, 2], &[1]),
            Err(GridError::LengthMismatch {
                expected: 1,
                found: 2
            })
        ));
    }

    #[test]
    fn typed_neighbour_exchange_on_the_empty_neighbourhood() {
        let comm: &dyn CommDyn = &Serial;
        let nb = Neighbourhood::empty();
        let (recv, counts) = comm.neighbour_all_to_all_v::<f64>(&nb, &[], &[]).unwrap();
        assert!(recv.is_empty() && counts.is_empty());
        assert!(matches!(
            comm.neighbour_all_to_all_v(&nb, &[1_i32], &[1]),
            Err(GridError::LengthMismatch {
                expected: 0,
                found: 1
            })
        ));
    }

    #[test]
    fn neighbourhoods_of_serial() {
        let nb = Neighbourhood::new(&Serial, vec![], vec![]).unwrap();
        assert!(nb.sources().is_empty() && nb.destinations().is_empty());
        assert_eq!(nb.comm().map(|c| c.size()), Some(1));
        for (sources, destinations) in [(vec![0], vec![]), (vec![], vec![0]), (vec![], vec![1])] {
            assert!(matches!(
                Neighbourhood::new(&Serial, sources, destinations),
                Err(GridError::IndexOutOfRange { len: 1, .. })
            ));
        }
        let dup = Serial.duplicate();
        assert_eq!((dup.rank(), dup.size()), (0, 1));
    }

    #[test]
    fn agreed_errors_on_one_rank() {
        let ok: Result<u32> = crate::comm::agree(&Serial, Ok(3));
        assert_eq!(ok.unwrap(), 3);
        let err: Result<u32> = crate::comm::agree(&Serial, Err(GridError::DuplicateOriginId(7)));
        assert!(matches!(err, Err(GridError::DuplicateOriginId(7))));
    }

    proptest! {
        #![proptest_config(config())]

        #[test]
        fn prop_comm_ext_on_serial(a in any::<i32>(), b in any::<i64>(), c in any::<u32>(),
                                   d in any::<u64>(), e in any::<f32>(), f in any::<f64>()) {
            comm_ext_identity(a, 0);
            comm_ext_identity(b, 0);
            comm_ext_identity(c, 0);
            comm_ext_identity(d, 0);
            // Bitwise comparison, so NaN payloads must survive too.
            let comm: &dyn CommDyn = &Serial;
            prop_assert_eq!(comm.sum(e).to_bits(), e.to_bits());
            prop_assert_eq!(comm.max(e).to_bits(), e.to_bits());
            prop_assert_eq!(comm.exscan_sum(e).to_bits(), 0.0_f32.to_bits());
            prop_assert_eq!(comm.sum(f).to_bits(), f.to_bits());
            prop_assert_eq!(comm.max(f).to_bits(), f.to_bits());
            prop_assert_eq!(comm.exscan_sum(f).to_bits(), 0.0_f64.to_bits());
        }

        #[test]
        fn prop_byte_level_reduce_every_dtype(
            dtype_idx in 0..6_usize,
            op_idx in 0..3_usize,
            words in proptest::collection::vec(any::<u64>(), 0..17),
        ) {
            let (dtype, op) = (DTYPES[dtype_idx], OPS[op_idx]);
            let size = dtype_size(dtype);
            let bytes: Vec<u8> = bytemuck::cast_slice(&words).to_vec();
            // An odd number of 4-byte elements for 4-byte types, too.
            let len = if size == 4 && !words.is_empty() { bytes.len() - 4 } else { bytes.len() };
            let original = bytes[..len].to_vec();
            let n = len / size;

            let mut buf = original.clone();
            Serial.all_reduce_bytes(&mut buf, dtype, op);
            prop_assert_eq!(&buf, &original);

            let mut buf = original.clone();
            Serial.exscan_bytes(&mut buf, dtype, op);
            prop_assert_eq!(buf, expected_identity(dtype, op, n));
        }

        #[test]
        fn prop_all_to_all_v_copies(
            dtype_idx in 0..6_usize,
            words in proptest::collection::vec(any::<u64>(), 0..17),
        ) {
            let dtype = DTYPES[dtype_idx];
            let send: Vec<u8> = bytemuck::cast_slice(&words).to_vec();
            let count = send.len() / dtype_size(dtype);
            let mut recv = vec![0xA5_u8; send.len()];
            Serial.all_to_all_v_bytes(&send, &[count], &mut recv, &[count], dtype).unwrap();
            prop_assert_eq!(recv, send);
        }
    }
}
