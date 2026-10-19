//! `GridError` and `Result`, re-exported by gridcl-parallel and gridcl (ADR-0007).

use core::fmt;

use cubecl::server::ServerError;

/// The result type of gridcl.
///
/// # Example
///
/// ```
/// use gridcl_device::{GridError, Result};
///
/// fn half(n: usize) -> Result<usize> {
///     if n % 2 == 0 { Ok(n / 2) } else { Err(GridError::LengthMismatch { expected: n + 1, found: n }) }
/// }
/// assert_eq!(half(4).unwrap(), 2);
/// assert!(half(3).is_err());
/// ```
pub type Result<T, E = GridError> = core::result::Result<T, E>;

/// Errors of gridcl. Later milestones add variants when they first use them.
///
/// # Example
///
/// ```
/// use gridcl_device::GridError;
///
/// let e = GridError::AllocationTooLarge { bytes: 2048, max: 1024 };
/// assert_eq!(e.to_string(), "an allocation of 2048 bytes exceeds the device's max_page_size of 1024 bytes");
/// ```
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum GridError {
    /// A size or index does not fit the type that must hold it.
    TooLarge,
    /// One allocation of `bytes` bytes exceeds the device's `max_page_size`.
    AllocationTooLarge { bytes: usize, max: u64 },
    /// The device does not support the scalar type.
    UnsupportedPrecision {
        scalar: &'static str,
        device: &'static str,
    },
    /// Two lengths that must agree differ.
    LengthMismatch { expected: usize, found: usize },
    /// The compute server reported an error.
    Server(ServerError),
}

impl fmt::Display for GridError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GridError::TooLarge => f.write_str("a size or index does not fit its type"),
            GridError::AllocationTooLarge { bytes, max } => write!(
                f,
                "an allocation of {bytes} bytes exceeds the device's max_page_size of {max} bytes"
            ),
            GridError::UnsupportedPrecision { scalar, device } => {
                write!(f, "{device} does not support {scalar}")
            }
            GridError::LengthMismatch { expected, found } => {
                write!(f, "length mismatch: expected {expected}, found {found}")
            }
            GridError::Server(error) => write!(f, "compute server error: {error}"),
        }
    }
}

impl std::error::Error for GridError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            GridError::Server(error) => Some(error),
            _ => None,
        }
    }
}

impl From<ServerError> for GridError {
    fn from(error: ServerError) -> Self {
        GridError::Server(error)
    }
}
