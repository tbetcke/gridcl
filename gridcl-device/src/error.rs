//! `GridError` and `Result`, re-exported by gridcl-parallel and gridcl (ADR-0007).

use core::fmt;

use cubecl::server::ServerError;

/// The result type of gridcl.
pub type Result<T, E = GridError> = core::result::Result<T, E>;

/// Errors of gridcl. Later milestones add variants when they first use them.
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
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        todo!()
    }
}

impl std::error::Error for GridError {}

impl From<ServerError> for GridError {
    fn from(_error: ServerError) -> Self {
        todo!()
    }
}
