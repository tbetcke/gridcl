//! `GridError` and `Result`, re-exported by gridcl-parallel and gridcl (ADR-0009).

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

/// A connectivity between two topological dimensions (Listing T1): `from` → `to`.
///
/// Defined here so that [`GridError::ConnectivityNotCreated`] can carry it; gridcl-topology and
/// gridcl re-export it (ADR-0012).
///
/// # Example
///
/// ```
/// use gridcl_device::{ConnKey, GridError};
///
/// let key = ConnKey { from: 2, to: 0 };
/// assert_eq!(key.to_string(), "2 -> 0");
/// let e = GridError::ConnectivityNotCreated(key);
/// assert_eq!(e.to_string(), "connectivity 2 -> 0 has not been created");
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ConnKey {
    /// The topological dimension of the rows.
    pub from: u8,
    /// The topological dimension of the entries.
    pub to: u8,
}

impl fmt::Display for ConnKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} -> {}", self.from, self.to)
    }
}

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
    /// The node family is not offered on this cell type (GLL exists on intervals,
    /// quadrilaterals and hexahedra only).
    UnsupportedNodeFamily {
        cell: &'static str,
        family: &'static str,
    },
    /// The degree is not supported on this cell type (degrees start at 1).
    UnsupportedDegree { cell: &'static str, degree: u32 },
    /// A cell type used where a different topological dimension is required.
    WrongCellDim {
        found: &'static str,
        expected_tdim: usize,
    },
    /// A mesh of dimensions `found` = (tdim, gdim) where `expected` was required.
    DimensionMismatch {
        expected: (usize, usize),
        found: (usize, usize),
    },
    /// The entities of this dimension have not been created (and those of the cell dimension
    /// never are: cells are in the cell blocks).
    EntitiesNotCreated(usize),
    /// The connectivity has not been created.
    ConnectivityNotCreated(ConnKey),
    /// gridcl does not store this connectivity.
    UnsupportedConnectivity(ConnKey),
    /// Two vertices, nodes or cells share an origin id.
    DuplicateOriginId(i64),
    /// An index is not below the length it indexes.
    IndexOutOfRange { index: usize, len: usize },
    /// A node is a vertex of one cell and a non-vertex node of another; the payload is the
    /// node's input index.
    NodeRoleMismatch(usize),
    /// A second cell block of the same cell type.
    DuplicateCellBlock(&'static str),
    /// The quadrature rule has no point set for this cell type.
    CellNotInRule(&'static str),
    /// A method declared before the milestone that implements it.
    NotYetImplemented(&'static str),
    /// A tag element (Gmsh element tag `tag`, of dimension `dim`) matches no mesh entity.
    UnmatchedTagElement { dim: usize, tag: i64 },
    /// A file could not be read or written: the path and the `std::io::Error` text.
    Io(String),
    /// A malformed file: the format, the byte offset of the error and what was wrong. An error
    /// in the file's content rather than its syntax (or in a model being written) has no byte
    /// offset: `offset` is then 0 and the message says so.
    Parse {
        format: &'static str,
        offset: u64,
        message: String,
    },
    /// The file's element type is not supported (A.7 for Gmsh).
    UnsupportedElementType { format: &'static str, type_id: i32 },
    /// No tag of this name exists in dimension `dim`.
    UnknownTag { name: String, dim: usize },
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
            GridError::UnsupportedNodeFamily { cell, family } => {
                write!(f, "node family {family} is not supported on {cell} cells")
            }
            GridError::UnsupportedDegree { cell, degree } => {
                write!(f, "degree {degree} is not supported on {cell} cells")
            }
            GridError::WrongCellDim {
                found,
                expected_tdim,
            } => write!(
                f,
                "{found} cells where cells of topological dimension {expected_tdim} are required"
            ),
            GridError::DimensionMismatch { expected, found } => write!(
                f,
                "dimension mismatch: expected (tdim, gdim) = {expected:?}, found {found:?}"
            ),
            GridError::EntitiesNotCreated(dim) => {
                write!(f, "the entities of dimension {dim} have not been created")
            }
            GridError::ConnectivityNotCreated(key) => {
                write!(f, "connectivity {key} has not been created")
            }
            GridError::UnsupportedConnectivity(key) => {
                write!(f, "connectivity {key} is not supported")
            }
            GridError::DuplicateOriginId(id) => write!(f, "duplicate origin id {id}"),
            GridError::IndexOutOfRange { index, len } => {
                write!(f, "index {index} is out of range for length {len}")
            }
            GridError::NodeRoleMismatch(node) => write!(
                f,
                "node {node} is a vertex of one cell and a non-vertex node of another"
            ),
            GridError::DuplicateCellBlock(cell) => {
                write!(f, "a second cell block of {cell} cells")
            }
            GridError::CellNotInRule(cell) => {
                write!(f, "the quadrature rule has no point set for {cell} cells")
            }
            GridError::NotYetImplemented(what) => write!(f, "not yet implemented: {what}"),
            GridError::UnmatchedTagElement { dim, tag } => write!(
                f,
                "tag element {tag} of dimension {dim} matches no mesh entity"
            ),
            GridError::Io(message) => write!(f, "I/O error: {message}"),
            GridError::Parse {
                format,
                offset,
                message,
            } => write!(f, "malformed {format} file at byte {offset}: {message}"),
            GridError::UnsupportedElementType { format, type_id } => {
                write!(f, "{format} element type {type_id} is not supported")
            }
            GridError::UnknownTag { name, dim } => {
                write!(f, "no tag named {name:?} in dimension {dim}")
            }
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

#[cfg(test)]
mod tests {
    use super::GridError;

    /// The `Display` lines of the M4 variants (ADR-0022).
    #[test]
    fn m4_variants_display() {
        let cases = [
            (
                GridError::UnmatchedTagElement { dim: 1, tag: 42 },
                "tag element 42 of dimension 1 matches no mesh entity",
            ),
            (
                GridError::Io("mesh.msh: No such file or directory".into()),
                "I/O error: mesh.msh: No such file or directory",
            ),
            (
                GridError::Parse {
                    format: "Gmsh",
                    offset: 17,
                    message: "expected an integer".into(),
                },
                "malformed Gmsh file at byte 17: expected an integer",
            ),
            (
                GridError::UnsupportedElementType {
                    format: "Gmsh",
                    type_id: 125,
                },
                "Gmsh element type 125 is not supported",
            ),
            (
                GridError::UnknownTag {
                    name: "scalp".into(),
                    dim: 2,
                },
                "no tag named \"scalp\" in dimension 2",
            ),
        ];
        for (e, want) in cases {
            assert_eq!(e.to_string(), want);
        }
    }
}
