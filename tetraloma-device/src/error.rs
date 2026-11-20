//! `GridError` and `Result`, re-exported by tetraloma-parallel and tetraloma (ADR-0009).

use core::fmt;

use cubecl::server::ServerError;

/// The result type of tetraloma: `Ok(T)` or a [`GridError`].
///
/// Every fallible function of tetraloma returns it, so `?` passes errors through, and a
/// [`GridError`] converts into `Box<dyn std::error::Error>`. The tutorial [Building a
/// mesh](https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#errors) shows the errors
/// of mesh creation.
///
/// # Example
///
/// ```
/// use tetraloma_device::{GridError, Result};
///
/// fn half(n: usize) -> Result<usize> {
///     if n % 2 == 0 { Ok(n / 2) } else { Err(GridError::LengthMismatch { expected: n + 1, found: n }) }
/// }
/// assert_eq!(half(4).unwrap(), 2);
/// assert!(half(3).is_err());
/// ```
pub type Result<T, E = GridError> = core::result::Result<T, E>;

// Listing T1; defined here so that `GridError::ConnectivityNotCreated` can carry it, and
// re-exported by tetraloma-topology and tetraloma (ADR-0012).
/// The name of a connectivity, from the entities of dimension `from` to those of dimension
/// `to`. In a mesh of triangles, `ConnKey { from: 2, to: 1 }` gives the edges of each cell, and
/// `ConnKey { from: 1, to: 2 }` the cells that contain each edge.
///
/// A connectivity is downward if `from > to` and upward if `from < to`. [Meshes and
/// topology](https://tetracl.github.io/tetraloma/concepts/meshes.html#connectivity) explains
/// both kinds. It prints as `from -> to`.
///
/// # Example
///
/// ```
/// use tetraloma_device::{ConnKey, GridError};
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

/// The errors of tetraloma.
///
/// Each variant names one kind of failure, and its `Display` text says what went wrong. The
/// enum is `#[non_exhaustive]`, so a `match` on it needs a `_` arm. Collective operations agree
/// on their errors: the rank that failed returns its own error, and the other ranks return
/// [`RemoteFailure`](GridError::RemoteFailure). The tutorial [Building a
/// mesh](https://tetracl.github.io/tetraloma/tutorials/build-mesh.html#errors) shows the most
/// common errors of mesh creation.
///
/// # Example
///
/// ```
/// use tetraloma_device::GridError;
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
    AllocationTooLarge {
        /// The size of the requested allocation in bytes.
        bytes: usize,
        /// The device's `max_page_size` in bytes.
        max: u64,
    },
    /// The device does not support the scalar type.
    UnsupportedPrecision {
        /// The name of the scalar type, as `"f64"`.
        scalar: &'static str,
        /// The name of the device.
        device: &'static str,
    },
    /// Two lengths that must agree differ.
    LengthMismatch {
        /// The required length.
        expected: usize,
        /// The length given.
        found: usize,
    },
    /// The compute server reported an error.
    Server(ServerError),
    /// The node family is not offered on this cell type (GLL exists on intervals,
    /// quadrilaterals and hexahedra only).
    UnsupportedNodeFamily {
        /// The name of the cell type.
        cell: &'static str,
        /// The name of the node family.
        family: &'static str,
    },
    /// The degree is not supported on this cell type (degrees start at 1).
    UnsupportedDegree {
        /// The name of the cell type.
        cell: &'static str,
        /// The degree asked for.
        degree: u32,
    },
    /// A cell type used where a different topological dimension is required.
    WrongCellDim {
        /// The name of the cell type given.
        found: &'static str,
        /// The topological dimension required.
        expected_tdim: usize,
    },
    /// A mesh of dimensions `found` = (tdim, gdim) where `expected` was required.
    DimensionMismatch {
        /// The required (tdim, gdim).
        expected: (usize, usize),
        /// The mesh's (tdim, gdim).
        found: (usize, usize),
    },
    /// The entities of this dimension have not been created (and those of the cell dimension
    /// never are: cells are in the cell blocks).
    EntitiesNotCreated(usize),
    /// The connectivity has not been created.
    ConnectivityNotCreated(ConnKey),
    /// tetraloma does not store this connectivity.
    UnsupportedConnectivity(ConnKey),
    /// Two vertices, nodes or cells share an origin id.
    DuplicateOriginId(i64),
    /// An index is not below the length it indexes.
    IndexOutOfRange {
        /// The offending index.
        index: usize,
        /// The length it must be below.
        len: usize,
    },
    /// A node is a vertex of one cell and a non-vertex node of another; the payload is the
    /// node's input index.
    NodeRoleMismatch(usize),
    /// A second cell block of the same cell type.
    DuplicateCellBlock(&'static str),
    /// The quadrature rule has no point set for this cell type.
    CellNotInRule(&'static str),
    /// A method that is declared but not implemented yet; the payload names it.
    NotYetImplemented(&'static str),
    /// A tag element (Gmsh element tag `tag`, of dimension `dim`) matches no mesh entity.
    UnmatchedTagElement {
        /// The topological dimension of the element.
        dim: usize,
        /// The Gmsh element tag.
        tag: i64,
    },
    /// A file could not be read or written: the path and the `std::io::Error` text.
    Io(String),
    /// A malformed file: the format, the byte offset of the error and what was wrong. An error
    /// in the file's content rather than its syntax (or in a model being written) has no byte
    /// offset: `offset` is then 0 and the message says so.
    Parse {
        /// The name of the file format.
        format: &'static str,
        /// The byte offset of the error in the file, or 0 for an error in its content.
        offset: u64,
        /// What was wrong.
        message: String,
    },
    /// The file's element type is not supported; [File
    /// formats](https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh) lists the
    /// supported Gmsh element types.
    UnsupportedElementType {
        /// The name of the file format.
        format: &'static str,
        /// The format's id of the element type.
        type_id: i32,
    },
    /// No tag of this name exists in dimension `dim`.
    UnknownTag {
        /// The tag name asked for.
        name: String,
        /// The topological dimension searched.
        dim: usize,
    },
    /// An MPI count or displacement of `count` elements exceeds `i32::MAX`, the limit of MPI's
    /// `int` counts.
    MessageTooLarge {
        /// The count or displacement, in elements.
        count: usize,
    },
    /// A collective operation failed on another rank; `rank` is the lowest failing rank, which
    /// returns its own error.
    RemoteFailure {
        /// The lowest rank on which the operation failed.
        rank: usize,
    },
    // ADR-0029, item 6.
    /// A partitioner library returned the status `code`; tetraloma's own conditions use negative
    /// codes: −1 for ParMETIS with an empty rank, −2 for a self-loop, −3 for an asymmetric
    /// graph. With `name` `"metis"`, a code after validation is METIS's own status
    /// (`rstatus_et`: −2 input error, −3 out of memory, −4 other error).
    PartitionerFailed {
        /// The partitioner's name, as `"metis"`.
        name: &'static str,
        /// The library's status, or tetraloma's negative code.
        code: i32,
    },
    // ADR-0029, item 6.
    /// A distributed partitioner was asked for more than one part on a communicator without an
    /// MPI handle (`CommDyn::raw` is `None`), on any number of ranks.
    NeedsMpiCommunicator(&'static str),
    // ADR-0039.
    /// A facet has more than two cells, counting owned cells on all ranks: the mesh is not a
    /// manifold (`orient`, `boundary`, `facet_submesh`).
    NonManifold,
    // ADR-0039.
    /// A tagged facet of `facet_submesh` has two cells: facet submeshes take boundary facets
    /// only.
    InteriorFacet,
    // ADR-0039.
    /// `OrientationRule::Outward` on a component with boundary facets, which encloses no
    /// volume: orient it with `AwayFrom` or `Towards`.
    NeedsReferencePoint,
    // ADR-0039.
    /// The manifold cannot be oriented consistently, as a Möbius strip.
    NonOrientable,
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
            GridError::MessageTooLarge { count } => write!(
                f,
                "a message count or displacement of {count} elements exceeds i32::MAX"
            ),
            GridError::RemoteFailure { rank } => {
                write!(f, "the operation failed on rank {rank}")
            }
            GridError::PartitionerFailed { name, code } => {
                write!(f, "the {name} partitioner failed with status {code}")
            }
            GridError::NeedsMpiCommunicator(name) => write!(
                f,
                "the {name} partitioner needs an MPI communicator for more than one part"
            ),
            GridError::NonManifold => f.write_str("a facet has more than two cells"),
            GridError::InteriorFacet => {
                f.write_str("a tagged facet is an interior facet, not a boundary facet")
            }
            GridError::NeedsReferencePoint => {
                f.write_str("outward orientation of an open component needs a reference point")
            }
            GridError::NonOrientable => f.write_str("the manifold is not orientable"),
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

    /// The `Display` lines of the M9 variants (ADR-0039).
    #[test]
    fn m9_variants_display() {
        let cases = [
            (GridError::NonManifold, "a facet has more than two cells"),
            (
                GridError::InteriorFacet,
                "a tagged facet is an interior facet, not a boundary facet",
            ),
            (
                GridError::NeedsReferencePoint,
                "outward orientation of an open component needs a reference point",
            ),
            (GridError::NonOrientable, "the manifold is not orientable"),
        ];
        for (e, want) in cases {
            assert_eq!(e.to_string(), want);
        }
    }

    /// The `Display` lines of the M7 variants (ADR-0029).
    #[test]
    fn m7_variants_display() {
        assert_eq!(
            GridError::PartitionerFailed {
                name: "parmetis",
                code: -1
            }
            .to_string(),
            "the parmetis partitioner failed with status -1"
        );
        assert_eq!(
            GridError::NeedsMpiCommunicator("ptscotch").to_string(),
            "the ptscotch partitioner needs an MPI communicator for more than one part"
        );
    }

    /// The `Display` lines of the M6 variants (ADR-0026).
    #[test]
    fn m6_variants_display() {
        assert_eq!(
            GridError::MessageTooLarge { count: 1 << 31 }.to_string(),
            "a message count or displacement of 2147483648 elements exceeds i32::MAX"
        );
        assert_eq!(
            GridError::RemoteFailure { rank: 3 }.to_string(),
            "the operation failed on rank 3"
        );
    }

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
