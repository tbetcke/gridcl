//! The MSH 4.1 parser: ASCII and binary, either byte order, 4- or 8-byte `size_t`.

use std::path::Path;

use gridcl_device::Result;

use super::MshFile;

/// Parses an MSH 4.1 file.
///
/// Header counts are checked with checked arithmetic, and no allocation is sized from a count
/// before the data it describes is there. Unknown sections are skipped.
///
/// # Errors
///
/// `Parse` for a malformed or truncated file, a version other than 4.1, or a node or element
/// tag ≤ 0; the offset is the byte where the error was found.
pub fn parse(bytes: &[u8]) -> Result<MshFile> {
    let _ = bytes;
    todo!("M4 T4")
}

/// Reads and parses the file at `path`.
///
/// # Errors
///
/// `Io` if the file cannot be read; errors of [`parse`].
pub fn read_file(path: &Path) -> Result<MshFile> {
    let _ = path;
    todo!("M4 T4")
}
