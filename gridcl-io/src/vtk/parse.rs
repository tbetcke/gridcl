//! The VTU parser.

use gridcl_device::Result;

use super::VtuFile;

/// Parses a VTU file: `UnstructuredGrid` with ASCII, inline base64 or appended (raw or base64)
/// data, with or without zlib, and UInt32 or UInt64 headers.
///
/// # Errors
///
/// `Parse { format: "VTU", .. }` for malformed, truncated or unsupported content.
pub fn parse_vtu(bytes: &[u8]) -> Result<VtuFile> {
    let _ = bytes;
    todo!()
}
