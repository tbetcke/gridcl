//! The VTU and PVTU writers: XML `version="2.2"`, little-endian, UInt64 headers.

use std::io::Write;

use gridcl_device::Result;

use super::VtuFile;

/// Where the binary data of a written file goes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binary {
    /// Raw bytes in one `AppendedData` section after the XML.
    Appended,
    /// Base64 inside each `DataArray`.
    Base64,
}

impl VtuFile {
    /// Writes the file: `type="UnstructuredGrid" version="2.2" byte_order="LittleEndian"
    /// header_type="UInt64"`, with every array as `binary` data, compressed by zlib in VTK's
    /// block layout if `zlib`.
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if the arrays do not fit together; `Io` from `w`.
    pub fn write_to(&self, w: &mut dyn Write, binary: Binary, zlib: bool) -> Result<()> {
        let _ = (w, binary, zlib);
        todo!()
    }
}

/// Writes the `PUnstructuredGrid` header naming `pieces`, with the point, point data and cell
/// data arrays declared as in `like`.
///
/// # Errors
///
/// `Io` from `w`.
pub fn write_pvtu(w: &mut dyn Write, pieces: &[String], like: &VtuFile) -> Result<()> {
    let _ = (w, pieces, like);
    todo!()
}
