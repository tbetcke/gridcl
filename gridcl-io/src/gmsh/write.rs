//! The MSH 4.1 writer.

use std::io::Write;

use gridcl_device::Result;

use super::MshFile;

/// The encoding of a written file. Binary files are little-endian with an 8-byte `size_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// Text, with shortest round-trip floats.
    Ascii,
    /// Binary, little-endian, 8-byte `size_t`.
    Binary,
}

impl MshFile {
    /// Writes the file as MSH 4.1 in `encoding`.
    ///
    /// # Errors
    ///
    /// `Io` if writing fails.
    pub fn write_to(&self, w: &mut dyn Write, encoding: Encoding) -> Result<()> {
        let _ = (w, encoding);
        todo!("M4 T7")
    }
}
