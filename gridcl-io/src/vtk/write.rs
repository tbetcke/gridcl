//! The VTU and PVTU writers: XML `version="2.2"`, little-endian, UInt64 headers.
//!
//! Each data array is stored as VTK's reader expects it (`vtkXMLDataParser`), as written by
//! `vtkXMLWriter` with `header_type="UInt64"`:
//!
//! - **Uncompressed:** one UInt64 holding the data's byte count, then the little-endian
//!   data. Inline, header and data form one base64 stream.
//! - **zlib:** the data is cut into blocks of [`BLOCK_SIZE`] bytes (the last may be shorter),
//!   each compressed to a separate zlib stream. The header is the UInt64s
//!   `[number of blocks, BLOCK_SIZE, byte count of the last block if partial else 0,
//!   compressed size of each block…]`; the compressed streams follow back to back. An empty
//!   array has the header `[0, BLOCK_SIZE, 0]` and no streams. Inline, the header and the
//!   streams are two base64 streams, each padded, one after the other.
//! - **Appended raw:** the blocks of all arrays follow the `_` of `<AppendedData
//!   encoding="raw">` in document order, and each `DataArray`'s `offset` is the byte offset of
//!   its header from just after the `_`.

use std::io::Write;

use flate2::Compression;
use flate2::write::ZlibEncoder;
use gridcl_device::{GridError, Result};

use super::{DataArray, DataValues, VtuFile, base64};

/// The uncompressed size of a zlib block, VTK's default.
const BLOCK_SIZE: usize = 32768;

/// The size of a header word (`header_type="UInt64"`).
const WORD: usize = 8;

/// Where the binary data of a written file goes.
///
/// # Example
///
/// ```
/// use gridcl_io::vtk::{Binary, DataValues, VtuFile};
///
/// let file = VtuFile {
///     points: DataValues::F64(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
///     connectivity: vec![0, 1],
///     offsets: vec![2],
///     types: vec![3],
///     point_data: vec![],
///     cell_data: vec![],
/// };
/// for (binary, format) in [(Binary::Appended, "format=\"appended\" offset=\"0\""), (Binary::Base64, "format=\"binary\"")] {
///     let mut bytes = Vec::new();
///     file.write_to(&mut bytes, binary, false)?;
///     let text = String::from_utf8_lossy(&bytes);
///     assert!(text.contains(&format!("Name=\"Points\" NumberOfComponents=\"3\" {format}")));
///     assert_eq!(text.contains("<AppendedData encoding=\"raw\">"), binary == Binary::Appended);
/// }
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Binary {
    /// Raw bytes in one `AppendedData` section after the XML.
    Appended,
    /// Base64 inside each `DataArray`.
    Base64,
}

/// A value stored little-endian.
trait LittleEndian: Copy {
    fn put(self, out: &mut Vec<u8>);
}

macro_rules! little_endian {
    ($($t:ty),*) => {$(
        impl LittleEndian for $t {
            fn put(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
        }
    )*};
}

little_endian!(f32, f64, i32, u32, i64, u8, u64);

/// `WORD` zero bytes (room for the uncompressed header), then `xs` little-endian.
fn with_header_room<T: LittleEndian>(xs: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(WORD + std::mem::size_of_val(xs));
    out.resize(WORD, 0);
    for &x in xs {
        x.put(&mut out);
    }
    out
}

/// The VTK type name of `v`.
fn type_name(v: &DataValues) -> &'static str {
    match v {
        DataValues::F32(_) => "Float32",
        DataValues::F64(_) => "Float64",
        DataValues::I32(_) => "Int32",
        DataValues::U32(_) => "UInt32",
        DataValues::I64(_) => "Int64",
        DataValues::U8(_) => "UInt8",
    }
}

/// The number of values in `v`.
fn value_count(v: &DataValues) -> usize {
    match v {
        DataValues::F32(x) => x.len(),
        DataValues::F64(x) => x.len(),
        DataValues::I32(x) => x.len(),
        DataValues::U32(x) => x.len(),
        DataValues::I64(x) => x.len(),
        DataValues::U8(x) => x.len(),
    }
}

/// [`with_header_room`] of the values of `v`.
fn values_with_header_room(v: &DataValues) -> Vec<u8> {
    match v {
        DataValues::F32(x) => with_header_room(x),
        DataValues::F64(x) => with_header_room(x),
        DataValues::I32(x) => with_header_room(x),
        DataValues::U32(x) => with_header_room(x),
        DataValues::I64(x) => with_header_room(x),
        DataValues::U8(x) => with_header_room(x),
    }
}

/// `s` as XML attribute text: the five predefined entities, and character references for tab,
/// line feed and carriage return, which attribute-value normalisation would otherwise turn
/// into spaces.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            '\t' => out.push_str("&#9;"),
            '\n' => out.push_str("&#10;"),
            '\r' => out.push_str("&#13;"),
            c => out.push(c),
        }
    }
    out
}

fn io_error(e: std::io::Error) -> GridError {
    GridError::Io(format!("writing a VTU file: {e}"))
}

/// One array as stored: `header`, then `data` (see the module documentation).
struct Stored {
    header: Vec<u8>,
    data: Vec<u8>,
}

/// Stores `bytes`, whose first `WORD` bytes are room for the uncompressed header, followed by
/// the data. Uncompressed, the header is written into that room and `data` stays empty.
fn store(mut bytes: Vec<u8>, zlib: bool) -> Result<Stored> {
    let raw = &bytes[WORD..];
    if !zlib {
        let n = raw.len() as u64;
        bytes[..WORD].copy_from_slice(&n.to_le_bytes());
        return Ok(Stored {
            header: bytes,
            data: Vec::new(),
        });
    }
    let blocks = raw.len().div_ceil(BLOCK_SIZE);
    let mut header = Vec::with_capacity(WORD * (3 + blocks));
    (blocks as u64).put(&mut header);
    (BLOCK_SIZE as u64).put(&mut header);
    ((raw.len() % BLOCK_SIZE) as u64).put(&mut header);
    let mut data = Vec::new();
    for block in raw.chunks(BLOCK_SIZE) {
        let start = data.len();
        let mut z = ZlibEncoder::new(data, Compression::default());
        z.write_all(block).map_err(io_error)?;
        data = z.finish().map_err(io_error)?;
        ((data.len() - start) as u64).put(&mut header);
    }
    Ok(Stored { header, data })
}

/// The file being written: the XML, and the appended data if `binary` is `Appended`.
struct Out {
    xml: String,
    appended: Vec<u8>,
    binary: Binary,
    zlib: bool,
}

impl Out {
    fn line(&mut self, indent: usize, s: &str) {
        for _ in 0..indent {
            self.xml.push_str("  ");
        }
        self.xml.push_str(s);
        self.xml.push('\n');
    }

    /// A `DataArray` element at `indent` whose values (after `WORD` bytes of room) are
    /// `bytes`.
    fn array(
        &mut self,
        indent: usize,
        type_name: &str,
        name: &str,
        components: usize,
        bytes: Vec<u8>,
    ) -> Result<()> {
        let stored = store(bytes, self.zlib)?;
        let open = format!(
            "<DataArray type=\"{type_name}\" Name=\"{}\" NumberOfComponents=\"{components}\" format=",
            escape(name)
        );
        match self.binary {
            Binary::Appended => {
                self.line(
                    indent,
                    &format!("{open}\"appended\" offset=\"{}\"/>", self.appended.len()),
                );
                self.appended.extend_from_slice(&stored.header);
                self.appended.extend_from_slice(&stored.data);
            }
            Binary::Base64 => {
                self.line(indent, &format!("{open}\"binary\">"));
                let mut text = Vec::new();
                base64::encode_into(&stored.header, &mut text);
                if self.zlib {
                    base64::encode_into(&stored.data, &mut text);
                }
                let text = String::from_utf8(text).expect("base64 is ASCII");
                self.line(indent + 1, &text);
                self.line(indent, "</DataArray>");
            }
        }
        Ok(())
    }

    fn data_arrays(&mut self, section: &str, arrays: &[DataArray]) -> Result<()> {
        self.line(3, &format!("<{section}>"));
        for a in arrays {
            let bytes = values_with_header_room(&a.values);
            self.array(4, type_name(&a.values), &a.name, a.components, bytes)?;
        }
        self.line(3, &format!("</{section}>"));
        Ok(())
    }
}

/// `LengthMismatch` unless every array of `arrays` holds `entries · components` values with
/// `components ≥ 1`.
fn check_data(arrays: &[DataArray], entries: usize) -> Result<()> {
    for a in arrays {
        let found = value_count(&a.values);
        if a.components == 0 {
            return Err(GridError::LengthMismatch {
                expected: 1,
                found: 0,
            });
        }
        let expected = entries.saturating_mul(a.components);
        if found != expected {
            return Err(GridError::LengthMismatch { expected, found });
        }
    }
    Ok(())
}

impl VtuFile {
    /// `LengthMismatch` unless the arrays fit together (see [`VtuFile::write_to`]); returns
    /// the number of points.
    fn check(&self) -> Result<usize> {
        let found = value_count(&self.points);
        if !found.is_multiple_of(3) {
            return Err(GridError::LengthMismatch {
                expected: found - found % 3,
                found,
            });
        }
        if self.offsets.len() != self.types.len() {
            return Err(GridError::LengthMismatch {
                expected: self.types.len(),
                found: self.offsets.len(),
            });
        }
        let as_len = |o: i64| usize::try_from(o).unwrap_or(0);
        let mut last = 0i64;
        for &o in &self.offsets {
            if o < last {
                return Err(GridError::LengthMismatch {
                    expected: as_len(last),
                    found: as_len(o),
                });
            }
            last = o;
        }
        if usize::try_from(last).ok() != Some(self.connectivity.len()) {
            return Err(GridError::LengthMismatch {
                expected: self.connectivity.len(),
                found: as_len(last),
            });
        }
        let points = found / 3;
        check_data(&self.point_data, points)?;
        check_data(&self.cell_data, self.types.len())?;
        Ok(points)
    }

    /// Writes the file: `type="UnstructuredGrid" version="2.2" byte_order="LittleEndian"
    /// header_type="UInt64"`, with every array as `binary` data, compressed by zlib in VTK's
    /// block layout if `zlib`.
    ///
    /// The one `Piece` holds `PointData`, `CellData`, `Points` (`Name="Points"`) and `Cells`
    /// (`connectivity` and `offsets` as Int64, `types` as UInt8). Every `DataArray` names its
    /// `type`, `Name`, `NumberOfComponents` and `format`. Names are escaped as XML. zlib uses
    /// blocks of 32768 bytes and a fixed compression level, so the same file always gives
    /// the same bytes.
    ///
    /// # Errors
    ///
    /// Checked before anything is written:
    ///
    /// - `LengthMismatch` if the arrays do not fit together:
    ///   - the length of `points` is not a multiple of 3 (`expected` is the largest multiple
    ///     not above it);
    ///   - `offsets` and `types` differ in length (`expected` is that of `types`);
    ///   - an offset is negative or below the one before it (`expected` is the one before,
    ///     or 0; a negative offset is reported as 0);
    ///   - the last offset (0 without cells) is not the length of `connectivity`;
    ///   - a point or cell data array has `components` 0 (`expected: 1, found: 0`), or does
    ///     not hold `components` values per point or cell.
    ///
    /// `Io` from `w`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::vtk::{Binary, DataArray, DataValues, VtuFile};
    ///
    /// // One triangle with a scalar on each point.
    /// let file = VtuFile {
    ///     points: DataValues::F32(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    ///     connectivity: vec![0, 1, 2],
    ///     offsets: vec![3],
    ///     types: vec![5],
    ///     point_data: vec![DataArray {
    ///         name: "u".into(),
    ///         components: 1,
    ///         values: DataValues::F64(vec![0.0, 1.0, 2.0]),
    ///     }],
    ///     cell_data: vec![],
    /// };
    /// let mut bytes = Vec::new();
    /// file.write_to(&mut bytes, Binary::Base64, true)?;
    /// let text = String::from_utf8(bytes).unwrap();
    /// assert!(text.starts_with(
    ///     "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"2.2\" \
    ///      byte_order=\"LittleEndian\" header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\">\n"
    /// ));
    /// assert!(text.contains("<Piece NumberOfPoints=\"3\" NumberOfCells=\"1\">"));
    ///
    /// // Arrays that do not fit together are rejected before anything is written.
    /// let mut bad = file.clone();
    /// bad.offsets = vec![2];
    /// let mut bytes = Vec::new();
    /// assert!(bad.write_to(&mut bytes, Binary::Appended, false).is_err());
    /// assert!(bytes.is_empty());
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn write_to(&self, w: &mut dyn Write, binary: Binary, zlib: bool) -> Result<()> {
        let points = self.check()?;
        let mut o = Out {
            xml: String::new(),
            appended: Vec::new(),
            binary,
            zlib,
        };
        o.line(0, "<?xml version=\"1.0\"?>");
        let compressor = if zlib {
            " compressor=\"vtkZLibDataCompressor\""
        } else {
            ""
        };
        o.line(
            0,
            &format!(
                "<VTKFile type=\"UnstructuredGrid\" version=\"2.2\" byte_order=\"LittleEndian\" \
                 header_type=\"UInt64\"{compressor}>"
            ),
        );
        o.line(1, "<UnstructuredGrid>");
        o.line(
            2,
            &format!(
                "<Piece NumberOfPoints=\"{points}\" NumberOfCells=\"{}\">",
                self.types.len()
            ),
        );
        o.data_arrays("PointData", &self.point_data)?;
        o.data_arrays("CellData", &self.cell_data)?;
        o.line(3, "<Points>");
        o.array(
            4,
            type_name(&self.points),
            "Points",
            3,
            values_with_header_room(&self.points),
        )?;
        o.line(3, "</Points>");
        o.line(3, "<Cells>");
        o.array(
            4,
            "Int64",
            "connectivity",
            1,
            with_header_room(&self.connectivity),
        )?;
        o.array(4, "Int64", "offsets", 1, with_header_room(&self.offsets))?;
        o.array(4, "UInt8", "types", 1, with_header_room(&self.types))?;
        o.line(3, "</Cells>");
        o.line(2, "</Piece>");
        o.line(1, "</UnstructuredGrid>");
        let mut bytes = o.xml.into_bytes();
        if binary == Binary::Appended {
            bytes.extend_from_slice(b"  <AppendedData encoding=\"raw\">\n   _");
            bytes.extend_from_slice(&o.appended);
            bytes.extend_from_slice(b"\n  </AppendedData>\n");
        }
        bytes.extend_from_slice(b"</VTKFile>\n");
        w.write_all(&bytes).map_err(io_error)
    }
}

/// Writes the `PUnstructuredGrid` header naming `pieces`, with the point, point data and cell
/// data arrays declared as in `like`.
///
/// The file is `type="PUnstructuredGrid" version="2.2" byte_order="LittleEndian"
/// header_type="UInt64"` with `GhostLevel="0"`: `PPointData` and `PCellData` with a
/// `PDataArray` (`type`, `Name`, `NumberOfComponents`) for each array of `like`, `PPoints`,
/// then one `<Piece Source="…"/>` per entry of `pieces`, a path relative to the PVTU file.
/// Names and paths are escaped as XML.
///
/// # Errors
///
/// `Io` from `w`.
///
/// # Example
///
/// ```
/// use gridcl_io::vtk::{DataArray, DataValues, VtuFile, write_pvtu};
///
/// let like = VtuFile {
///     points: DataValues::F64(vec![]),
///     connectivity: vec![],
///     offsets: vec![],
///     types: vec![],
///     point_data: vec![],
///     cell_data: vec![DataArray { name: "rank".into(), components: 1, values: DataValues::I32(vec![]) }],
/// };
/// let mut bytes = Vec::new();
/// write_pvtu(&mut bytes, &["out_0.vtu".to_string()], &like)?;
/// let text = String::from_utf8(bytes).unwrap();
/// assert!(text.contains("<PDataArray type=\"Int32\" Name=\"rank\" NumberOfComponents=\"1\"/>"));
/// assert!(text.contains("<PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>"));
/// assert!(text.contains("<Piece Source=\"out_0.vtu\"/>"));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn write_pvtu(w: &mut dyn Write, pieces: &[String], like: &VtuFile) -> Result<()> {
    let mut o = Out {
        xml: String::new(),
        appended: Vec::new(),
        binary: Binary::Appended,
        zlib: false,
    };
    let declare = |o: &mut Out, a: &DataArray| {
        o.line(
            3,
            &format!(
                "<PDataArray type=\"{}\" Name=\"{}\" NumberOfComponents=\"{}\"/>",
                type_name(&a.values),
                escape(&a.name),
                a.components
            ),
        );
    };
    o.line(0, "<?xml version=\"1.0\"?>");
    o.line(
        0,
        "<VTKFile type=\"PUnstructuredGrid\" version=\"2.2\" byte_order=\"LittleEndian\" \
         header_type=\"UInt64\">",
    );
    o.line(1, "<PUnstructuredGrid GhostLevel=\"0\">");
    o.line(2, "<PPointData>");
    for a in &like.point_data {
        declare(&mut o, a);
    }
    o.line(2, "</PPointData>");
    o.line(2, "<PCellData>");
    for a in &like.cell_data {
        declare(&mut o, a);
    }
    o.line(2, "</PCellData>");
    o.line(2, "<PPoints>");
    o.line(
        3,
        &format!(
            "<PDataArray type=\"{}\" Name=\"Points\" NumberOfComponents=\"3\"/>",
            type_name(&like.points)
        ),
    );
    o.line(2, "</PPoints>");
    for p in pieces {
        o.line(2, &format!("<Piece Source=\"{}\"/>", escape(p)));
    }
    o.line(1, "</PUnstructuredGrid>");
    o.line(0, "</VTKFile>");
    w.write_all(o.xml.as_bytes()).map_err(io_error)
}

#[cfg(test)]
mod tests {
    use std::io::Read;

    use flate2::read::ZlibDecoder;

    use super::*;

    /// Two triangles of a unit square, with point and cell data.
    fn square() -> VtuFile {
        VtuFile {
            points: DataValues::F64(vec![
                0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0,
            ]),
            connectivity: vec![0, 1, 2, 1, 3, 2],
            offsets: vec![3, 6],
            types: vec![5, 5],
            point_data: vec![DataArray {
                name: "u".into(),
                components: 1,
                values: DataValues::F32(vec![0.0, 1.0, 2.0, 3.0]),
            }],
            cell_data: vec![DataArray {
                name: "id".into(),
                components: 2,
                values: DataValues::U32(vec![1, 2, 3, 4]),
            }],
        }
    }

    fn write(file: &VtuFile, binary: Binary, zlib: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        file.write_to(&mut bytes, binary, zlib).unwrap();
        bytes
    }

    fn words(bytes: &[u8]) -> Vec<u64> {
        bytes
            .chunks(8)
            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// The appended data after the `_`.
    fn appended(bytes: &[u8]) -> &[u8] {
        let marker = b"<AppendedData encoding=\"raw\">\n   _";
        let at = bytes
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap();
        &bytes[at + marker.len()..]
    }

    #[test]
    fn write_xml_header_golden() {
        for (binary, zlib) in [
            (Binary::Appended, false),
            (Binary::Appended, true),
            (Binary::Base64, false),
            (Binary::Base64, true),
        ] {
            let compressor = if zlib {
                " compressor=\"vtkZLibDataCompressor\""
            } else {
                ""
            };
            let format = match binary {
                Binary::Appended => "\"appended\" offset=\"0\"/>",
                Binary::Base64 => "\"binary\">",
            };
            let expected = format!(
                "<?xml version=\"1.0\"?>\n\
                 <VTKFile type=\"UnstructuredGrid\" version=\"2.2\" byte_order=\"LittleEndian\" \
                 header_type=\"UInt64\"{compressor}>\n\
                 \x20 <UnstructuredGrid>\n\
                 \x20   <Piece NumberOfPoints=\"4\" NumberOfCells=\"2\">\n\
                 \x20     <PointData>\n\
                 \x20       <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"1\" \
                 format={format}\n"
            );
            let bytes = write(&square(), binary, zlib);
            assert!(bytes.starts_with(expected.as_bytes()), "{binary:?} {zlib}");
        }
    }

    #[test]
    fn write_appended_raw_golden() {
        let bytes = write(&square(), Binary::Appended, false);
        let text = String::from_utf8_lossy(&bytes);
        let xml = &text[..text.find("  <AppendedData").unwrap()];
        assert_eq!(
            xml,
            "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"UnstructuredGrid\" version=\"2.2\" byte_order=\"LittleEndian\" \
             header_type=\"UInt64\">\n\
             \x20 <UnstructuredGrid>\n\
             \x20   <Piece NumberOfPoints=\"4\" NumberOfCells=\"2\">\n\
             \x20     <PointData>\n\
             \x20       <DataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"1\" format=\"appended\" offset=\"0\"/>\n\
             \x20     </PointData>\n\
             \x20     <CellData>\n\
             \x20       <DataArray type=\"UInt32\" Name=\"id\" NumberOfComponents=\"2\" format=\"appended\" offset=\"24\"/>\n\
             \x20     </CellData>\n\
             \x20     <Points>\n\
             \x20       <DataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\" format=\"appended\" offset=\"48\"/>\n\
             \x20     </Points>\n\
             \x20     <Cells>\n\
             \x20       <DataArray type=\"Int64\" Name=\"connectivity\" NumberOfComponents=\"1\" format=\"appended\" offset=\"152\"/>\n\
             \x20       <DataArray type=\"Int64\" Name=\"offsets\" NumberOfComponents=\"1\" format=\"appended\" offset=\"208\"/>\n\
             \x20       <DataArray type=\"UInt8\" Name=\"types\" NumberOfComponents=\"1\" format=\"appended\" offset=\"232\"/>\n\
             \x20     </Cells>\n\
             \x20   </Piece>\n\
             \x20 </UnstructuredGrid>\n"
        );
        // The blocks: a byte count, then the data, back to back.
        let tail = appended(&bytes);
        assert_eq!(words(&tail[..8]), [16]);
        assert_eq!(
            &tail[8..24],
            [0.0f32, 1.0, 2.0, 3.0].map(f32::to_le_bytes).concat()
        );
        assert_eq!(words(&tail[24..32]), [16]);
        assert_eq!(words(&tail[48..56]), [96]);
        assert_eq!(words(&tail[152..160]), [48]);
        assert_eq!(words(&tail[160..208]), [0, 1, 2, 1, 3, 2]);
        assert_eq!(words(&tail[208..232]), [16, 3, 6]);
        assert_eq!(words(&tail[232..240]), [2]);
        assert_eq!(&tail[240..], b"\x05\x05\n  </AppendedData>\n</VTKFile>\n");
    }

    #[test]
    fn write_base64_inline() {
        let text = String::from_utf8(write(&square(), Binary::Base64, false)).unwrap();
        assert!(!text.contains("AppendedData"));
        // Uncompressed: the byte count and the data form one base64 stream.
        let raw = [0i64, 1, 2, 1, 3, 2].map(i64::to_le_bytes).concat();
        let mut expected = Vec::new();
        base64::encode_into(
            &[&48u64.to_le_bytes()[..], &raw[..]].concat(),
            &mut expected,
        );
        let expected = String::from_utf8(expected).unwrap();
        assert!(text.contains(&format!(
            "        <DataArray type=\"Int64\" Name=\"connectivity\" NumberOfComponents=\"1\" \
             format=\"binary\">\n          {expected}\n        </DataArray>\n"
        )));
        assert!(text.ends_with("    </Piece>\n  </UnstructuredGrid>\n</VTKFile>\n"));
    }

    #[test]
    fn write_base64_zlib_header_and_blocks_are_separate_streams() {
        let text = String::from_utf8(write(&square(), Binary::Base64, true)).unwrap();
        let open = "Name=\"types\" NumberOfComponents=\"1\" format=\"binary\">\n          ";
        let start = text.find(open).unwrap() + open.len();
        let line = &text.as_bytes()[start..start + text[start..].find('\n').unwrap()];
        // The header [1 block, 32768, 2 bytes, compressed size] is 32 bytes: 44 characters
        // ending in one `=`, then the zlib stream.
        assert_eq!(line[43], b'=');
        let header = base64::decode(&line[..44]).unwrap();
        let w = words(&header);
        assert_eq!(&w[..3], [1, BLOCK_SIZE as u64, 2]);
        let stream = base64::decode(&line[44..]).unwrap();
        assert_eq!(stream.len() as u64, w[3]);
        let mut types = Vec::new();
        ZlibDecoder::new(&stream[..])
            .read_to_end(&mut types)
            .unwrap();
        assert_eq!(types, [5, 5]);
        // The whole line decodes to the header followed by the stream.
        assert_eq!(base64::decode(line).unwrap(), [header, stream].concat());
    }

    /// Decompresses a stored zlib array of `raw_len` bytes, checking its header.
    fn unzip(stored: &Stored, raw_len: usize) -> Vec<u8> {
        let w = words(&stored.header);
        let blocks = raw_len.div_ceil(BLOCK_SIZE);
        assert_eq!(w.len(), 3 + blocks);
        assert_eq!(
            &w[..3],
            [
                blocks as u64,
                BLOCK_SIZE as u64,
                (raw_len % BLOCK_SIZE) as u64
            ]
        );
        assert_eq!(w[3..].iter().sum::<u64>(), stored.data.len() as u64);
        let mut out = Vec::new();
        let mut at = 0;
        for (k, &size) in w[3..].iter().enumerate() {
            let mut block = Vec::new();
            ZlibDecoder::new(&stored.data[at..at + size as usize])
                .read_to_end(&mut block)
                .unwrap();
            let full = k + 1 < blocks || raw_len.is_multiple_of(BLOCK_SIZE);
            let expected = if full {
                BLOCK_SIZE
            } else {
                raw_len % BLOCK_SIZE
            };
            assert_eq!(block.len(), expected);
            out.extend_from_slice(&block);
            at += size as usize;
        }
        out
    }

    #[test]
    fn write_zlib_block_headers() {
        // More than one block, exact multiples of the block size, one value, none.
        for n in [5000usize, 4096, 8192, 1, 0] {
            let values: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
            let bytes = with_header_room(&values);
            let raw = bytes[WORD..].to_vec();
            let stored = store(bytes, true).unwrap();
            assert_eq!(unzip(&stored, raw.len()), raw, "{n} values");
        }
        // The headers VTK 9.7's vtkXMLWriter writes for 5000, 4096, 8192, 1 and 0 doubles.
        let header = |n: usize| {
            let stored = store(with_header_room(&vec![0.0f64; n]), true).unwrap();
            words(&stored.header)[..3].to_vec()
        };
        assert_eq!(header(5000), [2, 32768, 7232]);
        assert_eq!(header(4096), [1, 32768, 0]);
        assert_eq!(header(8192), [2, 32768, 0]);
        assert_eq!(header(1), [1, 32768, 8]);
        assert_eq!(header(0), [0, 32768, 0]);
    }

    #[test]
    fn write_appended_zlib_offsets_point_at_headers() {
        let mut file = square();
        file.cell_data.push(DataArray {
            name: "big".into(),
            components: 5000,
            values: DataValues::F64((0..10000).map(f64::from).collect()),
        });
        let bytes = write(&file, Binary::Appended, true);
        let text = String::from_utf8_lossy(&bytes);
        let offsets: Vec<usize> = text
            .match_indices("offset=\"")
            .map(|(i, m)| {
                let s = &text[i + m.len()..];
                s[..s.find('"').unwrap()].parse().unwrap()
            })
            .collect();
        let tail = appended(&bytes);
        // u, id, big, points, connectivity, offsets, types.
        let raw_lens = [16, 16, 80000, 96, 48, 16, 2];
        assert_eq!(offsets.len(), raw_lens.len());
        let mut end = 0;
        for (&at, raw_len) in offsets.iter().zip(raw_lens) {
            assert_eq!(at, end);
            let blocks = words(&tail[at..at + 8])[0] as usize;
            let header = &tail[at..at + 8 * (3 + blocks)];
            let size: u64 = words(header)[3..].iter().sum();
            end = at + header.len() + size as usize;
            let stored = Stored {
                header: header.to_vec(),
                data: tail[at + header.len()..end].to_vec(),
            };
            let raw = unzip(&stored, raw_len);
            if raw_len == 80000 {
                assert_eq!(raw[79992..], 9999.0f64.to_le_bytes());
            }
        }
        assert_eq!(&tail[end..], b"\n  </AppendedData>\n</VTKFile>\n");
    }

    #[test]
    fn write_is_deterministic() {
        for binary in [Binary::Appended, Binary::Base64] {
            for zlib in [false, true] {
                assert_eq!(
                    write(&square(), binary, zlib),
                    write(&square(), binary, zlib)
                );
            }
        }
    }

    #[test]
    fn write_escapes_names() {
        let mut file = square();
        file.point_data[0].name = "a&b<c>d\"e'f\tg\nh".into();
        let escaped = "Name=\"a&amp;b&lt;c&gt;d&quot;e&apos;f&#9;g&#10;h\" NumberOfComponents";
        for binary in [Binary::Appended, Binary::Base64] {
            let bytes = write(&file, binary, false);
            assert!(String::from_utf8_lossy(&bytes).contains(escaped));
        }
        let mut bytes = Vec::new();
        write_pvtu(&mut bytes, &["a&b\"<>.vtu".to_string()], &file).unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert!(text.contains(escaped));
        assert!(text.contains("<Piece Source=\"a&amp;b&quot;&lt;&gt;.vtu\"/>"));
    }

    #[test]
    fn write_rejects_arrays_that_do_not_fit() {
        type Edit = fn(&mut VtuFile);
        let cases: [(Edit, usize, usize); 9] = [
            (|f| f.points = DataValues::F64(vec![0.0; 13]), 12, 13),
            (|f| f.types.push(5), 3, 2),
            (|f| f.offsets = vec![4, 3], 4, 3),
            (|f| f.offsets = vec![-1, 6], 0, 0),
            (|f| f.offsets = vec![3, 5], 6, 5),
            (|f| f.connectivity.push(0), 7, 6),
            (
                |f| f.point_data[0].values = DataValues::F32(vec![0.0; 5]),
                4,
                5,
            ),
            (|f| f.cell_data[0].components = 3, 6, 4),
            (|f| f.cell_data[0].components = 0, 1, 0),
        ];
        for (k, (edit, expected, found)) in cases.into_iter().enumerate() {
            let mut file = square();
            edit(&mut file);
            for binary in [Binary::Appended, Binary::Base64] {
                let mut bytes = Vec::new();
                let e = file.write_to(&mut bytes, binary, true).unwrap_err();
                assert!(
                    matches!(e, GridError::LengthMismatch { expected: x, found: y }
                        if (x, y) == (expected, found)),
                    "case {k}: {e:?}"
                );
                assert!(bytes.is_empty());
            }
        }
        // A file without points or cells is valid.
        let empty = VtuFile {
            points: DataValues::F32(vec![]),
            connectivity: vec![],
            offsets: vec![],
            types: vec![],
            point_data: vec![],
            cell_data: vec![],
        };
        for binary in [Binary::Appended, Binary::Base64] {
            for zlib in [false, true] {
                write(&empty, binary, zlib);
            }
        }
    }

    struct Failing;

    impl Write for Failing {
        fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
            Err(std::io::Error::other("disk full"))
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn write_maps_io_errors() {
        let e = square()
            .write_to(&mut Failing, Binary::Appended, false)
            .unwrap_err();
        assert!(
            matches!(e, GridError::Io(ref m) if m.contains("disk full")),
            "{e:?}"
        );
        let e = write_pvtu(&mut Failing, &[], &square()).unwrap_err();
        assert!(matches!(e, GridError::Io(_)), "{e:?}");
    }

    #[test]
    fn write_pvtu_golden() {
        let mut bytes = Vec::new();
        let pieces = ["out_0.vtu".to_string(), "out_1.vtu".to_string()];
        write_pvtu(&mut bytes, &pieces, &square()).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "<?xml version=\"1.0\"?>\n\
             <VTKFile type=\"PUnstructuredGrid\" version=\"2.2\" byte_order=\"LittleEndian\" \
             header_type=\"UInt64\">\n\
             \x20 <PUnstructuredGrid GhostLevel=\"0\">\n\
             \x20   <PPointData>\n\
             \x20     <PDataArray type=\"Float32\" Name=\"u\" NumberOfComponents=\"1\"/>\n\
             \x20   </PPointData>\n\
             \x20   <PCellData>\n\
             \x20     <PDataArray type=\"UInt32\" Name=\"id\" NumberOfComponents=\"2\"/>\n\
             \x20   </PCellData>\n\
             \x20   <PPoints>\n\
             \x20     <PDataArray type=\"Float64\" Name=\"Points\" NumberOfComponents=\"3\"/>\n\
             \x20   </PPoints>\n\
             \x20   <Piece Source=\"out_0.vtu\"/>\n\
             \x20   <Piece Source=\"out_1.vtu\"/>\n\
             \x20 </PUnstructuredGrid>\n\
             </VTKFile>\n"
        );
    }
}
