//! Chunked reading of binary MSH 4.1 files (docs/design.md, "Gmsh MSH 4.1", parallel read;
//! ADR-0035, rule 1): [`MshIndex::scan`] reads the small sections and the node and element
//! block headers, seeking over the block data, and [`MshIndex::read_nodes`] and
//! [`MshIndex::read_elements`] read only the bytes of a range of nodes or elements.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use gridcl_device::{GridError, Result};

use super::parse::{Binary, format_of, small_section};
use super::{BlockHeader, ElementBlock, FORMAT, MshFile, element_type};

/// The bytes one read of a scan asks for.
const STEP: usize = 1 << 16;

/// One node block: where its tags and coordinates are.
#[derive(Clone, Debug)]
struct NodeBlockAt {
    /// The index of its first node over all blocks.
    first: usize,
    count: usize,
    /// The doubles per node: 3, plus the parametric coordinates.
    doubles: usize,
    /// The byte offset of its first tag.
    tags_at: u64,
}

/// One element block: its header and where its elements are.
#[derive(Clone, Debug)]
struct ElementBlockAt {
    dim: usize,
    entity_tag: i32,
    element_type: i32,
    nodes_per_element: usize,
    /// The index of its first element among the elements of `dim`.
    first: usize,
    count: usize,
    /// The byte offset of its first element.
    at: u64,
}

/// The layout of an MSH 4.1 file: its encoding, its small sections (`$PhysicalNames`,
/// `$Entities`, `$PartitionedEntities`, `$Periodic`) and the byte offsets of its node and
/// element blocks.
///
/// An ASCII file is scanned only up to its `$MeshFormat`: [`is_binary`](MshIndex::is_binary)
/// is false and its chunks are refused (the parallel reader reads it on rank 0, ADR-0035,
/// rule 2).
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::{MshIndex, read_file};
///
/// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/triangle_p1-binary.msh"));
/// let index = MshIndex::scan(path)?;
/// assert!(index.is_binary());
/// // The second half of the triangles, as rank 1 of 2 reads them.
/// let n = index.num_elements(2);
/// let chunk = index.read_elements(2, n / 2..n)?;
/// let tags: Vec<i64> = chunk.blocks.iter().flat_map(|b| b.tags.clone()).collect();
/// let file = read_file(path)?;
/// let all: Vec<i64> = file.element_blocks.iter().filter(|b| b.dim == 2).flat_map(|b| b.tags.clone()).collect();
/// assert_eq!(tags, all[n / 2..]);
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Debug)]
pub struct MshIndex {
    path: PathBuf,
    binary: Option<Binary>,
    sections: MshFile,
    num_nodes: usize,
    node_blocks: Vec<NodeBlockAt>,
    num_elements: [usize; 4],
    element_blocks: Vec<ElementBlockAt>,
    /// The bytes the chunk reads have read.
    bytes_read: AtomicU64,
}

/// A range of nodes in file order: their tags and coordinates.
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::MshIndex;
///
/// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/interval_p1-binary.msh"));
/// let chunk = MshIndex::scan(path)?.read_nodes(0..2)?;
/// assert_eq!((chunk.tags.len(), chunk.coords.len()), (2, 2));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeChunk {
    /// The node tags.
    pub tags: Vec<i64>,
    /// The coordinates.
    pub coords: Vec<[f64; 3]>,
}

/// A range of the elements of one dimension in file order, as the element blocks that hold
/// them, cut to the range.
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::MshIndex;
///
/// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/interval_p1-binary.msh"));
/// let chunk = MshIndex::scan(path)?.read_elements(1, 0..1)?;
/// // One 2-node line (type 1).
/// assert_eq!((chunk.blocks[0].element_type, chunk.blocks[0].nodes.len()), (1, 2));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementChunk {
    /// The blocks, in file order.
    pub blocks: Vec<ElementBlock>,
}

/// `Parse` at `offset`.
fn parse_error(offset: u64, message: impl Into<String>) -> GridError {
    GridError::Parse {
        format: FORMAT,
        offset,
        message: message.into(),
    }
}

/// A file read at offsets, counting the bytes read.
struct Reader<'a> {
    file: File,
    len: u64,
    path: &'a Path,
    read: u64,
}

impl<'a> Reader<'a> {
    fn open(path: &'a Path) -> Result<Self> {
        let io = |e: std::io::Error| GridError::Io(format!("{}: {e}", path.display()));
        let file = File::open(path).map_err(io)?;
        let len = file.metadata().map_err(io)?.len();
        Ok(Reader {
            file,
            len,
            path,
            read: 0,
        })
    }

    fn io(&self, e: std::io::Error) -> GridError {
        GridError::Io(format!("{}: {e}", self.path.display()))
    }

    /// The `n` bytes at `offset`; `Parse` at `offset` if the file ends before.
    fn at(&mut self, offset: u64, n: usize) -> Result<Vec<u8>> {
        let end = offset.checked_add(n as u64).filter(|&e| e <= self.len);
        if end.is_none() {
            return Err(parse_error(offset, "unexpected end of file"));
        }
        let mut buf = vec![0; n];
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| self.io(e))?;
        self.file.read_exact(&mut buf).map_err(|e| self.io(e))?;
        self.read += n as u64;
        Ok(buf)
    }

    /// Up to `n` bytes at `offset`, fewer at the end of the file.
    fn up_to(&mut self, offset: u64, n: usize) -> Result<Vec<u8>> {
        let n = n.min(self.len.saturating_sub(offset) as usize);
        self.at(offset, n)
    }

    /// The bytes from `offset` through the end of `marker`, kept if `keep`, else only the
    /// offset after it; `Parse` at `offset` if the file has no `marker` after it.
    fn through(&mut self, offset: u64, marker: &[u8], keep: bool) -> Result<(Vec<u8>, u64)> {
        let mut kept = Vec::new();
        let mut window: Vec<u8> = Vec::new();
        let mut pos = offset;
        loop {
            let bytes = self.up_to(pos, STEP)?;
            if bytes.is_empty() {
                let name = String::from_utf8_lossy(marker);
                return Err(parse_error(offset, format!("no {name}")));
            }
            let start = window.len();
            window.extend_from_slice(&bytes);
            if let Some(i) = window.windows(marker.len()).position(|w| w == marker) {
                let end = i + marker.len();
                if keep {
                    kept.extend_from_slice(&window[..end]);
                }
                let after = pos - start as u64 + end as u64;
                return Ok((kept, after));
            }
            pos += bytes.len() as u64;
            // Keep the tail that may hold the start of the marker.
            let tail = window.len().saturating_sub(marker.len() - 1);
            if keep {
                kept.extend_from_slice(&window[..tail]);
            }
            window.drain(..tail);
        }
    }

    /// The next whitespace-separated token after `offset`, its offset and the offset after it;
    /// `None` at the end of the file.
    fn token(&mut self, offset: u64) -> Result<Option<(u64, String, u64)>> {
        let bytes = self.up_to(offset, 256)?;
        let Some(start) = bytes.iter().position(|b| !b.is_ascii_whitespace()) else {
            if offset + (bytes.len() as u64) < self.len {
                return Err(parse_error(offset, "expected a section"));
            }
            return Ok(None);
        };
        let len = bytes[start..]
            .iter()
            .position(|b| b.is_ascii_whitespace())
            .unwrap_or(bytes.len() - start);
        let text = String::from_utf8_lossy(&bytes[start..start + len]).into_owned();
        let at = offset + start as u64;
        Ok(Some((at, text, at + len as u64)))
    }

    /// The offset after the line break that ends a section header at `offset`.
    fn line_break(&mut self, offset: u64) -> Result<u64> {
        let b = self.up_to(offset, 2)?;
        match b.as_slice() {
            [b'\n', ..] => Ok(offset + 1),
            [b'\r', b'\n'] => Ok(offset + 2),
            _ => Err(parse_error(
                offset,
                "expected a line break after the section header",
            )),
        }
    }

    /// Expects the token `$End<section>` after `offset`; the offset after it.
    fn end(&mut self, offset: u64, section: &str) -> Result<u64> {
        match self.token(offset)? {
            Some((_, text, after)) if text.strip_prefix("$End") == Some(section) => Ok(after),
            Some((at, text, _)) => Err(parse_error(
                at,
                format!("expected $End{section}, found {text:?}"),
            )),
            None => Err(parse_error(offset, format!("expected $End{section}"))),
        }
    }
}

/// Decodes the values of one binary record.
struct Values<'a> {
    b: Binary,
    bytes: &'a [u8],
    pos: usize,
    /// The file offset of `bytes[0]`.
    base: u64,
}

impl Values<'_> {
    fn offset(&self) -> u64 {
        self.base + self.pos as u64
    }

    fn int(&mut self) -> i32 {
        let v = self.b.int(
            self.bytes[self.pos..self.pos + 4]
                .try_into()
                .expect("4 bytes"),
        );
        self.pos += 4;
        v
    }

    fn size(&mut self) -> u64 {
        let v = self.b.size(&self.bytes[self.pos..self.pos + self.b.size_t]);
        self.pos += self.b.size_t;
        v
    }

    fn float(&mut self) -> f64 {
        let v = self.b.float(
            self.bytes[self.pos..self.pos + 8]
                .try_into()
                .expect("8 bytes"),
        );
        self.pos += 8;
        v
    }

    /// A dimension 0–3, stored as an `int`.
    fn dim(&mut self) -> Result<usize> {
        let at = self.offset();
        let d = self.int();
        usize::try_from(d)
            .ok()
            .filter(|&d| d <= 3)
            .ok_or_else(|| parse_error(at, format!("dimension {d} is not 0-3")))
    }

    /// A node or element tag: a `size_t` ≥ 1 that fits an i64.
    fn tag(&mut self, what: &str) -> Result<i64> {
        let at = self.offset();
        let t = self.size();
        i64::try_from(t)
            .ok()
            .filter(|&t| t >= 1)
            .ok_or_else(|| parse_error(at, format!("{what} tag {t}: tags must be 1 to 2^63 - 1")))
    }
}

/// `count × bytes`, checked against the bytes left after `offset` in a file of `len` bytes;
/// `Parse` at `at` (the count's offset) otherwise.
fn extent(count: u64, bytes: u64, offset: u64, len: u64, at: u64) -> Result<u64> {
    count
        .checked_mul(bytes)
        .filter(|&n| offset.checked_add(n).is_some_and(|end| end <= len))
        .ok_or_else(|| parse_error(at, format!("count {count} exceeds the rest of the file")))
}

impl MshIndex {
    /// Scans the file at `path`: `$MeshFormat`, then for a binary file the small sections,
    /// parsed, and the node and element block headers, located by seeking over the block data.
    /// Other sections are skipped.
    ///
    /// # Errors
    ///
    /// `Io` if the file cannot be read; `Parse` with the byte offset for a malformed or
    /// truncated section; `UnsupportedElementType` for an element type not in A.7.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::MshIndex;
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/two_regions_p1-binary.msh"));
    /// let index = MshIndex::scan(path)?;
    /// assert_eq!(index.num_nodes(), 32);
    /// assert_eq!(index.num_elements(2), 35);
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn scan(path: &Path) -> Result<MshIndex> {
        let mut r = Reader::open(path)?;
        let (head, after) = r.through(0, b"$EndMeshFormat", true)?;
        let binary = format_of(&head)?;
        let mut index = MshIndex {
            path: path.to_path_buf(),
            binary,
            sections: MshFile::default(),
            num_nodes: 0,
            node_blocks: Vec::new(),
            num_elements: [0; 4],
            element_blocks: Vec::new(),
            bytes_read: AtomicU64::new(0),
        };
        let Some(b) = binary else {
            return Ok(index);
        };
        let mut pos = after;
        while let Some((at, header, after)) = r.token(pos)? {
            let Some(section) = header.strip_prefix('$') else {
                return Err(parse_error(
                    at,
                    format!("expected a section, found {header:?}"),
                ));
            };
            let body = r.line_break(after)?;
            pos = match section {
                "PhysicalNames" | "Entities" | "PartitionedEntities" | "Periodic" => {
                    let end = format!("$End{section}");
                    let (bytes, after) = r.through(body, end.as_bytes(), true)?;
                    small_section(section, &bytes, body, b, &mut index.sections)?;
                    after
                }
                "Nodes" => index.scan_nodes(&mut r, b, body)?,
                "Elements" => index.scan_elements(&mut r, b, body)?,
                "MeshFormat" => return Err(parse_error(at, "a second $MeshFormat")),
                _ if section.starts_with("End") => {
                    return Err(parse_error(at, format!("{header} without its section")));
                }
                _ => {
                    r.through(body, format!("$End{section}").as_bytes(), false)?
                        .1
                }
            };
        }
        Ok(index)
    }

    /// The node blocks of a `$Nodes` section whose data starts at `at`; the offset after
    /// `$EndNodes`.
    fn scan_nodes(&mut self, r: &mut Reader<'_>, b: Binary, at: u64) -> Result<u64> {
        let s = b.size_t as u64;
        let bytes = r.at(at, 4 * b.size_t)?;
        let mut v = Values {
            b,
            bytes: &bytes,
            pos: 0,
            base: at,
        };
        let num_blocks = v.size();
        let num_nodes = v.size();
        let mut pos = at + 4 * s;
        extent(num_blocks, 12 + s, pos, r.len, at)?;
        let mut total = 0usize;
        for _ in 0..num_blocks {
            let bytes = r.at(pos, 12 + b.size_t)?;
            let mut v = Values {
                b,
                bytes: &bytes,
                pos: 0,
                base: pos,
            };
            let dim = v.dim()?;
            let _entity_tag = v.int();
            let at_param = v.offset();
            let parametric = match v.int() {
                0 => 0,
                1 => dim,
                p => {
                    return Err(parse_error(
                        at_param,
                        format!("parametric flag {p} is not 0 or 1"),
                    ));
                }
            };
            let at_count = v.offset();
            let count = v.size();
            let doubles = 3 + parametric;
            let tags_at = pos + 12 + s;
            let len = extent(count, s + 8 * doubles as u64, tags_at, r.len, at_count)?;
            let count = count as usize;
            self.node_blocks.push(NodeBlockAt {
                first: total,
                count,
                doubles,
                tags_at,
            });
            total += count;
            pos = tags_at + len;
        }
        if total as u64 != num_nodes {
            return Err(parse_error(
                at,
                format!("{num_nodes} nodes announced, {total} found"),
            ));
        }
        self.num_nodes += total;
        r.end(pos, "Nodes")
    }

    /// The element blocks of an `$Elements` section whose data starts at `at`; the offset
    /// after `$EndElements`.
    fn scan_elements(&mut self, r: &mut Reader<'_>, b: Binary, at: u64) -> Result<u64> {
        let s = b.size_t as u64;
        let bytes = r.at(at, 4 * b.size_t)?;
        let mut v = Values {
            b,
            bytes: &bytes,
            pos: 0,
            base: at,
        };
        let num_blocks = v.size();
        let num_elements = v.size();
        let mut pos = at + 4 * s;
        extent(num_blocks, 12 + s, pos, r.len, at)?;
        let mut total = 0u64;
        for _ in 0..num_blocks {
            let bytes = r.at(pos, 12 + b.size_t)?;
            let mut v = Values {
                b,
                bytes: &bytes,
                pos: 0,
                base: pos,
            };
            let dim = v.dim()?;
            let entity_tag = v.int();
            let at_type = v.offset();
            let id = v.int();
            let ty = element_type(id)?;
            if gridcl_refcell::tdim(ty.cell) != dim {
                return Err(parse_error(
                    at_type,
                    format!("element type {id} in a block of dimension {dim}"),
                ));
            }
            let at_count = v.offset();
            let count = v.size();
            let first_at = pos + 12 + s;
            let k = ty.num_nodes as u64;
            let len = extent(count, (1 + k) * s, first_at, r.len, at_count)?;
            let count = count as usize;
            self.element_blocks.push(ElementBlockAt {
                dim,
                entity_tag,
                element_type: id,
                nodes_per_element: ty.num_nodes,
                first: self.num_elements[dim],
                count,
                at: first_at,
            });
            self.num_elements[dim] += count;
            total += count as u64;
            pos = first_at + len;
        }
        if total != num_elements {
            return Err(parse_error(
                at,
                format!("{num_elements} elements announced, {total} found"),
            ));
        }
        r.end(pos, "Elements")
    }

    /// Whether the file is binary; an ASCII file has no block offsets.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::MshIndex;
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/triangle_p1.msh"));
    /// assert!(!MshIndex::scan(path)?.is_binary());
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn is_binary(&self) -> bool {
        self.binary.is_some()
    }

    /// The number of nodes (0 for an ASCII file).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::MshIndex;
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/triangle_p1-binary.msh"));
    /// let index = MshIndex::scan(path)?;
    /// assert_eq!(index.read_nodes(0..index.num_nodes())?.tags.len(), index.num_nodes());
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn num_nodes(&self) -> usize {
        self.num_nodes
    }

    /// The number of elements of dimension `dim` (0 for an ASCII file or `dim` > 3).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::MshIndex;
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/tetrahedron_p1-binary.msh"));
    /// let index = MshIndex::scan(path)?;
    /// assert_eq!(index.num_elements(3), 100);
    /// assert_eq!(index.num_elements(4), 0);
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn num_elements(&self, dim: usize) -> usize {
        self.num_elements.get(dim).copied().unwrap_or(0)
    }

    /// The encoding, or `Parse` for an ASCII file.
    fn encoding(&self) -> Result<Binary> {
        self.binary
            .ok_or_else(|| parse_error(0, "chunked reading needs a binary file"))
    }

    /// The nodes `range` in file order over all node blocks.
    ///
    /// # Errors
    ///
    /// `Parse` for an ASCII file or a node tag ≤ 0; `IndexOutOfRange` if `range` ends beyond
    /// [`num_nodes`](MshIndex::num_nodes) or is reversed; `Io`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::{MshIndex, read_file};
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/quadrilateral_p2-binary.msh"));
    /// let chunk = MshIndex::scan(path)?.read_nodes(4..9)?;
    /// let file = read_file(path)?;
    /// let tags: Vec<i64> = file.node_blocks.iter().flat_map(|b| b.tags.clone()).collect();
    /// assert_eq!(chunk.tags, tags[4..9]);
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn read_nodes(&self, range: Range<usize>) -> Result<NodeChunk> {
        let b = self.encoding()?;
        if range.start > range.end || range.end > self.num_nodes {
            return Err(GridError::IndexOutOfRange {
                index: range.end.max(range.start),
                len: self.num_nodes,
            });
        }
        let mut out = NodeChunk {
            tags: Vec::with_capacity(range.len()),
            coords: Vec::with_capacity(range.len()),
        };
        if range.is_empty() {
            return Ok(out);
        }
        let mut r = Reader::open(&self.path)?;
        let s = b.size_t;
        for block in &self.node_blocks {
            let lo = range.start.max(block.first);
            let hi = range.end.min(block.first + block.count);
            if lo >= hi {
                continue;
            }
            let (lo, n) = ((lo - block.first) as u64, hi - lo);
            let at = block.tags_at + lo * s as u64;
            let bytes = r.at(at, n * s)?;
            let mut v = Values {
                b,
                bytes: &bytes,
                pos: 0,
                base: at,
            };
            for _ in 0..n {
                out.tags.push(v.tag("node")?);
            }
            let stride = 8 * block.doubles;
            let at = block.tags_at + (block.count * s) as u64 + lo * stride as u64;
            let bytes = r.at(at, n * stride)?;
            let mut v = Values {
                b,
                bytes: &bytes,
                pos: 0,
                base: at,
            };
            for _ in 0..n {
                out.coords.push([v.float(), v.float(), v.float()]);
                v.pos += 8 * (block.doubles - 3);
            }
        }
        self.bytes_read.fetch_add(r.read, Ordering::Relaxed);
        Ok(out)
    }

    /// The elements `range` of dimension `dim`, in file order within the dimension, as the
    /// blocks that hold them cut to the range.
    ///
    /// # Errors
    ///
    /// `Parse` for an ASCII file or a tag ≤ 0; `IndexOutOfRange` if `range` ends beyond
    /// [`num_elements`](MshIndex::num_elements) or is reversed; `Io`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_io::gmsh::MshIndex;
    ///
    /// let path = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/hybrid_p1-binary.msh"));
    /// let index = MshIndex::scan(path)?;
    /// // Rank 2 of 3: the last third of the cells, which may span blocks of several types.
    /// let n = index.num_elements(3);
    /// let chunk = index.read_elements(3, 2 * n / 3..n)?;
    /// let count: usize = chunk.blocks.iter().map(|b| b.tags.len()).sum();
    /// assert_eq!(count, n - 2 * n / 3);
    /// # Ok::<(), gridcl_device::GridError>(())
    /// ```
    pub fn read_elements(&self, dim: usize, range: Range<usize>) -> Result<ElementChunk> {
        let b = self.encoding()?;
        let n_dim = self.num_elements(dim);
        if range.start > range.end || range.end > n_dim {
            return Err(GridError::IndexOutOfRange {
                index: range.end.max(range.start),
                len: n_dim,
            });
        }
        let mut out = ElementChunk::default();
        if range.is_empty() {
            return Ok(out);
        }
        let mut r = Reader::open(&self.path)?;
        let s = b.size_t;
        for block in self.element_blocks.iter().filter(|e| e.dim == dim) {
            let lo = range.start.max(block.first);
            let hi = range.end.min(block.first + block.count);
            if lo >= hi {
                continue;
            }
            let (lo, n) = (lo - block.first, hi - lo);
            let k = block.nodes_per_element;
            let record = (1 + k) * s;
            let at = block.at + (lo * record) as u64;
            let bytes = r.at(at, n * record)?;
            let mut v = Values {
                b,
                bytes: &bytes,
                pos: 0,
                base: at,
            };
            let mut tags = Vec::with_capacity(n);
            let mut nodes = Vec::with_capacity(n * k);
            for _ in 0..n {
                tags.push(v.tag("element")?);
                for _ in 0..k {
                    nodes.push(v.tag("node")?);
                }
            }
            out.blocks.push(ElementBlock {
                dim,
                entity_tag: block.entity_tag,
                element_type: block.element_type,
                tags,
                nodes,
            });
        }
        self.bytes_read.fetch_add(r.read, Ordering::Relaxed);
        Ok(out)
    }

    /// The small sections of a binary file (`$PhysicalNames`, `$Entities`,
    /// `$PartitionedEntities`, `$Periodic`), without node or element blocks. For gridcl's
    /// parallel reader (ADR-0035).
    #[doc(hidden)]
    pub fn sections(&self) -> &MshFile {
        &self.sections
    }

    /// The headers of the element blocks in file order, for the chunk conversion's
    /// [`Layout`](super::Layout) (ADR-0035).
    #[doc(hidden)]
    pub fn block_headers(&self) -> Vec<BlockHeader> {
        self.element_blocks
            .iter()
            .map(|b| BlockHeader {
                dim: b.dim,
                entity_tag: b.entity_tag,
                element_type: b.element_type,
                count: b.count,
            })
            .collect()
    }

    /// The bytes the chunk reads have read so far: a test accessor (ADR-0035, T4b).
    #[doc(hidden)]
    pub fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }
}
