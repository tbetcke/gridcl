//! The MSH 4.1 parser: ASCII and binary, either byte order, 4- or 8-byte `size_t`.
//!
//! One [`Cursor`] reads the values of a section: whitespace-separated tokens in ASCII files,
//! fixed-width little- or big-endian values in binary ones (`$PhysicalNames` is text in both).
//! The section parsers are written once against it. Counts are checked against the bytes
//! left before anything is allocated from them ([`Cursor::count`]).

mod binary;

use std::path::Path;

use binary::Binary;

use gridcl_device::{GridError, Result};

use super::{
    ElementBlock, Entities, Entity, FORMAT, MshFile, NodeBlock, PartitionedEntities,
    PartitionedEntity, Periodic, PhysicalName,
};

/// A position in the file and the reads of one value kind.
struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
    /// The encoding of the values of a binary file, set by `$MeshFormat`; `None` for text.
    binary: Option<Binary>,
}

/// The smallest number of bytes one ASCII token takes: a digit and a separator.
const TOKEN: usize = 2;

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Cursor {
            bytes,
            pos: 0,
            binary: None,
        }
    }

    /// `Parse` at byte `offset`.
    fn error_at(&self, offset: usize, message: impl Into<String>) -> GridError {
        GridError::Parse {
            format: FORMAT,
            offset: offset as u64,
            message: message.into(),
        }
    }

    /// `Parse` at the current position.
    fn error(&self, message: impl Into<String>) -> GridError {
        self.error_at(self.pos, message)
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn at_end(&mut self) -> bool {
        self.skip_whitespace();
        self.pos == self.bytes.len()
    }

    /// The next whitespace-separated token and its offset.
    fn token(&mut self) -> Result<(usize, &'a str)> {
        self.skip_whitespace();
        let start = self.pos;
        while self.pos < self.bytes.len() && !self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(self.error("unexpected end of file"));
        }
        let text = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| self.error_at(start, "a token that is not UTF-8"))?;
        Ok((start, text))
    }

    /// The next token parsed as `T`.
    fn parse<T: std::str::FromStr>(&mut self, what: &str) -> Result<T> {
        let (at, text) = self.token()?;
        text.parse()
            .map_err(|_| self.error_at(at, format!("expected {what}, found {text:?}")))
    }

    /// The next `n` bytes of a binary value.
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.remaining() < n {
            return Err(self.error("unexpected end of file"));
        }
        let b = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(b)
    }

    /// The next `N` bytes of a binary value.
    fn take_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        Ok(self.take(N)?.try_into().expect("N bytes"))
    }

    /// An `int` of the format.
    fn int(&mut self) -> Result<i32> {
        match self.binary {
            Some(b) => Ok(b.int(self.take_array()?)),
            None => self.parse("an integer"),
        }
    }

    /// A `size_t` of the format.
    fn size(&mut self) -> Result<u64> {
        match self.binary {
            Some(b) => Ok(b.size(self.take(b.size_t)?)),
            None => self.parse("an unsigned integer"),
        }
    }

    /// A `double` of the format.
    fn float(&mut self) -> Result<f64> {
        match self.binary {
            Some(b) => Ok(b.float(self.take_array()?)),
            None => self.parse("a number"),
        }
    }

    /// Consumes the line break after a section header; binary data starts after it.
    fn header_end(&mut self) -> Result<()> {
        if self.binary.is_some() {
            if self.bytes.get(self.pos) == Some(&b'\r') {
                self.pos += 1;
            }
            if self.bytes.get(self.pos) != Some(&b'\n') {
                return Err(self.error("expected a line break after the section header"));
            }
            self.pos += 1;
        }
        Ok(())
    }

    /// A dimension 0–3, stored as an `int`.
    fn dim(&mut self) -> Result<usize> {
        let at = self.pos;
        let d = self.int()?;
        match usize::try_from(d) {
            Ok(d) if d <= 3 => Ok(d),
            _ => Err(self.error_at(at, format!("dimension {d} is not 0-3"))),
        }
    }

    /// A count of items that take at least `min_bytes` bytes each. Fails if the rest of the
    /// file cannot hold them, so the count can size an allocation.
    fn count(&mut self, min_bytes: usize) -> Result<usize> {
        let at = self.pos;
        let n = self.size()?;
        let fits = usize::try_from(n).ok().filter(|&n| {
            n.checked_mul(min_bytes)
                .is_some_and(|b| b <= self.remaining())
        });
        fits.ok_or_else(|| self.error_at(at, format!("count {n} exceeds the rest of the file")))
    }

    /// A node or element tag: a `size_t` ≥ 1 that fits an i64.
    fn tag(&mut self, what: &str) -> Result<i64> {
        let at = self.pos;
        let t = self.size()?;
        match i64::try_from(t) {
            Ok(t) if t >= 1 => Ok(t),
            _ => Err(self.error_at(at, format!("{what} tag {t}: tags must be 1 to 2^63 - 1"))),
        }
    }

    /// The bytes one value of each kind takes at least: (int, size_t, double).
    fn widths(&self) -> (usize, usize, usize) {
        match self.binary {
            Some(b) => (4, b.size_t, 8),
            None => (TOKEN, TOKEN, TOKEN),
        }
    }

    /// Expects the line `$End<section>`.
    fn end(&mut self, section: &str) -> Result<()> {
        let (at, text) = self.token()?;
        if text.strip_prefix("$End") == Some(section) {
            Ok(())
        } else {
            Err(self.error_at(at, format!("expected $End{section}, found {text:?}")))
        }
    }

    /// Skips an unknown section up to and including `$End<section>`.
    fn skip_section(&mut self, section: &str) -> Result<()> {
        let end = format!("$End{section}");
        let at = self.pos;
        let found = self.bytes[self.pos..]
            .windows(end.len())
            .position(|w| w == end.as_bytes())
            .ok_or_else(|| self.error_at(at, format!("no {end}")))?;
        self.pos += found + end.len();
        Ok(())
    }
}

/// `$MeshFormat`: version 4.1; for binary files the endianness marker, which sets the
/// encoding of the cursor.
fn mesh_format(c: &mut Cursor<'_>) -> Result<()> {
    let (at, version) = c.token()?;
    if version != "4.1" {
        return Err(c.error_at(at, format!("MSH version {version}: only 4.1 is supported")));
    }
    let at = c.pos;
    let file_type = c.int()?;
    let data_size = c.int()?;
    if data_size != 4 && data_size != 8 {
        return Err(c.error_at(at, format!("data size {data_size} is not 4 or 8")));
    }
    match file_type {
        0 => {}
        1 => {
            // The line break, then the int 1 in the file's byte order.
            c.binary = Some(Binary {
                big_endian: false,
                size_t: data_size as usize,
            });
            c.header_end()?;
            let at = c.pos;
            let one = c.take_array()?;
            c.binary = Some(
                Binary::from_marker(one, data_size as usize)
                    .ok_or_else(|| c.error_at(at, "an endianness marker that is not 1"))?,
            );
        }
        _ => return Err(c.error_at(at, format!("file type {file_type} is not 0 or 1"))),
    }
    c.end("MeshFormat")
}

/// `$PhysicalNames`: text in binary files too.
fn physical_names(c: &mut Cursor<'_>) -> Result<Vec<PhysicalName>> {
    let binary = c.binary.take();
    let names = physical_name_list(c);
    c.binary = binary;
    names
}

/// The entries of `$PhysicalNames`, read as text.
fn physical_name_list(c: &mut Cursor<'_>) -> Result<Vec<PhysicalName>> {
    let n = c.count(3 * TOKEN + 2)?;
    let mut names = Vec::with_capacity(n);
    for _ in 0..n {
        let dim = c.dim()?;
        let tag = c.int()?;
        c.skip_whitespace();
        let at = c.pos;
        let rest = &c.bytes[c.pos..];
        if rest.first() != Some(&b'"') {
            return Err(c.error("expected a quoted name"));
        }
        let len = rest[1..]
            .iter()
            .position(|&b| b == b'"')
            .ok_or_else(|| c.error_at(at, "unterminated name"))?;
        let name = std::str::from_utf8(&rest[1..1 + len])
            .map_err(|_| c.error_at(at, "a name that is not UTF-8"))?;
        names.push(PhysicalName {
            dim,
            tag,
            name: name.to_owned(),
        });
        c.pos += len + 2;
    }
    c.end("PhysicalNames")?;
    Ok(names)
}

/// A list of `int`s preceded by its `size_t` length.
fn int_list(c: &mut Cursor<'_>) -> Result<Vec<i32>> {
    let (int, _, _) = c.widths();
    let n = c.count(int)?;
    (0..n).map(|_| c.int()).collect()
}

/// Three `double`s.
fn point(c: &mut Cursor<'_>) -> Result<[f64; 3]> {
    Ok([c.float()?, c.float()?, c.float()?])
}

/// One entity of dimension `dim` of `$Entities` (or its tail in `$PartitionedEntities`, after
/// the tag): coordinates or bounding box, physical tags, bounding entities.
fn entity_body(c: &mut Cursor<'_>, tag: i32, dim: usize) -> Result<Entity> {
    let min = point(c)?;
    let max = if dim == 0 { min } else { point(c)? };
    let physical_tags = int_list(c)?;
    let boundary = if dim == 0 { Vec::new() } else { int_list(c)? };
    Ok(Entity {
        tag,
        min,
        max,
        physical_tags,
        boundary,
    })
}

/// `$Entities`.
fn entities(c: &mut Cursor<'_>) -> Result<Entities> {
    let (int, _, float) = c.widths();
    let mut counts = [0usize; 4];
    for n in &mut counts {
        *n = c.count(int + 3 * float)?;
    }
    let mut out = Entities::default();
    for (dim, &n) in counts.iter().enumerate() {
        let list = match dim {
            0 => &mut out.points,
            1 => &mut out.curves,
            2 => &mut out.surfaces,
            _ => &mut out.volumes,
        };
        list.reserve(n);
        for _ in 0..n {
            let tag = c.int()?;
            list.push(entity_body(c, tag, dim)?);
        }
    }
    c.end("Entities")?;
    Ok(out)
}

/// `$PartitionedEntities`.
fn partitioned_entities(c: &mut Cursor<'_>) -> Result<PartitionedEntities> {
    let (int, _, float) = c.widths();
    let num_partitions = c.count(0)?;
    let num_ghosts = c.count(2 * int)?;
    let ghost_entities = (0..num_ghosts)
        .map(|_| Ok((c.int()?, c.int()?)))
        .collect::<Result<Vec<_>>>()?;
    let mut counts = [0usize; 4];
    for n in &mut counts {
        *n = c.count(3 * int + 3 * float)?;
    }
    let mut out = PartitionedEntities {
        num_partitions,
        ghost_entities,
        ..PartitionedEntities::default()
    };
    for (dim, &n) in counts.iter().enumerate() {
        let list = match dim {
            0 => &mut out.points,
            1 => &mut out.curves,
            2 => &mut out.surfaces,
            _ => &mut out.volumes,
        };
        list.reserve(n);
        for _ in 0..n {
            let tag = c.int()?;
            let parent_dim = c.dim()?;
            let parent_tag = c.int()?;
            let partitions = int_list(c)?;
            list.push(PartitionedEntity {
                entity: entity_body(c, tag, dim)?,
                parent_dim,
                parent_tag,
                partitions,
            });
        }
    }
    c.end("PartitionedEntities")?;
    Ok(out)
}

/// `$Periodic`.
fn periodic(c: &mut Cursor<'_>) -> Result<Vec<Periodic>> {
    let (int, size, float) = c.widths();
    let n = c.count(3 * int + 2 * size)?;
    let mut links = Vec::with_capacity(n);
    for _ in 0..n {
        let dim = c.dim()?;
        let tag = c.int()?;
        let master_tag = c.int()?;
        let num_affine = c.count(float)?;
        let affine = (0..num_affine).map(|_| c.float()).collect::<Result<_>>()?;
        let num_nodes = c.count(2 * size)?;
        let nodes = (0..num_nodes)
            .map(|_| Ok((c.tag("node")?, c.tag("node")?)))
            .collect::<Result<_>>()?;
        links.push(Periodic {
            dim,
            tag,
            master_tag,
            affine,
            nodes,
        });
    }
    c.end("Periodic")?;
    Ok(links)
}

/// `$Nodes`. Parametric coordinates are read and dropped.
fn nodes(c: &mut Cursor<'_>) -> Result<Vec<NodeBlock>> {
    let (int, size, float) = c.widths();
    let at = c.pos;
    let num_blocks = c.count(3 * int + size)?;
    let num_nodes = c.size()?;
    let _min_tag = c.size()?;
    let _max_tag = c.size()?;
    let mut blocks = Vec::with_capacity(num_blocks);
    let mut total = 0u64;
    for _ in 0..num_blocks {
        let dim = c.dim()?;
        let entity_tag = c.int()?;
        let at_param = c.pos;
        let parametric = match c.int()? {
            0 => 0,
            1 => dim,
            p => return Err(c.error_at(at_param, format!("parametric flag {p} is not 0 or 1"))),
        };
        let n = c.count(size + 3 * float)?;
        let tags = (0..n).map(|_| c.tag("node")).collect::<Result<Vec<_>>>()?;
        let mut coords = Vec::with_capacity(n);
        for _ in 0..n {
            coords.push(point(c)?);
            for _ in 0..parametric {
                c.float()?;
            }
        }
        total += n as u64;
        blocks.push(NodeBlock {
            dim,
            entity_tag,
            tags,
            coords,
        });
    }
    if total != num_nodes {
        return Err(c.error_at(at, format!("{num_nodes} nodes announced, {total} found")));
    }
    c.end("Nodes")?;
    Ok(blocks)
}

/// `$Elements`.
fn elements(c: &mut Cursor<'_>) -> Result<Vec<ElementBlock>> {
    let (int, size, _) = c.widths();
    let at = c.pos;
    let num_blocks = c.count(3 * int + size)?;
    let num_elements = c.size()?;
    let _min_tag = c.size()?;
    let _max_tag = c.size()?;
    let mut blocks = Vec::with_capacity(num_blocks);
    let mut total = 0u64;
    for _ in 0..num_blocks {
        let dim = c.dim()?;
        let entity_tag = c.int()?;
        let at_type = c.pos;
        let element_type = c.int()?;
        let ty = super::element_type(element_type)?;
        if gridcl_refcell::tdim(ty.cell) != dim {
            return Err(c.error_at(
                at_type,
                format!("element type {element_type} in a block of dimension {dim}"),
            ));
        }
        let k = ty.num_nodes;
        let n = c.count((1 + k) * size)?;
        let mut tags = Vec::with_capacity(n);
        let mut nodes = Vec::with_capacity(n * k);
        for _ in 0..n {
            tags.push(c.tag("element")?);
            for _ in 0..k {
                nodes.push(c.tag("node")?);
            }
        }
        total += n as u64;
        blocks.push(ElementBlock {
            dim,
            entity_tag,
            element_type,
            tags,
            nodes,
        });
    }
    if total != num_elements {
        return Err(c.error_at(
            at,
            format!("{num_elements} elements announced, {total} found"),
        ));
    }
    c.end("Elements")?;
    Ok(blocks)
}

/// Parses an MSH 4.1 file.
///
/// Header counts are checked with checked arithmetic, and no allocation is sized from a count
/// before the data it describes is there. Unknown sections are skipped.
///
/// # Errors
///
/// `Parse` for a malformed or truncated file, a version other than 4.1, or a node or element
/// tag ≤ 0; the offset is the byte where the error was found. `UnsupportedElementType` for an
/// element type not in A.7.
///
/// # Example
///
/// ```
/// use gridcl_io::gmsh::parse;
///
/// let text = b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
///     $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n";
/// let file = parse(text)?;
/// assert_eq!(file.node_blocks[0].tags, [1, 2, 3]);
/// assert_eq!(file.node_blocks[0].coords[1], [1.0, 0.0, 0.0]);
/// assert_eq!((file.element_blocks[0].element_type, &file.element_blocks[0].nodes[..]), (2, &[1, 2, 3][..]));
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn parse(bytes: &[u8]) -> Result<MshFile> {
    let mut c = Cursor::new(bytes);
    let (at, first) = c.token()?;
    if first != "$MeshFormat" {
        return Err(c.error_at(at, "not a Gmsh file: no $MeshFormat"));
    }
    mesh_format(&mut c)?;
    let mut file = MshFile::default();
    while !c.at_end() {
        let (at, header) = c.token()?;
        let Some(section) = header.strip_prefix('$') else {
            return Err(c.error_at(at, format!("expected a section, found {header:?}")));
        };
        c.header_end()?;
        match section {
            "PhysicalNames" => file.physical_names.extend(physical_names(&mut c)?),
            "Entities" => file.entities = Some(entities(&mut c)?),
            "PartitionedEntities" => {
                file.partitioned_entities = Some(partitioned_entities(&mut c)?)
            }
            "Periodic" => file.periodic.extend(periodic(&mut c)?),
            "Nodes" => file.node_blocks.extend(nodes(&mut c)?),
            "Elements" => file.element_blocks.extend(elements(&mut c)?),
            "MeshFormat" => return Err(c.error_at(at, "a second $MeshFormat")),
            _ if section.starts_with("End") => {
                return Err(c.error_at(at, format!("{header} without its section")));
            }
            _ => c.skip_section(section)?,
        }
    }
    Ok(file)
}

/// Reads and parses the file at `path`.
///
/// # Errors
///
/// `Io` if the file cannot be read; errors of [`parse`].
///
/// # Example
///
/// ```
/// use gridcl_device::GridError;
/// use gridcl_io::gmsh::read_file;
///
/// let missing = std::path::Path::new("no-such-file.msh");
/// assert!(matches!(read_file(missing), Err(GridError::Io(_))));
/// ```
pub fn read_file(path: &Path) -> Result<MshFile> {
    let bytes =
        std::fs::read(path).map_err(|e| GridError::Io(format!("{}: {e}", path.display())))?;
    parse(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small complete ASCII file: two triangles, names, entities, a comment section.
    const FILE: &str = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
$PhysicalNames\n2\n1 5 \"left side\"\n2 7 \"domain\"\n$EndPhysicalNames\n\
$Comments\nanything $Nodes here\n$EndComments\n\
$Entities\n1 1 1 0\n1 0 0 0 0\n1 0 0 0 0 1 0 1 5 2 1 -1\n\
1 0 0 0 1 1 0 1 7 1 1\n$EndEntities\n\
$Nodes\n2 4 1 4\n0 1 0 1\n1\n0 0 0\n2 1 1 3\n2\n3\n4\n1 0 0 0.5 0\n1 1 0 1 1\n0 1 0 0 1\n$EndNodes\n\
$Elements\n2 3 1 3\n1 1 1 1\n3 1 4\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";

    fn parse_err(text: &str) -> (u64, String) {
        match parse(text.as_bytes()) {
            Err(GridError::Parse {
                format: "Gmsh",
                offset,
                message,
            }) => (offset, message),
            other => panic!("expected a Parse error, got {other:?}"),
        }
    }

    #[test]
    fn parse_sections() {
        let f = parse(FILE.as_bytes()).unwrap();
        assert_eq!(
            f.physical_names,
            [
                PhysicalName {
                    dim: 1,
                    tag: 5,
                    name: "left side".into()
                },
                PhysicalName {
                    dim: 2,
                    tag: 7,
                    name: "domain".into()
                },
            ]
        );
        let e = f.entities.as_ref().unwrap();
        assert_eq!(e.points.len(), 1);
        assert_eq!(e.points[0].min, e.points[0].max);
        assert_eq!(e.curves[0].physical_tags, [5]);
        assert_eq!(e.curves[0].boundary, [1, -1]);
        assert_eq!(e.curves[0].max, [0.0, 1.0, 0.0]);
        assert_eq!(e.surfaces[0].physical_tags, [7]);
        assert_eq!(e.surfaces[0].boundary, [1]);
        assert!(e.volumes.is_empty());
        // Parametric coordinates of the surface block are dropped.
        assert_eq!(f.node_blocks.len(), 2);
        assert_eq!((f.node_blocks[1].dim, f.node_blocks[1].entity_tag), (2, 1));
        assert_eq!(f.node_blocks[1].tags, [2, 3, 4]);
        assert_eq!(f.node_blocks[1].coords[2], [0.0, 1.0, 0.0]);
        assert_eq!(f.element_blocks[0].element_type, 1);
        assert_eq!(f.element_blocks[1].tags, [1, 2]);
        assert_eq!(f.element_blocks[1].nodes, [1, 2, 3, 1, 3, 4]);
        assert!(f.partitioned_entities.is_none() && f.periodic.is_empty());
    }

    #[test]
    fn parse_partitioned_entities_and_periodic() {
        let text = FILE.replace(
            "$Nodes",
            "$PartitionedEntities\n2\n1\n4 2\n1 1 0 0\n\
             5 0 1 1 2 0.5 0 0 1 5\n\
             6 1 2 2 1 2 0 0 0 1 1 0 0 2 5 -5\n$EndPartitionedEntities\n\
             $Periodic\n2\n1 2 1\n16 1 0 0 1 0 1 0 0 0 0 1 0 0 0 0 1\n1\n3 2\n\
             0 4 1\n0\n0\n$EndPeriodic\n$Nodes",
        );
        let f = parse(text.as_bytes()).unwrap();
        let p = f.partitioned_entities.unwrap();
        assert_eq!(
            (p.num_partitions, &p.ghost_entities[..]),
            (2, &[(4, 2)][..])
        );
        assert_eq!(p.points.len(), 1);
        assert_eq!((p.points[0].parent_dim, p.points[0].parent_tag), (0, 1));
        assert_eq!(p.points[0].partitions, [2]);
        assert_eq!(p.points[0].entity.min, [0.5, 0.0, 0.0]);
        assert_eq!(p.points[0].entity.physical_tags, [5]);
        let curve = &p.curves[0];
        assert_eq!(
            (curve.entity.tag, curve.parent_dim, curve.parent_tag),
            (6, 1, 2)
        );
        assert_eq!(curve.partitions, [1, 2]);
        assert_eq!(curve.entity.max, [1.0, 1.0, 0.0]);
        assert_eq!(curve.entity.boundary, [5, -5]);
        assert_eq!(f.periodic.len(), 2);
        assert_eq!(
            (
                f.periodic[0].dim,
                f.periodic[0].tag,
                f.periodic[0].master_tag
            ),
            (1, 2, 1)
        );
        assert_eq!(f.periodic[0].affine.len(), 16);
        assert_eq!(f.periodic[0].nodes, [(3, 2)]);
        assert!(f.periodic[1].affine.is_empty() && f.periodic[1].nodes.is_empty());
        // The rest of the file is unchanged.
        assert_eq!(f.node_blocks, parse(FILE.as_bytes()).unwrap().node_blocks);
        let (_, message) = parse_err(&text.replace("\n1\n3 2\n", "\n1\n3 0\n"));
        assert!(message.contains("tag 0"), "{message}");
    }

    /// The corpus's partitioned and periodic files (gridcl-io/tests/fixtures/gmsh).
    #[test]
    fn parse_partitioned_and_periodic_fixtures() {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/gmsh");
        let f = read_file(&dir.join("partitioned_p1.msh")).unwrap();
        let p = f.partitioned_entities.unwrap();
        assert_eq!(p.num_partitions, 2);
        // Interface entities have a parent of higher dimension.
        assert!(p.curves.iter().any(|e| e.parent_dim == 2));
        assert!(p.surfaces.iter().all(|e| e.parent_dim == 2));
        let f = read_file(&dir.join("periodic_p1.msh")).unwrap();
        assert!(f.periodic.iter().any(|l| l.dim == 1 && !l.nodes.is_empty()));
        assert!(f.periodic.iter().all(|l| l.affine.len() == 16));
    }

    #[test]
    fn parse_error_offsets() {
        let (offset, message) = parse_err("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n");
        assert_eq!(offset, 12);
        assert!(message.contains("2.2"), "{message}");
        let (offset, _) = parse_err("$MeshFormat\n4.1 3 8\n$EndMeshFormat\n");
        assert_eq!(offset, 15);
        let bad = FILE.replace("2 1 2 2\n1 1 2 3", "2 1 2 2\n1 1 x 3");
        let (offset, message) = parse_err(&bad);
        assert_eq!(offset as usize, bad.find("1 1 x 3").unwrap() + 4);
        assert!(message.contains("\"x\""), "{message}");
        let (_, message) = parse_err(&FILE.replace("$EndNodes", "$EndNode"));
        assert!(message.contains("$EndNodes"), "{message}");
        let (offset, _) = parse_err("hello");
        assert_eq!(offset, 0);
    }

    #[test]
    fn parse_rejects_non_positive_tags() {
        for (from, to) in [
            ("2 1 1 3\n2\n3", "2 1 1 3\n0\n3"),
            ("2 1 1 3\n2\n3", "2 1 1 3\n-2\n3"),
            ("1 1 2 3\n2 1 3 4", "0 1 2 3\n2 1 3 4"),
            ("1 1 2 3\n2 1 3 4", "1 1 0 3\n2 1 3 4"),
        ] {
            let text = FILE.replace(from, to);
            assert_ne!(text, FILE);
            parse_err(&text);
        }
    }

    #[test]
    fn parse_rejects_huge_counts_without_allocating() {
        for text in [
            FILE.replace("$Nodes\n2 4 1 4", "$Nodes\n18446744073709551615 4 1 4"),
            FILE.replace("2 1 1 3\n2\n3", "2 1 1 18446744073709551615\n2\n3"),
            FILE.replace("2 1 1 3\n2\n3", "2 1 1 99999999999\n2\n3"),
            FILE.replace("2 1 2 2\n1 1 2 3", "2 1 2 4611686018427387904\n1 1 2 3"),
            FILE.replace("$PhysicalNames\n2", "$PhysicalNames\n9999999999999"),
            FILE.replace(
                "1 0 0 0 0 1 0 1 5 2 1 -1",
                "1 0 0 0 0 1 0 1 5 999999999 1 -1",
            ),
        ] {
            let (_, message) = parse_err(&text);
            assert!(message.contains("exceeds"), "{message}");
        }
    }

    #[test]
    fn parse_rejects_wrong_totals_and_types() {
        let (_, message) = parse_err(&FILE.replace("$Nodes\n2 4 1 4", "$Nodes\n2 5 1 4"));
        assert!(message.contains("5 nodes announced"), "{message}");
        let (_, message) = parse_err(&FILE.replace("$Elements\n2 3 1 3", "$Elements\n2 4 1 3"));
        assert!(message.contains("4 elements announced"), "{message}");
        let text = FILE.replace("2 1 2 2\n1 1 2 3", "2 1 20 2\n1 1 2 3");
        assert!(matches!(
            parse(text.as_bytes()),
            Err(GridError::UnsupportedElementType {
                format: "Gmsh",
                type_id: 20
            })
        ));
        let (_, message) = parse_err(&FILE.replace("2 1 2 2\n1 1 2 3", "1 1 2 2\n1 1 2 3"));
        assert!(message.contains("dimension 1"), "{message}");
    }

    /// Every prefix of the file parses without panicking; it succeeds only when it ends
    /// between sections.
    #[test]
    fn parse_truncated_files() {
        let bytes = FILE.as_bytes();
        let ends: Vec<usize> = FILE
            .match_indices("$End")
            .map(|(i, _)| i + FILE[i..].find('\n').unwrap())
            .collect();
        for len in 0..bytes.len() {
            let ok = parse(&bytes[..len]).is_ok();
            let between = ends.iter().any(|&e| len == e || len == e + 1);
            // The first section must be complete; a cut inside a section must fail.
            assert_eq!(ok, between, "prefix of {len} bytes");
        }
    }

    #[test]
    fn parse_corrupt_bytes_never_panic() {
        // Overwrite each byte with characters that break tokens or numbers.
        for (i, _) in FILE.char_indices() {
            for b in [b'x', b'-', b'9', b' ', 0xff] {
                let mut bytes = FILE.as_bytes().to_vec();
                bytes[i] = b;
                let _ = parse(&bytes);
            }
        }
    }
}
