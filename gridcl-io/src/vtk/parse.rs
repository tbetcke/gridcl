//! The VTU parser.
//!
//! A small XML reader builds the element tree up to `<AppendedData>`; the bytes after its `_`
//! are not XML and are read through the `offset` of each appended `DataArray`. Each array is
//! then decoded as `vtkXMLDataParser` does (see the module documentation of the writer for the
//! binary layout), with the header word size of `header_type` and the byte order of
//! `byte_order`.
//!
//! Every count read from the file is checked against the bytes that hold it before anything is
//! allocated from it; zlib output is read through a limit, so a block can never grow beyond
//! what its header announced.

use std::io::Read;

use flate2::read::ZlibDecoder;
use gridcl_device::{GridError, Result};

use super::{DataArray, DataValues, FORMAT, VtuFile, base64};

/// The deepest element nesting accepted (a VTU file needs six).
const MAX_DEPTH: usize = 64;

/// `Parse` at byte `offset`.
fn error(offset: usize, message: impl Into<String>) -> GridError {
    GridError::Parse {
        format: FORMAT,
        offset: offset as u64,
        message: message.into(),
    }
}

/// The first position of `pat` in `bytes` at or after `from`.
fn find(bytes: &[u8], from: usize, pat: &[u8]) -> Option<usize> {
    bytes
        .get(from..)?
        .windows(pat.len())
        .position(|w| w == pat)
        .map(|i| from + i)
}

/// One XML element.
struct Element<'a> {
    name: &'a str,
    /// The byte offset of its `<`.
    offset: usize,
    attrs: Vec<(&'a str, String)>,
    children: Vec<Element<'a>>,
    /// The byte ranges of its direct text content.
    text: Vec<(usize, usize)>,
}

impl<'a> Element<'a> {
    fn attr(&self, name: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(n, _)| *n == name)
            .map(|(_, v)| v.as_str())
    }

    /// The attribute `name`, which must be present.
    fn required(&self, name: &str) -> Result<&str> {
        self.attr(name).ok_or_else(|| {
            error(
                self.offset,
                format!("<{}> has no {name} attribute", self.name),
            )
        })
    }

    /// The attribute `name` as a count, `default` if absent.
    fn count(&self, name: &str, default: Option<usize>) -> Result<usize> {
        let text = match (self.attr(name), default) {
            (Some(t), _) => t,
            (None, Some(d)) => return Ok(d),
            (None, None) => self.required(name)?,
        };
        text.trim().parse().map_err(|_| {
            error(
                self.offset,
                format!("{name}=\"{text}\" is not a non-negative integer"),
            )
        })
    }

    fn children_named<'s>(&'s self, name: &str) -> impl Iterator<Item = &'s Element<'a>> {
        let name = name.to_owned();
        self.children.iter().filter(move |c| c.name == name)
    }

    /// The child `name`, which must occur at most once; `None` if it does not occur.
    fn optional_child(&self, name: &str) -> Result<Option<&Element<'a>>> {
        let mut it = self.children_named(name);
        let first = it.next();
        if let Some(second) = it.next() {
            return Err(error(
                second.offset,
                format!("more than one <{name}> in <{}>", self.name),
            ));
        }
        Ok(first)
    }

    /// The child `name`, which must occur exactly once.
    fn child(&self, name: &str) -> Result<&Element<'a>> {
        self.optional_child(name)?
            .ok_or_else(|| error(self.offset, format!("<{}> has no <{name}>", self.name)))
    }
}

/// The `<AppendedData>` section.
struct Appended {
    /// The offset of the first byte after the `_`.
    start: usize,
    base64: bool,
}

/// The element tree of a file.
struct Document<'a> {
    root: Element<'a>,
    appended: Option<Appended>,
}

/// Replaces the five predefined entities and numeric character references in an attribute
/// value at byte `at`.
fn unescape(raw: &[u8], at: usize) -> Result<String> {
    let s = std::str::from_utf8(raw).map_err(|_| error(at, "an attribute that is not UTF-8"))?;
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let semi = rest[amp..]
            .find(';')
            .ok_or_else(|| error(at, "an unterminated entity in an attribute"))?;
        let entity = &rest[amp + 1..amp + semi];
        let c = match entity {
            "amp" => Some('&'),
            "lt" => Some('<'),
            "gt" => Some('>'),
            "quot" => Some('"'),
            "apos" => Some('\''),
            _ => {
                let code = if let Some(hex) = entity.strip_prefix("#x") {
                    u32::from_str_radix(hex, 16).ok()
                } else if let Some(dec) = entity.strip_prefix('#') {
                    dec.parse().ok()
                } else {
                    None
                };
                code.and_then(char::from_u32)
            }
        };
        let c = c.ok_or_else(|| error(at, format!("unknown entity &{entity}; in an attribute")))?;
        out.push(c);
        rest = &rest[amp + semi + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

/// Whether `c` ends an element or attribute name.
fn ends_name(c: u8) -> bool {
    c.is_ascii_whitespace() || matches!(c, b'/' | b'>' | b'=')
}

/// The XML reader: a position in the file.
struct Xml<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Xml<'a> {
    fn skip_whitespace(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn name(&mut self) -> Result<&'a str> {
        let start = self.pos;
        while self.peek().is_some_and(|c| !ends_name(c)) {
            self.pos += 1;
        }
        if start == self.pos {
            return Err(error(start, "expected a name"));
        }
        std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| error(start, "a name that is not UTF-8"))
    }

    fn expect(&mut self, c: u8) -> Result<()> {
        if self.peek() != Some(c) {
            return Err(error(self.pos, format!("expected '{}'", c as char)));
        }
        self.pos += 1;
        Ok(())
    }

    /// A start tag from `<` on: the element, and whether it closes itself.
    fn start_tag(&mut self) -> Result<(Element<'a>, bool)> {
        let offset = self.pos;
        self.pos += 1;
        let name = self.name()?;
        let mut attrs: Vec<(&str, String)> = Vec::new();
        loop {
            self.skip_whitespace();
            match self.peek() {
                None => return Err(error(offset, format!("unterminated <{name}>"))),
                Some(b'>') => {
                    self.pos += 1;
                    break;
                }
                Some(b'/') => {
                    self.pos += 1;
                    self.expect(b'>')?;
                    let element = Element {
                        name,
                        offset,
                        attrs,
                        children: Vec::new(),
                        text: Vec::new(),
                    };
                    return Ok((element, true));
                }
                Some(_) => {
                    let at = self.pos;
                    let key = self.name()?;
                    self.skip_whitespace();
                    self.expect(b'=')?;
                    self.skip_whitespace();
                    let quote = self
                        .peek()
                        .filter(|&q| q == b'"' || q == b'\'')
                        .ok_or_else(|| error(self.pos, "expected a quoted attribute value"))?;
                    let start = self.pos + 1;
                    let end = find(self.bytes, start, &[quote])
                        .ok_or_else(|| error(at, "an unterminated attribute value"))?;
                    self.pos = end + 1;
                    if attrs.iter().any(|(k, _)| *k == key) {
                        return Err(error(at, format!("attribute {key} repeated")));
                    }
                    attrs.push((key, unescape(&self.bytes[start..end], start)?));
                }
            }
        }
        let element = Element {
            name,
            offset,
            attrs,
            children: Vec::new(),
            text: Vec::new(),
        };
        Ok((element, false))
    }

    /// Skips from `<` past `close`.
    fn skip_past(&mut self, close: &[u8], what: &str) -> Result<()> {
        let end = find(self.bytes, self.pos, close)
            .ok_or_else(|| error(self.pos, format!("an unterminated {what}")))?;
        self.pos = end + close.len();
        Ok(())
    }
}

/// Adds the finished `element` to its parent, or makes it the root.
fn attach<'a>(stack: &mut [Element<'a>], root: &mut Option<Element<'a>>, element: Element<'a>) {
    match stack.last_mut() {
        Some(parent) => parent.children.push(element),
        None => *root = Some(element),
    }
}

/// Reads the element tree, up to the `_` of `<AppendedData>` if there is one.
fn read_xml(bytes: &[u8]) -> Result<Document<'_>> {
    let mut x = Xml { bytes, pos: 0 };
    let mut stack: Vec<Element> = Vec::new();
    let mut root = None;
    let mut appended = None;
    loop {
        let start = x.pos;
        let lt = find(bytes, start, b"<").unwrap_or(bytes.len());
        if lt > start {
            match stack.last_mut() {
                Some(top) => top.text.push((start, lt)),
                None if bytes[start..lt].iter().all(u8::is_ascii_whitespace) => {}
                None => return Err(error(start, "text outside the root element")),
            }
        }
        x.pos = lt;
        if lt == bytes.len() {
            if let Some(open) = stack.last() {
                return Err(error(
                    open.offset,
                    format!("<{}> is not closed (the file ends first)", open.name),
                ));
            }
            break;
        }
        let rest = &bytes[lt..];
        if rest.starts_with(b"<?") {
            x.skip_past(b"?>", "processing instruction")?;
        } else if rest.starts_with(b"<!--") {
            x.skip_past(b"-->", "comment")?;
        } else if rest.starts_with(b"<!") {
            return Err(error(lt, "DOCTYPE and CDATA sections are not supported"));
        } else if rest.starts_with(b"</") {
            x.pos += 2;
            let name = x.name()?;
            x.skip_whitespace();
            x.expect(b'>')?;
            let open = stack
                .pop()
                .ok_or_else(|| error(lt, format!("</{name}> closes no element")))?;
            if open.name != name {
                return Err(error(
                    lt,
                    format!("</{name}> closes <{}> (at byte {})", open.name, open.offset),
                ));
            }
            attach(&mut stack, &mut root, open);
        } else {
            let (element, closed) = x.start_tag()?;
            if root.is_some() && stack.is_empty() {
                return Err(error(lt, "a second root element"));
            }
            if element.name == "AppendedData" && !closed {
                let encoding = element.required("encoding")?;
                let base64 = match encoding {
                    "raw" => false,
                    "base64" => true,
                    _ => {
                        return Err(error(
                            lt,
                            format!("AppendedData encoding=\"{encoding}\" is not supported"),
                        ));
                    }
                };
                x.skip_whitespace();
                if x.peek() != Some(b'_') {
                    return Err(error(x.pos, "expected '_' to start the appended data"));
                }
                appended = Some(Appended {
                    start: x.pos + 1,
                    base64,
                });
                // The rest is binary: close every open element here.
                attach(&mut stack, &mut root, element);
                while let Some(open) = stack.pop() {
                    attach(&mut stack, &mut root, open);
                }
                break;
            }
            if closed {
                attach(&mut stack, &mut root, element);
            } else {
                if stack.len() == MAX_DEPTH {
                    return Err(error(
                        lt,
                        format!("elements nested deeper than {MAX_DEPTH}"),
                    ));
                }
                stack.push(element);
            }
        }
    }
    let root = root.ok_or_else(|| error(0, "no root element"))?;
    Ok(Document { root, appended })
}

/// The value type of a `DataArray`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    F32,
    F64,
    I32,
    U32,
    I64,
    U8,
}

impl Kind {
    fn from_name(name: &str) -> Option<Kind> {
        Some(match name {
            "Float32" => Kind::F32,
            "Float64" => Kind::F64,
            "Int32" => Kind::I32,
            "UInt32" => Kind::U32,
            "Int64" => Kind::I64,
            "UInt8" => Kind::U8,
            _ => return None,
        })
    }

    fn size(self) -> usize {
        match self {
            Kind::F32 | Kind::I32 | Kind::U32 => 4,
            Kind::F64 | Kind::I64 => 8,
            Kind::U8 => 1,
        }
    }
}

/// `bytes` as `N`-byte values in the given byte order.
fn values<const N: usize, T>(
    bytes: &[u8],
    big_endian: bool,
    le: fn([u8; N]) -> T,
    be: fn([u8; N]) -> T,
) -> Vec<T> {
    let from = if big_endian { be } else { le };
    bytes.as_chunks::<N>().0.iter().map(|&c| from(c)).collect()
}

/// The values of `kind` stored in `bytes`, a whole number of values.
fn from_bytes(kind: Kind, bytes: &[u8], big_endian: bool) -> DataValues {
    match kind {
        Kind::F32 => DataValues::F32(values(
            bytes,
            big_endian,
            f32::from_le_bytes,
            f32::from_be_bytes,
        )),
        Kind::F64 => DataValues::F64(values(
            bytes,
            big_endian,
            f64::from_le_bytes,
            f64::from_be_bytes,
        )),
        Kind::I32 => DataValues::I32(values(
            bytes,
            big_endian,
            i32::from_le_bytes,
            i32::from_be_bytes,
        )),
        Kind::U32 => DataValues::U32(values(
            bytes,
            big_endian,
            u32::from_le_bytes,
            u32::from_be_bytes,
        )),
        Kind::I64 => DataValues::I64(values(
            bytes,
            big_endian,
            i64::from_le_bytes,
            i64::from_be_bytes,
        )),
        Kind::U8 => DataValues::U8(bytes.to_vec()),
    }
}

/// The ASCII tokens of `ranges` parsed as `T`; exactly `expected` of them.
fn ascii<T: std::str::FromStr>(
    bytes: &[u8],
    ranges: &[(usize, usize)],
    expected: usize,
    at: usize,
) -> Result<Vec<T>> {
    let mut out = Vec::new();
    for &(start, end) in ranges {
        let mut pos = start;
        while pos < end {
            while pos < end && bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            let token_start = pos;
            while pos < end && !bytes[pos].is_ascii_whitespace() {
                pos += 1;
            }
            if token_start == pos {
                break;
            }
            if out.len() == expected {
                return Err(error(
                    token_start,
                    format!("more than the {expected} values expected"),
                ));
            }
            let token = &bytes[token_start..pos];
            let value = std::str::from_utf8(token)
                .ok()
                .and_then(|t| t.parse().ok())
                .ok_or_else(|| {
                    error(
                        token_start,
                        format!(
                            "{:?} is not a value of the array's type",
                            String::from_utf8_lossy(token)
                        ),
                    )
                })?;
            out.push(value);
        }
    }
    if out.len() != expected {
        return Err(error(
            at,
            format!("{} values where {expected} are expected", out.len()),
        ));
    }
    Ok(out)
}

/// How the binary arrays of a file are stored.
struct Layout<'a> {
    bytes: &'a [u8],
    big_endian: bool,
    /// The header word size: 4 (`UInt32`) or 8 (`UInt64`).
    word: usize,
    zlib: bool,
    appended: Option<&'a Appended>,
}

impl Layout<'_> {
    /// The `k`-th header word of `s`, if `s` holds it.
    fn word(&self, s: &[u8], k: usize) -> Option<u64> {
        let w = s.get(k * self.word..(k + 1) * self.word)?;
        Some(match (self.word, self.big_endian) {
            (4, false) => u64::from(u32::from_le_bytes(w.try_into().ok()?)),
            (4, true) => u64::from(u32::from_be_bytes(w.try_into().ok()?)),
            (_, false) => u64::from_le_bytes(w.try_into().ok()?),
            (_, true) => u64::from_be_bytes(w.try_into().ok()?),
        })
    }

    /// The `k`-th header word of `s` as a size; an error at `at` if `s` ends first.
    fn size(&self, s: &[u8], k: usize, at: usize) -> Result<usize> {
        let w = self
            .word(s, k)
            .ok_or_else(|| error(at, "the array's header is truncated"))?;
        usize::try_from(w).map_err(|_| error(at, format!("header word {w} is too large")))
    }

    /// The number of bytes the header of the array at the start of `s` takes.
    fn header_len(&self, s: &[u8], at: usize) -> Result<usize> {
        if !self.zlib {
            return Ok(self.word);
        }
        let blocks = self.size(s, 0, at)?;
        blocks
            .checked_add(3)
            .and_then(|n| n.checked_mul(self.word))
            .ok_or_else(|| error(at, format!("{blocks} compressed blocks")))
    }

    /// The byte count of the array data stored after the header `h` (compressed if zlib).
    fn stored_len(&self, h: &[u8], at: usize) -> Result<usize> {
        if !self.zlib {
            return self.size(h, 0, at);
        }
        let blocks = self.size(h, 0, at)?;
        let mut total = 0usize;
        for k in 0..blocks {
            total = total
                .checked_add(self.size(h, 3 + k, at)?)
                .ok_or_else(|| error(at, "the compressed block sizes overflow"))?;
        }
        Ok(total)
    }

    /// The `expected` data bytes of the array whose header starts `s` (at byte `at` of the
    /// file). Bytes after the array are ignored unless `exact`.
    fn unpack(&self, s: &[u8], expected: usize, exact: bool, at: usize) -> Result<Vec<u8>> {
        let header_len = self.header_len(s, at)?;
        if s.len() < header_len {
            return Err(error(at, "the array's header is truncated"));
        }
        let stored = self.stored_len(&s[..header_len], at)?;
        let end = header_len
            .checked_add(stored)
            .filter(|&end| end <= s.len())
            .ok_or_else(|| {
                error(
                    at,
                    format!(
                        "the array holds {stored} bytes but only {} follow its header",
                        s.len() - header_len
                    ),
                )
            })?;
        if exact && end != s.len() {
            return Err(error(
                at,
                format!("{} bytes after the array", s.len() - end),
            ));
        }
        let data = &s[header_len..end];
        if !self.zlib {
            if stored != expected {
                return Err(error(
                    at,
                    format!("the array holds {stored} bytes where {expected} are expected"),
                ));
            }
            return Ok(data.to_vec());
        }
        let blocks = self.size(s, 0, at)?;
        let block_size = self.size(s, 1, at)?;
        let last = self.size(s, 2, at)?;
        if last > block_size {
            return Err(error(
                at,
                format!("last block of {last} bytes exceeds the block size {block_size}"),
            ));
        }
        let raw = match blocks {
            0 => Some(0),
            _ => (blocks - 1)
                .checked_mul(block_size)
                .and_then(|n| n.checked_add(if last == 0 { block_size } else { last })),
        };
        if raw != Some(expected) {
            return Err(error(
                at,
                format!(
                    "{blocks} compressed blocks of {block_size} bytes (the last {last}) do not \
                     hold the {expected} bytes expected"
                ),
            ));
        }
        let mut out = Vec::new();
        let mut from = 0;
        for k in 0..blocks {
            let size = self.size(s, 3 + k, at)?;
            let want = if k + 1 == blocks && last != 0 {
                last
            } else {
                block_size
            };
            let before = out.len();
            ZlibDecoder::new(&data[from..from + size])
                .take(want as u64)
                .read_to_end(&mut out)
                .map_err(|e| error(at, format!("zlib block {k}: {e}")))?;
            if out.len() - before != want {
                return Err(error(
                    at,
                    format!(
                        "zlib block {k} holds {} bytes where {want} are expected",
                        out.len() - before
                    ),
                ));
            }
            from += size;
        }
        Ok(out)
    }

    /// `nbytes` decoded from the base64 text `text` from character `from` on, which starts a
    /// stream of at least `nbytes` bytes.
    fn base64_bytes(&self, text: &[u8], from: usize, nbytes: usize, at: usize) -> Result<Vec<u8>> {
        let end = nbytes
            .div_ceil(3)
            .checked_mul(4)
            .and_then(|n| n.checked_add(from))
            .filter(|&end| end <= text.len())
            .ok_or_else(|| error(at, "the appended base64 data is truncated"))?;
        let mut d =
            base64::decode(&text[from..end]).map_err(|e| error(at + from + e.offset, e.message))?;
        if d.len() < nbytes {
            return Err(error(at, "the appended base64 data is truncated"));
        }
        d.truncate(nbytes);
        Ok(d)
    }

    /// The `expected` data bytes of the array at `offset` of appended base64 data: the header
    /// and the data form one base64 stream, or two under zlib.
    fn unpack_appended_base64(&self, text: &[u8], expected: usize, at: usize) -> Result<Vec<u8>> {
        // The fixed part of the header: the byte count, or the three words before the block
        // sizes.
        let fixed = if self.zlib { 3 * self.word } else { self.word };
        let header_len = self.header_len(&self.base64_bytes(text, 0, fixed, at)?, at)?;
        let header = self.base64_bytes(text, 0, header_len, at)?;
        let stored = self.stored_len(&header, at)?;
        let whole = if self.zlib {
            let data_from = header_len.div_ceil(3) * 4;
            let data = self.base64_bytes(text, data_from, stored, at)?;
            [header, data].concat()
        } else {
            let len = header_len
                .checked_add(stored)
                .ok_or_else(|| error(at, format!("the array holds {stored} bytes")))?;
            self.base64_bytes(text, 0, len, at)?
        };
        self.unpack(&whole, expected, true, at)
    }

    /// The values of the `DataArray` `el`: `expected` of them.
    fn read(&self, el: &Element, expected: usize) -> Result<DataValues> {
        let type_name = el.required("type")?;
        let kind = Kind::from_name(type_name).ok_or_else(|| {
            error(
                el.offset,
                format!("DataArray type=\"{type_name}\" is not supported"),
            )
        })?;
        let expected_bytes = expected
            .checked_mul(kind.size())
            .ok_or_else(|| error(el.offset, format!("{expected} values do not fit in memory")))?;
        let bytes = match el.attr("format").unwrap_or("ascii") {
            "ascii" => {
                let (b, r, at) = (self.bytes, &el.text[..], el.offset);
                return Ok(match kind {
                    Kind::F32 => DataValues::F32(ascii(b, r, expected, at)?),
                    Kind::F64 => DataValues::F64(ascii(b, r, expected, at)?),
                    Kind::I32 => DataValues::I32(ascii(b, r, expected, at)?),
                    Kind::U32 => DataValues::U32(ascii(b, r, expected, at)?),
                    Kind::I64 => DataValues::I64(ascii(b, r, expected, at)?),
                    Kind::U8 => DataValues::U8(ascii(b, r, expected, at)?),
                });
            }
            "binary" => {
                let mut text = Vec::new();
                for &(start, end) in &el.text {
                    text.extend_from_slice(&self.bytes[start..end]);
                }
                let decoded = base64::decode(&text).map_err(|e| {
                    // The file offset of the e.offset-th byte of the joined text.
                    let mut k = e.offset;
                    let mut at = el.offset;
                    for &(start, end) in &el.text {
                        if k < end - start {
                            at = start + k;
                            break;
                        }
                        k -= end - start;
                    }
                    error(at, e.message)
                })?;
                self.unpack(&decoded, expected_bytes, true, el.offset)?
            }
            "appended" => {
                let appended = self.appended.ok_or_else(|| {
                    error(el.offset, "an appended DataArray without <AppendedData>")
                })?;
                let offset = el.count("offset", None)?;
                let at = appended
                    .start
                    .checked_add(offset)
                    .filter(|&at| at <= self.bytes.len())
                    .ok_or_else(|| {
                        error(
                            el.offset,
                            format!("offset=\"{offset}\" is past the end of the file"),
                        )
                    })?;
                let s = &self.bytes[at..];
                if appended.base64 {
                    self.unpack_appended_base64(s, expected_bytes, at)?
                } else {
                    self.unpack(s, expected_bytes, false, at)?
                }
            }
            other => {
                return Err(error(
                    el.offset,
                    format!("DataArray format=\"{other}\" is not supported"),
                ));
            }
        };
        Ok(from_bytes(kind, &bytes, self.big_endian))
    }

    /// The `DataArray` children of a `PointData` or `CellData` element, `entries` per
    /// component.
    fn data_arrays(&self, section: Option<&Element>, entries: usize) -> Result<Vec<DataArray>> {
        let Some(section) = section else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        for el in section.children_named("DataArray") {
            let components = el.count("NumberOfComponents", Some(1))?;
            if components == 0 {
                return Err(error(el.offset, "NumberOfComponents=\"0\""));
            }
            let expected = entries
                .checked_mul(components)
                .ok_or_else(|| error(el.offset, format!("{entries} × {components} values")))?;
            out.push(DataArray {
                name: el.required("Name")?.to_string(),
                components,
                values: self.read(el, expected)?,
            });
        }
        Ok(out)
    }
}

/// Integer `values` as i64; an error at `at` for floating-point arrays.
fn integers(values: DataValues, what: &str, at: usize) -> Result<Vec<i64>> {
    Ok(match values {
        DataValues::I32(v) => v.into_iter().map(i64::from).collect(),
        DataValues::U32(v) => v.into_iter().map(i64::from).collect(),
        DataValues::I64(v) => v,
        DataValues::U8(v) => v.into_iter().map(i64::from).collect(),
        DataValues::F32(_) | DataValues::F64(_) => {
            return Err(error(at, format!("{what} has a floating-point type")));
        }
    })
}

/// The `DataArray` named `name` among the children of `cells`.
fn cells_array<'e, 'a>(cells: &'e Element<'a>, name: &str) -> Result<&'e Element<'a>> {
    let mut it = cells
        .children_named("DataArray")
        .filter(|el| el.attr("Name") == Some(name));
    let first = it
        .next()
        .ok_or_else(|| error(cells.offset, format!("<Cells> has no {name} array")))?;
    if let Some(second) = it.next() {
        return Err(error(second.offset, format!("a second {name} array")));
    }
    if first.count("NumberOfComponents", Some(1))? != 1 {
        return Err(error(
            first.offset,
            format!("{name} must have one component"),
        ));
    }
    Ok(first)
}

/// Parses a VTU file: `UnstructuredGrid` with ASCII, inline base64 or appended (raw or base64)
/// data, with or without zlib, and UInt32 or UInt64 headers.
///
/// The arrays come back as stored. Every `version` is accepted; the caller interprets the node
/// order (VTK renumbers Lagrange hexahedra in files before version 2.1). The file has one
/// `Piece` with `NumberOfPoints` and `NumberOfCells`, `Points` (one `DataArray` with three
/// components) and `Cells` (`connectivity`, `offsets` and `types`; other arrays there are
/// ignored), and optionally `PointData` and `CellData`. Data arrays have type `Float32`,
/// `Float64`, `Int32`, `UInt32`, `Int64` or `UInt8`; `connectivity` and `offsets` may have any
/// of the integer types, and `types` any whose values fit in a `UInt8`. `byte_order` defaults
/// to `LittleEndian`, `header_type` to `UInt32`, `format` to `ascii` and `NumberOfComponents`
/// to 1. The only `compressor` read is `vtkZLibDataCompressor`, with any block size.
/// `InformationKey` children of a `DataArray` and other unknown elements are skipped.
///
/// The result always fits together as [`VtuFile::write_to`] requires, and every connectivity
/// entry is a point index.
///
/// # Errors
///
/// `Parse { format: "VTU", .. }` for malformed, truncated or unsupported content, with the
/// byte offset of the element, value or binary header at fault: an array holding a number of
/// values other than its `Piece` implies, offsets that decrease or do not end at the length of
/// `connectivity`, or a connectivity entry that is not a point index.
///
/// # Example
///
/// ```
/// use gridcl_io::vtk::{Binary, DataValues, VtuFile, parse_vtu};
///
/// let file = VtuFile {
///     points: DataValues::F64(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
///     connectivity: vec![0, 1, 2],
///     offsets: vec![3],
///     types: vec![5],
///     point_data: vec![],
///     cell_data: vec![],
/// };
/// let mut bytes = Vec::new();
/// file.write_to(&mut bytes, Binary::Appended, true)?;
/// assert_eq!(parse_vtu(&bytes)?, file);
///
/// // VTK's ASCII format, with the attributes VTK leaves to their defaults.
/// let ascii = br#"<VTKFile type="UnstructuredGrid" version="0.1">
///   <UnstructuredGrid><Piece NumberOfPoints="2" NumberOfCells="1">
///     <Points><DataArray type="Float32" NumberOfComponents="3">0 0 0 1 0 0</DataArray></Points>
///     <Cells>
///       <DataArray type="Int32" Name="connectivity">0 1</DataArray>
///       <DataArray type="Int32" Name="offsets">2</DataArray>
///       <DataArray type="UInt8" Name="types">3</DataArray>
///     </Cells>
///   </Piece></UnstructuredGrid>
/// </VTKFile>"#;
/// let line = parse_vtu(ascii)?;
/// assert_eq!(line.points, DataValues::F32(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
/// assert_eq!((line.connectivity, line.types), (vec![0, 1], vec![3]));
///
/// // Truncated files are rejected.
/// assert!(parse_vtu(&bytes[..bytes.len() / 2]).is_err());
/// # Ok::<(), gridcl_device::GridError>(())
/// ```
pub fn parse_vtu(bytes: &[u8]) -> Result<VtuFile> {
    let doc = read_xml(bytes)?;
    let root = &doc.root;
    if root.name != "VTKFile" {
        return Err(error(
            root.offset,
            format!("the root is <{}>, not <VTKFile>", root.name),
        ));
    }
    let file_type = root.required("type")?;
    if file_type != "UnstructuredGrid" {
        return Err(error(
            root.offset,
            format!("type=\"{file_type}\" is not UnstructuredGrid"),
        ));
    }
    let big_endian = match root.attr("byte_order").unwrap_or("LittleEndian") {
        "LittleEndian" => false,
        "BigEndian" => true,
        other => return Err(error(root.offset, format!("byte_order=\"{other}\""))),
    };
    let word = match root.attr("header_type").unwrap_or("UInt32") {
        "UInt32" => 4,
        "UInt64" => 8,
        other => return Err(error(root.offset, format!("header_type=\"{other}\""))),
    };
    let zlib = match root.attr("compressor") {
        None => false,
        Some("vtkZLibDataCompressor") => true,
        Some(other) => {
            return Err(error(
                root.offset,
                format!("compressor=\"{other}\" is not supported"),
            ));
        }
    };
    let layout = Layout {
        bytes,
        big_endian,
        word,
        zlib,
        appended: doc.appended.as_ref(),
    };
    let piece = root.child("UnstructuredGrid")?.child("Piece")?;
    let num_points = piece.count("NumberOfPoints", None)?;
    let num_cells = piece.count("NumberOfCells", None)?;

    let points_el = piece.child("Points")?;
    let points_array = points_el.child("DataArray")?;
    if points_array.count("NumberOfComponents", Some(1))? != 3 {
        return Err(error(
            points_array.offset,
            "points must have three components",
        ));
    }
    let point_values = num_points
        .checked_mul(3)
        .ok_or_else(|| error(piece.offset, format!("NumberOfPoints=\"{num_points}\"")))?;
    let points = layout.read(points_array, point_values)?;

    let cells = piece.child("Cells")?;
    let offsets_el = cells_array(cells, "offsets")?;
    let offsets = integers(
        layout.read(offsets_el, num_cells)?,
        "offsets",
        offsets_el.offset,
    )?;
    let mut last = 0i64;
    for &o in &offsets {
        if o < last {
            return Err(error(
                offsets_el.offset,
                format!("offset {o} is below the one before it, {last}"),
            ));
        }
        last = o;
    }
    let types_el = cells_array(cells, "types")?;
    let types = integers(layout.read(types_el, num_cells)?, "types", types_el.offset)?
        .into_iter()
        .map(|t| u8::try_from(t).map_err(|_| error(types_el.offset, format!("cell type {t}"))))
        .collect::<Result<Vec<u8>>>()?;
    let connectivity_el = cells_array(cells, "connectivity")?;
    let num_entries = usize::try_from(last)
        .map_err(|_| error(offsets_el.offset, format!("the last offset {last}")))?;
    let connectivity = integers(
        layout.read(connectivity_el, num_entries)?,
        "connectivity",
        connectivity_el.offset,
    )?;
    if let Some(&c) = connectivity
        .iter()
        .find(|&&c| usize::try_from(c).map_or(true, |c| c >= num_points))
    {
        return Err(error(
            connectivity_el.offset,
            format!("connectivity entry {c} is not one of the {num_points} points"),
        ));
    }

    let point_data = layout.data_arrays(piece.optional_child("PointData")?, num_points)?;
    let cell_data = layout.data_arrays(piece.optional_child("CellData")?, num_cells)?;
    Ok(VtuFile {
        points,
        connectivity,
        offsets,
        types,
        point_data,
        cell_data,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use flate2::Compression;
    use flate2::write::ZlibEncoder;
    use proptest::prelude::*;

    use super::super::write::escape;
    use super::super::{Binary, base64};
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
            point_data: vec![
                DataArray {
                    name: "u".into(),
                    components: 1,
                    values: DataValues::F32(vec![0.0, -1.5, 2.25, 3e-40]),
                },
                DataArray {
                    name: "a&b <c>".into(),
                    components: 2,
                    values: DataValues::I64(vec![i64::MIN, -1, 0, 1, 2, 3, 4, i64::MAX]),
                },
            ],
            cell_data: vec![
                DataArray {
                    name: "id".into(),
                    components: 3,
                    values: DataValues::U32(vec![1, 2, 3, 4, 5, u32::MAX]),
                },
                DataArray {
                    name: "rank".into(),
                    components: 1,
                    values: DataValues::I32(vec![i32::MIN, 7]),
                },
                DataArray {
                    name: "flag".into(),
                    components: 1,
                    values: DataValues::U8(vec![0, 255]),
                },
            ],
        }
    }

    fn write(file: &VtuFile, binary: Binary, zlib: bool) -> Vec<u8> {
        let mut bytes = Vec::new();
        file.write_to(&mut bytes, binary, zlib).unwrap();
        bytes
    }

    /// Where a test-written file stores its arrays.
    #[derive(Clone, Copy, Debug)]
    enum Format {
        Ascii,
        Inline,
        AppendedRaw,
        AppendedBase64,
    }

    /// A layout the writer does not produce, written by [`write_layout`].
    #[derive(Clone, Copy, Debug)]
    struct Layout {
        format: Format,
        /// The header word size, 4 or 8.
        word: usize,
        zlib: bool,
        big_endian: bool,
    }

    /// Every layout: 4 formats × 2 header types × zlib × byte order.
    fn layouts() -> Vec<Layout> {
        let mut out = Vec::new();
        for format in [
            Format::Ascii,
            Format::Inline,
            Format::AppendedRaw,
            Format::AppendedBase64,
        ] {
            for word in [4, 8] {
                for zlib in [false, true] {
                    for big_endian in [false, true] {
                        out.push(Layout {
                            format,
                            word,
                            zlib,
                            big_endian,
                        });
                    }
                }
            }
        }
        out
    }

    /// The zlib block size of test-written files, small so arrays span several blocks.
    const TEST_BLOCK: usize = 16;

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

    macro_rules! each_value {
        ($v:expr, $x:ident => $body:expr) => {
            match $v {
                DataValues::F32(xs) => xs.iter().for_each(|&$x| $body),
                DataValues::F64(xs) => xs.iter().for_each(|&$x| $body),
                DataValues::I32(xs) => xs.iter().for_each(|&$x| $body),
                DataValues::U32(xs) => xs.iter().for_each(|&$x| $body),
                DataValues::I64(xs) => xs.iter().for_each(|&$x| $body),
                DataValues::U8(xs) => xs.iter().for_each(|&$x| $body),
            }
        };
    }

    fn value_bytes(v: &DataValues, big_endian: bool) -> Vec<u8> {
        let mut out = Vec::new();
        each_value!(v, x => out.extend_from_slice(&if big_endian {
            x.to_be_bytes().to_vec()
        } else {
            x.to_le_bytes().to_vec()
        }));
        out
    }

    fn ascii_text(v: &DataValues) -> String {
        let mut out = Vec::new();
        each_value!(v, x => out.push(x.to_string()));
        out.join(" ")
    }

    /// The header and the data of `raw` stored in `layout` (see the writer's module
    /// documentation).
    fn stored(raw: &[u8], layout: Layout) -> (Vec<u8>, Vec<u8>) {
        let word = |x: usize, out: &mut Vec<u8>| match (layout.word, layout.big_endian) {
            (4, false) => out.extend_from_slice(&(x as u32).to_le_bytes()),
            (4, true) => out.extend_from_slice(&(x as u32).to_be_bytes()),
            (_, false) => out.extend_from_slice(&(x as u64).to_le_bytes()),
            (_, true) => out.extend_from_slice(&(x as u64).to_be_bytes()),
        };
        let mut header = Vec::new();
        if !layout.zlib {
            word(raw.len(), &mut header);
            return (header, raw.to_vec());
        }
        word(raw.len().div_ceil(TEST_BLOCK), &mut header);
        word(TEST_BLOCK, &mut header);
        word(raw.len() % TEST_BLOCK, &mut header);
        let mut data = Vec::new();
        for block in raw.chunks(TEST_BLOCK) {
            let mut z = ZlibEncoder::new(Vec::new(), Compression::fast());
            z.write_all(block).unwrap();
            let compressed = z.finish().unwrap();
            word(compressed.len(), &mut header);
            data.extend_from_slice(&compressed);
        }
        (header, data)
    }

    /// The base64 text of an array: one stream, or header and data separately under zlib.
    fn base64_text(header: &[u8], data: &[u8], zlib: bool) -> Vec<u8> {
        let mut text = Vec::new();
        if zlib {
            base64::encode_into(header, &mut text);
            base64::encode_into(data, &mut text);
        } else {
            base64::encode_into(&[header, data].concat(), &mut text);
        }
        text
    }

    /// Writes `file` in `layout`, the way VTK might: comments, `InformationKey` children and
    /// `Range` attributes, which the parser skips.
    fn write_layout(file: &VtuFile, layout: Layout) -> Vec<u8> {
        let mut xml = String::new();
        let mut appended: Vec<u8> = Vec::new();
        let mut array = |xml: &mut String, name: &str, components: usize, v: &DataValues| {
            let open = format!(
                "<DataArray type=\"{}\" Name=\"{}\" NumberOfComponents=\"{components}\" \
                 RangeMin=\"0\" format=",
                type_name(v),
                escape(name)
            );
            let (header, data) = stored(&value_bytes(v, layout.big_endian), layout);
            match layout.format {
                Format::Ascii => {
                    xml.push_str(&format!(
                        "{open}\"ascii\">\n{}\n<InformationKey name=\"L2_NORM_RANGE\">\
                         <Value index=\"0\">\n0\n</Value></InformationKey>\n</DataArray>\n",
                        ascii_text(v)
                    ));
                }
                Format::Inline => {
                    let text = base64_text(&header, &data, layout.zlib);
                    xml.push_str(&format!(
                        "{open}'binary'>\n {} \n</DataArray>\n",
                        String::from_utf8(text).unwrap()
                    ));
                }
                Format::AppendedRaw => {
                    xml.push_str(&format!(
                        "{open}\"appended\" offset=\"{}\"/>\n",
                        appended.len()
                    ));
                    appended.extend_from_slice(&header);
                    appended.extend_from_slice(&data);
                }
                Format::AppendedBase64 => {
                    xml.push_str(&format!(
                        "{open}\"appended\" offset=\"{}\" />\n",
                        appended.len()
                    ));
                    appended.extend_from_slice(&base64_text(&header, &data, layout.zlib));
                }
            }
        };
        xml.push_str("<?xml version=\"1.0\"?>\n<!-- written by a test -->\n");
        xml.push_str(&format!(
            "<VTKFile type=\"UnstructuredGrid\" version=\"0.1\" byte_order=\"{}\" \
             header_type=\"UInt{}\"{}>\n<UnstructuredGrid>\n",
            if layout.big_endian {
                "BigEndian"
            } else {
                "LittleEndian"
            },
            layout.word * 8,
            if layout.zlib {
                " compressor=\"vtkZLibDataCompressor\""
            } else {
                ""
            }
        ));
        let points = match &file.points {
            DataValues::F32(x) => x.len() / 3,
            DataValues::F64(x) => x.len() / 3,
            _ => unreachable!(),
        };
        xml.push_str(&format!(
            "<Piece NumberOfPoints = '{points}' NumberOfCells=\"{}\">\n",
            file.types.len()
        ));
        // Cells first and Points last: the order of the sections does not matter.
        xml.push_str("<Cells>\n");
        array(&mut xml, "types", 1, &DataValues::U8(file.types.clone()));
        array(
            &mut xml,
            "connectivity",
            1,
            &DataValues::I64(file.connectivity.clone()),
        );
        array(
            &mut xml,
            "offsets",
            1,
            &DataValues::I64(file.offsets.clone()),
        );
        xml.push_str("</Cells>\n<PointData Scalars=\"u\">\n");
        for a in &file.point_data {
            array(&mut xml, &a.name, a.components, &a.values);
        }
        xml.push_str("</PointData>\n<CellData>\n");
        for a in &file.cell_data {
            array(&mut xml, &a.name, a.components, &a.values);
        }
        xml.push_str("</CellData>\n<Points>\n");
        array(&mut xml, "Points", 3, &file.points);
        xml.push_str("</Points>\n</Piece>\n</UnstructuredGrid>\n");
        let mut bytes = xml.into_bytes();
        match layout.format {
            Format::AppendedRaw | Format::AppendedBase64 => {
                let encoding = match layout.format {
                    Format::AppendedRaw => "raw",
                    _ => "base64",
                };
                bytes.extend_from_slice(
                    format!("<AppendedData encoding=\"{encoding}\">\n  _").as_bytes(),
                );
                bytes.extend_from_slice(&appended);
                bytes.extend_from_slice(b"\n</AppendedData>\n");
            }
            Format::Ascii | Format::Inline => {}
        }
        bytes.extend_from_slice(b"</VTKFile>\n");
        bytes
    }

    /// Every way of writing `file`: the writer's four and the test layouts.
    fn encodings(file: &VtuFile) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        for binary in [Binary::Appended, Binary::Base64] {
            for zlib in [false, true] {
                out.push((
                    format!("writer {binary:?} zlib {zlib}"),
                    write(file, binary, zlib),
                ));
            }
        }
        for layout in layouts() {
            out.push((format!("{layout:?}"), write_layout(file, layout)));
        }
        out
    }

    fn assert_parse_error(result: Result<VtuFile>, context: &str) -> (u64, String) {
        match result {
            Err(GridError::Parse {
                format: "VTU",
                offset,
                message,
            }) => (offset, message),
            other => panic!("{context}: expected a VTU parse error, got {other:?}"),
        }
    }

    #[test]
    fn parse_reads_every_encoding() {
        for (what, bytes) in encodings(&square()) {
            let parsed = parse_vtu(&bytes).unwrap_or_else(|e| panic!("{what}: {e}"));
            assert_eq!(parsed, square(), "{what}");
        }
    }

    #[test]
    fn parse_reads_empty_files() {
        let empty = VtuFile {
            points: DataValues::F32(vec![]),
            connectivity: vec![],
            offsets: vec![],
            types: vec![],
            point_data: vec![],
            cell_data: vec![DataArray {
                name: "none".into(),
                components: 2,
                values: DataValues::F64(vec![]),
            }],
        };
        for (what, bytes) in encodings(&empty) {
            assert_eq!(parse_vtu(&bytes).unwrap(), empty, "{what}");
        }
    }

    #[test]
    fn parse_reads_large_zlib_arrays() {
        // Several of the writer's 32768-byte blocks, the last partial.
        let mut file = square();
        file.cell_data.push(DataArray {
            name: "big".into(),
            components: 5000,
            values: DataValues::F64((0..10000).map(|i| f64::from(i).sqrt()).collect()),
        });
        for (what, bytes) in encodings(&file) {
            assert_eq!(parse_vtu(&bytes).unwrap(), file, "{what}");
        }
    }

    #[test]
    fn parse_reads_vtk_ascii_layout() {
        // As VTK 9.7 writes ASCII files: version 0.1, UInt32 headers, a compressor that does
        // not apply to ASCII arrays, Range attributes and InformationKey children.
        let text = br#"<?xml version="1.0"?>
<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian" header_type="UInt32" compressor="vtkZLibDataCompressor">
  <UnstructuredGrid>
    <Piece NumberOfPoints="3" NumberOfCells="1">
      <PointData>
      </PointData>
      <CellData>
      </CellData>
      <Points>
        <DataArray type="Float64" Name="Points" NumberOfComponents="3" format="ascii" RangeMin="0" RangeMax="1">
          0 0 0 1 0 0
          0.5 0.25 1e-3
          <InformationKey name="L2_NORM_RANGE" location="vtkDataArray" length="2">
            <Value index="0">
              0
            </Value>
          </InformationKey>
        </DataArray>
      </Points>
      <Cells>
        <DataArray type="Int64" Name="connectivity" format="ascii" RangeMin="0" RangeMax="2">
          0 1 2
        </DataArray>
        <DataArray type="Int64" Name="offsets" format="ascii" RangeMin="3" RangeMax="3">
          3
        </DataArray>
        <DataArray type="UInt8" Name="types" format="ascii" RangeMin="69" RangeMax="69">
          69
        </DataArray>
      </Cells>
    </Piece>
  </UnstructuredGrid>
</VTKFile>
"#;
        let file = parse_vtu(text).unwrap();
        assert_eq!(
            file.points,
            DataValues::F64(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, 0.25, 1e-3])
        );
        assert_eq!(
            (file.connectivity, file.offsets, file.types),
            (vec![0, 1, 2], vec![3], vec![69])
        );
        assert!(file.point_data.is_empty() && file.cell_data.is_empty());
    }

    #[test]
    fn parse_reads_escaped_names() {
        let mut file = square();
        let names = ["a&b<c>d\"e'f\tg\nh\ri", "é ü", ""];
        for (a, name) in file.point_data.iter_mut().zip(names) {
            a.name = name.into();
        }
        file.cell_data[0].name = names[0].into();
        for (what, bytes) in encodings(&file) {
            assert_eq!(parse_vtu(&bytes).unwrap(), file, "{what}");
        }
        // Hexadecimal references, and the other quote character.
        let bytes = String::from_utf8(write(&square(), Binary::Base64, false))
            .unwrap()
            .replace("Name=\"u\"", "Name='&#x41;&#66;\"'");
        let parsed = parse_vtu(bytes.as_bytes()).unwrap();
        assert_eq!(parsed.point_data[0].name, "AB\"");
    }

    #[test]
    fn parse_accepts_any_integer_type_for_cells() {
        let text = String::from_utf8(write_layout(
            &square(),
            Layout {
                format: Format::Ascii,
                word: 4,
                zlib: false,
                big_endian: false,
            },
        ))
        .unwrap();
        for (connectivity, offsets, types) in
            [("Int32", "UInt32", "Int64"), ("UInt8", "UInt8", "Int32")]
        {
            let edited = text
                .replacen(
                    "type=\"Int64\" Name=\"connectivity\"",
                    &format!("type=\"{connectivity}\" Name=\"connectivity\""),
                    1,
                )
                .replacen(
                    "type=\"Int64\" Name=\"offsets\"",
                    &format!("type=\"{offsets}\" Name=\"offsets\""),
                    1,
                )
                .replacen(
                    "type=\"UInt8\" Name=\"types\"",
                    &format!("type=\"{types}\" Name=\"types\""),
                    1,
                );
            assert_eq!(
                parse_vtu(edited.as_bytes()).unwrap(),
                square(),
                "{connectivity}"
            );
        }
    }

    /// Every prefix of every encoding is rejected with `Parse`, or (for appended files cut
    /// after their data) read in full; none panics.
    #[test]
    fn parse_rejects_truncated_files() {
        let file = square();
        for (what, bytes) in encodings(&file) {
            let data_end = bytes
                .windows(b"</AppendedData>".len())
                .position(|w| w == b"</AppendedData>")
                .unwrap_or(bytes.len());
            for len in 0..bytes.len() {
                match parse_vtu(&bytes[..len]) {
                    Ok(parsed) => {
                        // Only whitespace of the appended section is missing.
                        assert!(
                            bytes[len.min(data_end)..data_end]
                                .iter()
                                .all(u8::is_ascii_whitespace),
                            "{what}: prefix of {len} bytes read"
                        );
                        assert_eq!(parsed, file, "{what}: prefix of {len} bytes");
                    }
                    Err(e) => {
                        assert_parse_error(Err(e), &format!("{what}, {len} bytes"));
                    }
                }
            }
        }
    }

    /// Builds a file with no cells around a `Points` element, and appended raw `data`.
    fn crafted(root: &str, points: usize, array: &str, data: &[u8]) -> Vec<u8> {
        let mut bytes = format!(
            "<VTKFile type=\"UnstructuredGrid\" byte_order=\"LittleEndian\" {root}>\
             <UnstructuredGrid><Piece NumberOfPoints=\"{points}\" NumberOfCells=\"0\">\
             <Points>{array}</Points><Cells>\
             <DataArray type=\"Int64\" Name=\"connectivity\"></DataArray>\
             <DataArray type=\"Int64\" Name=\"offsets\"></DataArray>\
             <DataArray type=\"UInt8\" Name=\"types\"></DataArray>\
             </Cells></Piece></UnstructuredGrid><AppendedData encoding=\"raw\">_"
        )
        .into_bytes();
        bytes.extend_from_slice(data);
        bytes
    }

    fn words(ws: &[u64]) -> Vec<u8> {
        ws.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// Counts in the file that would need terabytes are rejected before anything is allocated
    /// from them (an allocation that large would abort the test).
    #[test]
    fn parse_does_not_allocate_from_claimed_sizes() {
        let appended =
            "<DataArray type=\"UInt8\" NumberOfComponents=\"3\" format=\"appended\" offset=\"0\"/>";
        let huge = 1usize << 42;
        let cases: Vec<(&str, usize, &str, Vec<u8>)> = vec![
            // A byte count far beyond the file.
            (
                "header_type=\"UInt64\"",
                huge,
                appended,
                words(&[3 * huge as u64]),
            ),
            ("header_type=\"UInt64\"", 1, appended, words(&[u64::MAX])),
            // Block counts and sizes beyond the file.
            (
                "header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"",
                1,
                appended,
                words(&[u64::MAX, 3, 0]),
            ),
            (
                "header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"",
                1,
                appended,
                words(&[1 << 40, 3, 0]),
            ),
            (
                "header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"",
                1,
                appended,
                words(&[2, 3, 0, u64::MAX, u64::MAX]),
            ),
            // One block announcing 3 · 2^42 bytes that holds three.
            (
                "header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"",
                huge,
                appended,
                [
                    words(&[1, 3 * huge as u64, 0, 11]),
                    vec![
                        0x78, 0x01, 0x63, 0x60, 0x60, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01,
                    ],
                ]
                .concat(),
            ),
            // ASCII values far fewer than announced.
            (
                "",
                huge,
                "<DataArray type=\"Float64\" NumberOfComponents=\"3\">0 0 0</DataArray>",
                vec![],
            ),
            // A count that overflows.
            (
                "",
                usize::MAX / 2,
                "<DataArray type=\"Float64\" NumberOfComponents=\"3\"></DataArray>",
                vec![],
            ),
        ];
        for (k, (root, points, array, data)) in cases.into_iter().enumerate() {
            let (_, message) = assert_parse_error(
                parse_vtu(&crafted(root, points, array, &data)),
                &format!("case {k}"),
            );
            assert!(!message.is_empty());
        }
        // The crafted file itself is valid: three points of zeros.
        let three = [
            0x78, 0x01, 0x63, 0x60, 0x60, 0x00, 0x00, 0x00, 0x03, 0x00, 0x01,
        ];
        let ok = crafted(
            "header_type=\"UInt64\" compressor=\"vtkZLibDataCompressor\"",
            1,
            appended,
            &[words(&[1, 3, 0, 11]), three.to_vec()].concat(),
        );
        assert_eq!(
            parse_vtu(&ok).unwrap().points,
            DataValues::U8(vec![0, 0, 0])
        );
    }

    /// A valid ASCII file to edit.
    fn ascii_square() -> String {
        String::from_utf8(write_layout(
            &square(),
            Layout {
                format: Format::Ascii,
                word: 4,
                zlib: false,
                big_endian: false,
            },
        ))
        .unwrap()
    }

    #[test]
    fn parse_rejects_malformed_content() {
        let base = ascii_square();
        let at = |pat: &str| base.find(pat).unwrap() as u64;
        let cases: Vec<(String, &str, Option<u64>)> = vec![
            (
                base.replace("<VTKFile", "<VTK")
                    .replace("</VTKFile>", "</VTK>"),
                "not <VTKFile>",
                Some(at("<VTKFile")),
            ),
            (
                base.replace("type=\"UnstructuredGrid\"", "type=\"PUnstructuredGrid\""),
                "not UnstructuredGrid",
                None,
            ),
            (
                base.replace(
                    "</Piece>",
                    "</Piece><Piece NumberOfPoints=\"0\" NumberOfCells=\"0\"></Piece>",
                ),
                "more than one <Piece>",
                None,
            ),
            (
                base.replace("<Points>", "<Pts>")
                    .replace("</Points>", "</Pts>"),
                "has no <Points>",
                None,
            ),
            (
                base.replace("vtkZLibDataCompressor", "vtkLZ4DataCompressor")
                    .replace(
                        "byte_order",
                        "compressor=\"vtkLZ4DataCompressor\" byte_order",
                    ),
                "",
                None,
            ),
            (
                base.replacen("type=\"Float32\"", "type=\"Int16\"", 1),
                "Int16\" is not supported",
                None,
            ),
            (
                base.replace("byte_order=\"LittleEndian\"", "byte_order=\"MiddleEndian\""),
                "MiddleEndian",
                None,
            ),
            (
                base.replace("header_type=\"UInt32\"", "header_type=\"UInt16\""),
                "UInt16",
                None,
            ),
            (
                base.replace("Name=\"a&amp;b", "Name=\"a&nbsp;b"),
                "unknown entity &nbsp;",
                None,
            ),
            (
                base.replace("Name=\"a&amp;b", "Name=\"a&b"),
                "unknown entity",
                None,
            ),
            (
                base.replacen("format=\"ascii\"", "format=\"hex\"", 1),
                "format=\"hex\"",
                None,
            ),
            (
                base.replacen("format=\"ascii\"", "format=\"appended\" offset=\"0\"", 1),
                "without <AppendedData>",
                None,
            ),
            (
                base.replacen("\n3 6\n", "\n4 3\n", 1),
                "below the one before",
                None,
            ),
            (
                base.replacen("\n3 6\n", "\n3 5\n", 1),
                "more than the 5 values",
                None,
            ),
            (
                base.replacen("\n0 1 2 1 3 2\n", "\n0 1 2 1 4 2\n", 1),
                "entry 4 is not one of the 4 points",
                None,
            ),
            (
                base.replacen("\n0 1 2 1 3 2\n", "\n0 1 2 1 -3 2\n", 1),
                "entry -3",
                None,
            ),
            (
                base.replacen("\n5 5\n", "\n5 300\n", 1),
                "is not a value",
                None,
            ),
            (
                base.replacen("\n5 5\n", "\n5 5 5\n", 1),
                "more than the 2 values",
                Some(at("\n5 5\n") + 5),
            ),
            (
                base.replacen("\n5 5\n", "\n5 x\n", 1),
                "\"x\" is not a value",
                Some(at("\n5 5\n") + 3),
            ),
            (
                base.replace(
                    "NumberOfComponents=\"3\" RangeMin=\"0\" format=\"ascii\">\n0 0 0",
                    "NumberOfComponents=\"2\" RangeMin=\"0\" format=\"ascii\">\n0 0 0",
                ),
                "three components",
                None,
            ),
            (
                base.replace(
                    "Name=\"u\" NumberOfComponents=\"1\"",
                    "Name=\"u\" NumberOfComponents=\"0\"",
                ),
                "NumberOfComponents=\"0\"",
                None,
            ),
            (
                base.replace("NumberOfCells=\"2\"", "NumberOfCells=\"-2\""),
                "not a non-negative integer",
                None,
            ),
            (base.replace("</Cells>", "</Cell>"), "closes <Cells>", None),
            (
                base.replace("</VTKFile>", ""),
                "is not closed",
                Some(at("<VTKFile")),
            ),
            (
                base.replace("</VTKFile>", "</VTKFile>junk"),
                "outside the root",
                None,
            ),
            (
                base.replace("</VTKFile>", "</VTKFile><VTKFile/>"),
                "a second root",
                None,
            ),
            (
                base.replace("<!-- written", "<![CDATA[ written")
                    .replace("test -->", "test ]]>"),
                "CDATA",
                None,
            ),
            (
                base.replace("<UnstructuredGrid>", "<UnstructuredGrid a=\"1\" a=\"2\">"),
                "repeated",
                None,
            ),
            (
                base.replace("<UnstructuredGrid>", "<UnstructuredGrid a=1>"),
                "quoted",
                None,
            ),
            (
                base.replace("<Cells>", &"<x>".repeat(70))
                    .replace("</Cells>", &"</x>".repeat(70)),
                "nested deeper",
                None,
            ),
            (String::new(), "no root element", Some(0)),
        ];
        for (k, (text, needle, offset)) in cases.into_iter().enumerate() {
            let (found, message) =
                assert_parse_error(parse_vtu(text.as_bytes()), &format!("case {k}"));
            assert!(
                message.contains(needle),
                "case {k}: {message:?} lacks {needle:?}"
            );
            if let Some(offset) = offset {
                assert_eq!(found, offset, "case {k}: {message}");
            }
        }
    }

    #[test]
    fn parse_reports_base64_errors_at_their_byte() {
        let text = String::from_utf8(write(&square(), Binary::Base64, false)).unwrap();
        let open = "Name=\"types\" NumberOfComponents=\"1\" format=\"binary\">\n          ";
        let start = text.find(open).unwrap() + open.len();
        let mut bytes = text.into_bytes();
        bytes[start + 5] = b'!';
        let (offset, message) = assert_parse_error(parse_vtu(&bytes), "invalid character");
        assert_eq!(offset, (start + 5) as u64, "{message}");
        assert!(message.contains("base64"), "{message}");
    }

    #[test]
    fn parse_rejects_corrupt_binary_headers() {
        // Uncompressed: a byte count other than the data's. zlib: a block that does not
        // decompress, or decompresses to the wrong size.
        let file = square();
        let bytes = write(&file, Binary::Appended, false);
        let marker = b"encoding=\"raw\">\n   _";
        let data = bytes
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap()
            + marker.len();
        let mut wrong_count = bytes.clone();
        wrong_count[data] = 15; // `u` holds 16 bytes
        let (offset, message) = assert_parse_error(parse_vtu(&wrong_count), "byte count");
        assert_eq!(offset, data as u64);
        assert!(message.contains("15 bytes where 16"), "{message}");

        let zipped = write(&file, Binary::Appended, true);
        let data = zipped
            .windows(marker.len())
            .position(|w| w == marker)
            .unwrap()
            + marker.len();
        // `u`: header [1, 32768, 16, size], then the stream.
        let mut bad_stream = zipped.clone();
        bad_stream[data + 32] ^= 0xff;
        let (_, message) = assert_parse_error(parse_vtu(&bad_stream), "zlib stream");
        assert!(message.contains("zlib block 0"), "{message}");
        let mut bad_last = zipped.clone();
        bad_last[data + 16] = 15;
        let (_, message) = assert_parse_error(parse_vtu(&bad_last), "last block size");
        assert!(message.contains("do not hold the 16 bytes"), "{message}");
        let mut too_long = zipped;
        // Block size 32, last block 64.
        too_long[data + 8..data + 24].copy_from_slice(&words(&[32, 64]));
        let (_, message) = assert_parse_error(parse_vtu(&too_long), "last block above size");
        assert!(message.contains("exceeds the block size"), "{message}");
    }

    fn finite_f32() -> impl Strategy<Value = f32> {
        use proptest::num::f32;
        f32::POSITIVE | f32::NEGATIVE | f32::NORMAL | f32::SUBNORMAL | f32::ZERO | f32::INFINITE
    }

    fn finite_f64() -> impl Strategy<Value = f64> {
        use proptest::num::f64;
        f64::POSITIVE | f64::NEGATIVE | f64::NORMAL | f64::SUBNORMAL | f64::ZERO | f64::INFINITE
    }

    fn values(n: usize) -> BoxedStrategy<DataValues> {
        use proptest::collection::vec;
        prop_oneof![
            vec(finite_f32(), n).prop_map(DataValues::F32),
            vec(finite_f64(), n).prop_map(DataValues::F64),
            vec(any::<i32>(), n).prop_map(DataValues::I32),
            vec(any::<u32>(), n).prop_map(DataValues::U32),
            vec(any::<i64>(), n).prop_map(DataValues::I64),
            vec(any::<u8>(), n).prop_map(DataValues::U8),
        ]
        .boxed()
    }

    fn array(entries: usize) -> impl Strategy<Value = DataArray> {
        ("[a-zA-Z0-9 _&<>\"'\t\n\ré]{0,8}", 1usize..4).prop_flat_map(move |(name, components)| {
            values(entries * components).prop_map(move |values| DataArray {
                name: name.clone(),
                components,
                values,
            })
        })
    }

    /// A random small file whose arrays fit together.
    fn file() -> impl Strategy<Value = VtuFile> {
        use proptest::collection::vec;
        (1usize..6, vec(1usize..5, 0..5), any::<bool>()).prop_flat_map(|(n, sizes, f32_points)| {
            let offsets: Vec<i64> = sizes
                .iter()
                .scan(0i64, |end, &s| {
                    *end += s as i64;
                    Some(*end)
                })
                .collect();
            let total = sizes.iter().sum::<usize>();
            let points = if f32_points {
                vec(finite_f32(), 3 * n).prop_map(DataValues::F32).boxed()
            } else {
                vec(finite_f64(), 3 * n).prop_map(DataValues::F64).boxed()
            };
            (
                points,
                vec(0..n as i64, total),
                vec(any::<u8>(), sizes.len()),
                vec(array(n), 0..3),
                vec(array(sizes.len()), 0..3),
            )
                .prop_map(
                    move |(points, connectivity, types, point_data, cell_data)| VtuFile {
                        points,
                        connectivity,
                        offsets: offsets.clone(),
                        types,
                        point_data,
                        cell_data,
                    },
                )
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        /// `parse(write(f)) == f` for the writer's four encodings and every test layout.
        #[test]
        fn parse_round_trips_every_encoding(file in file()) {
            for (what, bytes) in encodings(&file) {
                let parsed = parse_vtu(&bytes);
                prop_assert!(matches!(&parsed, Ok(f) if *f == file), "{}: {:?}", what, parsed);
            }
        }

        /// Overwritten bytes give the file back or `Parse`, never a panic.
        #[test]
        fn parse_survives_corrupt_bytes(
            k in 0usize..36,
            edits in proptest::collection::vec((any::<prop::sample::Index>(), any::<u8>()), 1..4),
        ) {
            let (what, mut bytes) = encodings(&square()).swap_remove(k);
            for (at, value) in edits {
                let at = at.index(bytes.len());
                bytes[at] = value;
            }
            match parse_vtu(&bytes) {
                Ok(_) | Err(GridError::Parse { format: "VTU", .. }) => {}
                Err(e) => prop_assert!(false, "{}: {:?}", what, e),
            }
        }
    }
}
