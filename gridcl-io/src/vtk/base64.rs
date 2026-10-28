//! Base64 (RFC 4648, standard alphabet, `=` padding) for VTU binary data.
//!
//! The decoder accepts padding at the end of every four-character quantum, not only the last,
//! so it decodes the concatenation of separately padded streams to the concatenation of their
//! bytes. VTK writes compressed inline arrays that way: the base64 of the block header followed
//! by the base64 of the compressed blocks. ASCII whitespace is skipped anywhere.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// Appends the base64 encoding of `data` to `out`, padded with `=`.
pub(crate) fn encode_into(data: &[u8], out: &mut Vec<u8>) {
    out.reserve(data.len().div_ceil(3) * 4);
    for chunk in data.chunks(3) {
        let b = [
            chunk[0],
            *chunk.get(1).unwrap_or(&0),
            *chunk.get(2).unwrap_or(&0),
        ];
        let n = u32::from_be_bytes([0, b[0], b[1], b[2]]);
        for k in 0..4 {
            if k <= chunk.len() {
                out.push(ALPHABET[(n >> (18 - 6 * k)) as usize & 63]);
            } else {
                out.push(b'=');
            }
        }
    }
}

/// Why base64 input was rejected, and at which byte of the input.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecodeError {
    /// The byte offset in the input.
    pub offset: usize,
    /// What is wrong there.
    pub message: &'static str,
}

/// The 6-bit value of an alphabet character.
fn sextet(c: u8) -> Option<u32> {
    let v = match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        b'/' => 63,
        _ => return None,
    };
    Some(u32::from(v))
}

/// Decodes `input`, skipping ASCII whitespace.
///
/// # Errors
///
/// A character outside the alphabet, padding other than `xx==` or `xxx=` at the end of a
/// quantum, or input that ends inside a quantum.
pub(crate) fn decode(input: &[u8]) -> Result<Vec<u8>, DecodeError> {
    let mut out = Vec::with_capacity(input.len() / 4 * 3);
    // The quantum being read: its characters' values, their count and the padding count.
    let mut n = 0u32;
    let mut len = 0usize;
    let mut pad = 0usize;
    let mut start = 0usize;
    for (i, &c) in input.iter().enumerate() {
        if c.is_ascii_whitespace() {
            continue;
        }
        if len == 0 {
            start = i;
        }
        if c == b'=' {
            if len < 2 {
                return Err(DecodeError {
                    offset: i,
                    message: "base64 padding in the first two characters of a quantum",
                });
            }
            pad += 1;
        } else if pad > 0 {
            return Err(DecodeError {
                offset: i,
                message: "base64 data after padding inside a quantum",
            });
        } else {
            let Some(v) = sextet(c) else {
                return Err(DecodeError {
                    offset: i,
                    message: "a character outside the base64 alphabet",
                });
            };
            n |= v << (18 - 6 * len);
        }
        len += 1;
        if len == 4 {
            let bytes = n.to_be_bytes();
            out.extend_from_slice(&bytes[1..4 - pad]);
            (n, len, pad) = (0, 0, 0);
        }
    }
    if len != 0 {
        return Err(DecodeError {
            offset: start,
            message: "base64 input ends inside a quantum",
        });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn encode(data: &[u8]) -> String {
        let mut out = Vec::new();
        encode_into(data, &mut out);
        String::from_utf8(out).unwrap()
    }

    const RFC4648: [(&str, &str); 7] = [
        ("", ""),
        ("f", "Zg=="),
        ("fo", "Zm8="),
        ("foo", "Zm9v"),
        ("foob", "Zm9vYg=="),
        ("fooba", "Zm9vYmE="),
        ("foobar", "Zm9vYmFy"),
    ];

    #[test]
    fn rfc4648_vectors_both_ways() {
        for (plain, coded) in RFC4648 {
            assert_eq!(encode(plain.as_bytes()), coded);
            assert_eq!(decode(coded.as_bytes()).unwrap(), plain.as_bytes());
        }
    }

    #[test]
    fn every_byte_value_round_trips() {
        let data: Vec<u8> = (0..=255).chain((0..=255).rev()).collect();
        for k in 0..data.len() {
            assert_eq!(decode(encode(&data[..k]).as_bytes()).unwrap(), &data[..k]);
        }
        assert_eq!(encode(&[0xfb, 0xff]), "+/8=");
    }

    #[test]
    fn whitespace_is_skipped() {
        assert_eq!(decode(b"  Zm9v\n\tYmE=\r\n ").unwrap(), b"fooba");
        assert_eq!(decode(b" \n ").unwrap(), b"");
    }

    #[test]
    fn concatenated_padded_streams_decode_to_concatenated_bytes() {
        let coded = format!("{}{}{}", encode(b"f"), encode(b"fo"), encode(b"foo"));
        assert_eq!(decode(coded.as_bytes()).unwrap(), b"ffofoo");
    }

    #[test]
    fn invalid_input_is_rejected() {
        let cases: [(&[u8], usize); 8] = [
            (b"Zm9", 0),       // ends inside a quantum
            (b"Zm9vY", 4),     // likewise, in the second quantum
            (b"Zm-v", 2),      // outside the alphabet
            (b"Zm9v\0AAA", 4), // NUL
            (b"=AAA", 0),      // padding first
            (b"Z===", 1),      // padding second
            (b"Zm=v", 3),      // data after padding
            (b"Zm9v\xc3\xa9AA", 4),
        ];
        for (input, offset) in cases {
            let e = decode(input).unwrap_err();
            assert_eq!(e.offset, offset, "{input:?}: {}", e.message);
        }
    }
}
