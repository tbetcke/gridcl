//! The value encoding of binary MSH 4.1 files: `int` in 4 bytes, `size_t` in the 4 or 8 bytes
//! of the data-size field, `double` in 8 bytes, all in the byte order of the endianness
//! marker (the `int` 1 after the `$MeshFormat` line).

/// The encoding of a binary file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Binary {
    /// Big-endian values.
    pub big_endian: bool,
    /// The width of `size_t`: 4 or 8.
    pub size_t: usize,
}

impl Binary {
    /// The encoding whose marker is `one` (the four bytes of the `int` 1) and whose `size_t`
    /// has `size_t` bytes; `None` if the marker is neither byte order of 1.
    pub fn from_marker(one: [u8; 4], size_t: usize) -> Option<Self> {
        let big_endian = match one {
            [1, 0, 0, 0] => false,
            [0, 0, 0, 1] => true,
            _ => return None,
        };
        Some(Binary { big_endian, size_t })
    }

    /// An `int` from its four bytes.
    pub fn int(self, b: [u8; 4]) -> i32 {
        if self.big_endian {
            i32::from_be_bytes(b)
        } else {
            i32::from_le_bytes(b)
        }
    }

    /// A `size_t` from its `size_t` bytes.
    pub fn size(self, b: &[u8]) -> u64 {
        debug_assert_eq!(b.len(), self.size_t);
        let mut word = [0u8; 8];
        if self.big_endian {
            word[8 - b.len()..].copy_from_slice(b);
            u64::from_be_bytes(word)
        } else {
            word[..b.len()].copy_from_slice(b);
            u64::from_le_bytes(word)
        }
    }

    /// A `double` from its eight bytes.
    pub fn float(self, b: [u8; 8]) -> f64 {
        if self.big_endian {
            f64::from_be_bytes(b)
        } else {
            f64::from_le_bytes(b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_binary_values_in_both_byte_orders() {
        let le = Binary::from_marker([1, 0, 0, 0], 8).unwrap();
        let be = Binary::from_marker([0, 0, 0, 1], 4).unwrap();
        assert!(!le.big_endian && be.big_endian);
        assert!(Binary::from_marker([0, 1, 0, 0], 8).is_none());
        assert_eq!(le.int((-7i32).to_le_bytes()), -7);
        assert_eq!(be.int((-7i32).to_be_bytes()), -7);
        assert_eq!(le.size(&(1u64 << 40).to_le_bytes()), 1 << 40);
        assert_eq!(be.size(&123_456u32.to_be_bytes()), 123_456);
        assert_eq!(le.float(0.1f64.to_le_bytes()), 0.1);
        assert_eq!(be.float((-2.5f64).to_be_bytes()), -2.5);
    }
}
