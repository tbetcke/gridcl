//! `SortKeys`: keys of one to four u32 words, planar and word-major.

use cubecl::prelude::*;

use crate::{DeviceBuffer, Result};

/// `len` keys of 1..=4 u32 words, planar: word w of key i at `w * pad_len(len) + i`; word 0
/// most significant.
pub struct SortKeys {
    words: DeviceBuffer<u32>,
    num_words: usize,
    len: usize,
}

// Listing S1b has no `is_empty`; the public API follows it.
#[allow(clippy::len_without_is_empty)]
impl SortKeys {
    /// `len` keys of `num_words` words with unspecified contents.
    pub fn uninit(_client: &Client, _len: usize, _num_words: usize) -> Result<Self> {
        todo!()
    }

    /// Uploads key-major host keys: key i is `keys[i * num_words..(i + 1) * num_words]`.
    pub fn from_host(_client: &Client, _keys: &[u32], _num_words: usize) -> Result<Self> {
        todo!()
    }

    /// The keys, key-major.
    pub fn to_host(&self) -> Result<Vec<u32>> {
        todo!()
    }

    /// The number of keys.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Words per key, 1..=4.
    pub fn num_words(&self) -> usize {
        self.num_words
    }

    /// The planar word buffer.
    pub fn words(&self) -> &DeviceBuffer<u32> {
        &self.words
    }

    /// The planar word buffer, mutable.
    pub fn words_mut(&mut self) -> &mut DeviceBuffer<u32> {
        &mut self.words
    }
}
