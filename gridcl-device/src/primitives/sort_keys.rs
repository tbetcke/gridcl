//! `SortKeys`: keys of one to four u32 words, planar and word-major.

use cubecl::prelude::*;

use crate::buffer::pad_len;
use crate::{DeviceBuffer, GridError, Result};

/// `len` keys of 1..=4 u32 words, planar: word w of key i at `w * pad_len(len) + i`; word 0
/// most significant. Each plane is padded to `VECTOR_PAD`, so the lanes of a vector are
/// independent keys.
///
/// # Example
///
/// ```
/// use gridcl_device::SortKeys;
///
/// let client = gridcl_testing::test_client();
/// // Two keys of two words, key-major: (1, 2) and (3, 4).
/// let keys = SortKeys::from_host(&client, &[1, 2, 3, 4], 2).unwrap();
/// assert_eq!((keys.len(), keys.num_words()), (2, 2));
/// assert_eq!(keys.to_host().unwrap(), vec![1, 2, 3, 4]);
/// // Planar: word 0 of both keys, then word 1, each plane padded to 16.
/// let words = keys.words().to_host().unwrap();
/// assert_eq!((words[0], words[1], words[16], words[17]), (1, 3, 2, 4));
/// ```
pub struct SortKeys {
    words: DeviceBuffer<u32>,
    num_words: usize,
    len: usize,
}

// Listing S1b has no `is_empty`; the public API follows it.
#[allow(clippy::len_without_is_empty)]
impl SortKeys {
    /// `len` keys of `num_words` words with unspecified contents.
    ///
    /// # Errors
    ///
    /// `TooLarge` if the planar length overflows, `AllocationTooLarge` above the device's
    /// `max_page_size`.
    ///
    /// # Panics
    ///
    /// If `num_words` is not in 1..=4.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{SortKeys, VECTOR_PAD};
    ///
    /// let client = gridcl_testing::test_client();
    /// let keys = SortKeys::uninit(&client, 5, 3).unwrap();
    /// assert_eq!(keys.words().len(), 3 * VECTOR_PAD);
    /// ```
    pub fn uninit(client: &Client, len: usize, num_words: usize) -> Result<Self> {
        let words = DeviceBuffer::uninit(client, planar_len(len, num_words)?)?;
        Ok(SortKeys {
            words,
            num_words,
            len,
        })
    }

    /// Uploads key-major host keys: key i is `keys[i * num_words..(i + 1) * num_words]`.
    /// The padding of each plane is zero.
    ///
    /// # Errors
    ///
    /// As for [`uninit`](Self::uninit).
    ///
    /// # Panics
    ///
    /// If `num_words` is not in 1..=4, or `keys.len()` is not a multiple of it.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::SortKeys;
    ///
    /// let client = gridcl_testing::test_client();
    /// let keys = SortKeys::from_host(&client, &[7, 3, 5], 1).unwrap();
    /// assert_eq!(keys.to_host().unwrap(), vec![7, 3, 5]);
    /// ```
    pub fn from_host(client: &Client, keys: &[u32], num_words: usize) -> Result<Self> {
        check_num_words(num_words);
        assert!(
            keys.len().is_multiple_of(num_words),
            "{} words are not whole keys of {num_words} words",
            keys.len()
        );
        let len = keys.len() / num_words;
        let pad = pad_len(len)?;
        let mut planar = vec![0u32; planar_len(len, num_words)?];
        for (i, key) in keys.chunks_exact(num_words).enumerate() {
            for (w, &word) in key.iter().enumerate() {
                planar[w * pad + i] = word;
            }
        }
        Ok(SortKeys {
            words: DeviceBuffer::from_host(client, &planar)?,
            num_words,
            len,
        })
    }

    /// The keys, key-major (an explicit, blocking device-to-host copy).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::SortKeys;
    ///
    /// let client = gridcl_testing::test_client();
    /// let keys = SortKeys::from_host(&client, &[1, 0, 0, 9], 2).unwrap();
    /// assert_eq!(keys.to_host().unwrap(), vec![1, 0, 0, 9]);
    /// ```
    pub fn to_host(&self) -> Result<Vec<u32>> {
        let planar = self.words.to_host()?;
        let pad = pad_len(self.len)?;
        let mut keys = Vec::with_capacity(self.len * self.num_words);
        for i in 0..self.len {
            keys.extend((0..self.num_words).map(|w| planar[w * pad + i]));
        }
        Ok(keys)
    }

    /// The number of keys.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::SortKeys;
    ///
    /// let client = gridcl_testing::test_client();
    /// // Three two-word keys, key-major on the host.
    /// assert_eq!(SortKeys::from_host(&client, &[1, 2, 3, 4, 5, 6], 2).unwrap().len(), 3);
    /// ```
    pub fn len(&self) -> usize {
        self.len
    }

    /// Words per key, 1..=4.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::SortKeys;
    ///
    /// let client = gridcl_testing::test_client();
    /// assert_eq!(SortKeys::from_host(&client, &[1, 2, 3, 4, 5, 6], 2).unwrap().num_words(), 2);
    /// ```
    pub fn num_words(&self) -> usize {
        self.num_words
    }

    /// The planar word buffer: `num_words · pad_len(len)` words.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{SortKeys, VECTOR_PAD};
    ///
    /// let client = gridcl_testing::test_client();
    /// // Keys [1, 2] and [3, 4]: word 0 of every key comes first, then word 1.
    /// let keys = SortKeys::from_host(&client, &[1, 2, 3, 4], 2).unwrap();
    /// let planar = keys.words().to_host().unwrap();
    /// assert_eq!(planar.len(), 2 * VECTOR_PAD);
    /// assert_eq!((planar[0], planar[1]), (1, 3));
    /// assert_eq!((planar[VECTOR_PAD], planar[VECTOR_PAD + 1]), (2, 4));
    /// ```
    pub fn words(&self) -> &DeviceBuffer<u32> {
        &self.words
    }

    /// The planar word buffer, mutable.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{SortKeys, fill};
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut keys = SortKeys::from_host(&client, &[5, 6, 7], 1).unwrap();
    /// fill(keys.words_mut(), 0u32).unwrap();
    /// assert_eq!(keys.to_host().unwrap(), vec![0, 0, 0]);
    /// ```
    pub fn words_mut(&mut self) -> &mut DeviceBuffer<u32> {
        &mut self.words
    }
}

/// Words of the planar buffer: `num_words · pad_len(len)`.
fn planar_len(len: usize, num_words: usize) -> Result<usize> {
    check_num_words(num_words);
    pad_len(len)?
        .checked_mul(num_words)
        .ok_or(GridError::TooLarge)
}

fn check_num_words(num_words: usize) {
    assert!(
        (1..=4).contains(&num_words),
        "sort keys have 1 to 4 words, not {num_words}"
    );
}
