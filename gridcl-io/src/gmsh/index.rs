//! Chunked reading of binary MSH 4.1 files (docs/design.md, "Gmsh MSH 4.1", parallel read;
//! ADR-0035, rule 1): [`MshIndex::scan`] reads the small sections and the block headers by
//! seeking, and [`MshIndex::read_nodes`] and [`MshIndex::read_elements`] read only the bytes of
//! a range of nodes or elements.

use std::ops::Range;
use std::path::Path;

use gridcl_device::Result;

use super::ElementBlock;

/// The layout of an MSH 4.1 file: its encoding, its small sections and the byte offsets of
/// its node and element blocks.
#[derive(Clone, Debug)]
pub struct MshIndex {
    _layout: (),
}

/// A range of nodes in file order: their tags and coordinates.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NodeChunk {
    /// The node tags.
    pub tags: Vec<i64>,
    /// The coordinates.
    pub coords: Vec<[f64; 3]>,
}

/// A range of the elements of one dimension in file order, as the element blocks that hold
/// them, cut to the range.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementChunk {
    /// The blocks, in file order.
    pub blocks: Vec<ElementBlock>,
}

impl MshIndex {
    /// Scans the file at `path`: the small sections are parsed, the node and element block
    /// headers located by seeking.
    ///
    /// # Errors
    ///
    /// `Io` if the file cannot be read; `Parse` with the byte offset for a malformed section.
    pub fn scan(path: &Path) -> Result<MshIndex> {
        let _ = path;
        todo!("MshIndex::scan (M8 T1)")
    }

    /// Whether the file is binary; an ASCII file has no block offsets.
    pub fn is_binary(&self) -> bool {
        todo!("MshIndex::is_binary (M8 T1)")
    }

    /// The number of nodes.
    pub fn num_nodes(&self) -> usize {
        todo!("MshIndex::num_nodes (M8 T1)")
    }

    /// The number of elements of dimension `dim`.
    pub fn num_elements(&self, dim: usize) -> usize {
        let _ = dim;
        todo!("MshIndex::num_elements (M8 T1)")
    }

    /// The nodes `range` in file order.
    ///
    /// # Errors
    ///
    /// `Io`; `Parse` for a malformed block.
    pub fn read_nodes(&self, range: Range<usize>) -> Result<NodeChunk> {
        let _ = range;
        todo!("MshIndex::read_nodes (M8 T1)")
    }

    /// The elements `range` of dimension `dim`, in file order within the dimension.
    ///
    /// # Errors
    ///
    /// `Io`; `Parse` for a malformed block.
    pub fn read_elements(&self, dim: usize, range: Range<usize>) -> Result<ElementChunk> {
        let _ = (dim, range);
        todo!("MshIndex::read_elements (M8 T1)")
    }
}
