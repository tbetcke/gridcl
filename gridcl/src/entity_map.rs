//! `EntityMap`: the map from the cells, vertices and nodes of a submesh to those of its parent
//! (docs/design.md, "Tags, regions and submeshes"; ADR-0039, item 1).

use gridcl_device::{DeviceBuffer, Result};

/// The map from a submesh to its parent, on the device, for every local child cell, vertex and
/// node, owned or ghost (ADR-0039, item 1).
#[derive(Debug, Default)]
pub struct EntityMap {
    _private: (),
}

impl EntityMap {
    /// The dimension of the parent entities the child cells come from: tdim − 1 for `boundary`
    /// and `facet_submesh`, 1 for `edge_submesh`, tdim for `cell_submesh`.
    pub fn parent_dim(&self) -> usize {
        todo!("EntityMap::parent_dim: M9 T3")
    }

    /// The flat local parent entity of each local child cell, in flat child order (over the
    /// child's cell blocks in block order).
    pub fn child_to_parent(&self) -> &DeviceBuffer<u32> {
        todo!("EntityMap::child_to_parent: M9 T3")
    }

    /// The sparse inverse of [`child_to_parent`](EntityMap::child_to_parent): the parent flat
    /// indices that have a child, ascending, and the flat child cell of each.
    pub fn parent_to_child(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<u32>) {
        todo!("EntityMap::parent_to_child: M9 T3")
    }

    /// The parent local vertex of each child local vertex.
    pub fn vertex_to_parent(&self) -> &DeviceBuffer<u32> {
        todo!("EntityMap::vertex_to_parent: M9 T3")
    }

    /// The parent geometry node of each child geometry node.
    pub fn node_to_parent(&self) -> &DeviceBuffer<u32> {
        todo!("EntityMap::node_to_parent: M9 T4a")
    }

    /// A deep copy of every device array.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    pub fn try_clone(&self) -> Result<Self> {
        todo!("EntityMap::try_clone: M9 T3")
    }
}
