//! `MeshTags`, `TagRegistry` and `Codim` (docs/design.md, "Tags, regions and submeshes";
//! ADR-0022).

use gridcl_device::{DeviceBuffer, Result};

/// Tags of the entities of one dimension, on the device: entries sorted by (index, value), an
/// index repeating only with distinct values (an element in several physical groups). Indices
/// are flat entity indices; cells (dim = tdim) are numbered over the cell blocks in block
/// order.
pub struct MeshTags {
    /// The topological dimension of the tagged entities.
    pub dim: usize,
    /// The flat local entity index of each entry.
    pub indices: DeviceBuffer<u32>,
    /// The tag value of each entry.
    pub values: DeviceBuffer<i32>,
}

/// Codimension addressing (NGSolve): a dimension relative to D_TOP.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codim {
    /// D_TOP.
    Vol,
    /// D_TOP − 1.
    Bnd,
    /// D_TOP − 2.
    BBnd,
    /// D_TOP − 3.
    BBBnd,
}

impl Codim {
    /// The dimension this codimension addresses in a mesh of topological dimension `tdim`;
    /// `None` below 0.
    pub fn dim(self, tdim: usize) -> Option<usize> {
        let _ = tdim;
        todo!("M4 T10")
    }
}

/// The tags of a mesh: per dimension one physical and one elementary [`MeshTags`], the name
/// table (name, dimension) → value, and each physical group's size, recorded on the host when
/// the tags were read. `Default` is empty.
#[derive(Default)]
pub struct TagRegistry {}

impl TagRegistry {
    /// The physical names: (name, dimension, value).
    pub fn names(&self) -> impl Iterator<Item = (&str, usize, i32)> {
        todo!("M4 T10");
        #[allow(unreachable_code)]
        std::iter::empty()
    }

    /// The value of the physical group `name` in dimension `dim`.
    pub fn lookup(&self, dim: usize, name: &str) -> Option<i32> {
        let _ = (dim, name);
        todo!("M4 T10")
    }

    /// The physical tags of dimension `dim`, if any.
    pub fn physical(&self, dim: usize) -> Option<&MeshTags> {
        let _ = dim;
        todo!("M4 T10")
    }

    /// The elementary (entity) tags of dimension `dim`, if any.
    pub fn elementary(&self, dim: usize) -> Option<&MeshTags> {
        let _ = dim;
        todo!("M4 T10")
    }

    /// The number of entities in physical group `value` of dimension `dim`.
    pub fn group_size(&self, dim: usize, value: i32) -> Option<usize> {
        let _ = (dim, value);
        todo!("M4 T10")
    }

    /// The entities of physical group `value` of dimension `dim`, selected on the device.
    ///
    /// # Errors
    ///
    /// `UnknownTag` if dimension `dim` has no such group; allocation and device errors.
    pub fn group(&self, dim: usize, value: i32) -> Result<MeshTags> {
        let _ = (dim, value);
        todo!("M4 T10")
    }

    /// The entities of the physical group `name` in the dimension `codim` addresses.
    ///
    /// # Errors
    ///
    /// `UnknownTag` if that dimension has no group of this name; allocation and device errors.
    pub fn named(&self, codim: Codim, name: &str) -> Result<MeshTags> {
        let _ = (codim, name);
        todo!("M4 T10")
    }
}
