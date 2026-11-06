//! Cell submeshes (ADR-0039, item 4): the tagged cells as a mesh of the same dimensions.
//!
//! Each parent cell block with tagged cells on some rank gives one child block with the
//! parent's element. A child cell keeps its parent's vertex row, dofmap row, origin id (item 7)
//! and flags; its owner and ghost status are the parent's (item 6).

use gridcl_device::{DeviceBuffer, DeviceStrided, GridError, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::{Comm, CommDyn};

use super::select::{self, ParentBlock, Selected};
use super::topology::{self, ChildTopology};
use super::{ChildBlock, fit, gather_rows};
use crate::{Mesh, MeshTags};

/// The child topology of the cells of `mesh` tagged by `tags`, the child blocks' geometry
/// (the parent elements and dofmap rows) and the child cells' flags (the parents'), per child
/// block. Collective.
///
/// # Errors
///
/// `DimensionMismatch` for tags of another dimension than tdim; `TooLarge`; allocation, device
/// and communication errors.
#[allow(clippy::type_complexity)]
pub(crate) fn cell_children<F, TD, GD, C>(
    mesh: &Mesh<F, TD, GD, C>,
    tags: &MeshTags,
) -> Result<(ChildTopology<TD>, Vec<ChildBlock>, Vec<DeviceBuffer<u32>>)>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    let comm: &dyn CommDyn = &mesh.comm;
    if tags.dim != TD::N {
        return Err(GridError::DimensionMismatch {
            expected: (TD::N, GD::N),
            found: (tags.dim, GD::N),
        });
    }
    let parent = &mesh.topology;
    let mut blocks = Vec::new();
    let mut offset = 0usize;
    for b in parent.cell_blocks() {
        blocks.push(ParentBlock {
            cell: b.cell.into(),
            offset,
            index_map: &b.index_map,
        });
        offset += b.index_map.num_local();
    }
    fit(offset)?;
    let mask = select::tag_mask(tags, offset)?;
    let mut selected = Vec::new();
    let mut geometry = Vec::new();
    let mut flags = Vec::new();
    for c in select::select(comm, &mask, &blocks)? {
        let t = &parent.cell_blocks()[c.block];
        let g = &mesh.geometry.blocks()[c.block];
        let (nvc, k) = (t.vertices.stride, g.dofmap.stride);
        let off = c.parent_offset;
        geometry.push((
            c.cell,
            g.element.degree,
            g.element.family,
            DeviceStrided {
                data: gather_rows(&g.dofmap.data, &c.parents, k as usize, off)?,
                stride: k,
            },
        ));
        flags.push(gather_rows(&t.flags, &c.parents, 1, off)?);
        selected.push(Selected {
            cell: c.cell,
            rows: DeviceStrided {
                data: gather_rows(&t.vertices.data, &c.parents, nvc as usize, off)?,
                stride: nvc,
            },
            origin_ids: gather_rows(&t.origin_ids, &c.parents, 2, off)?,
            parents: c.parents,
            num_owned: c.num_owned,
            parent_map: c.parent_map,
            parent_offset: off,
            facet_of: None,
        });
    }
    let child = topology::child_topology(comm, parent, selected, false)?;
    Ok((child, geometry, flags))
}
