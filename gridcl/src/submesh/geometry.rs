//! The child geometry of a submesh (ADR-0039, items 8 and 9): child dofmaps through the closure
//! node maps, node compaction, `node_to_parent`, and the parent's origin, so that child
//! coordinates are bitwise the parent's.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, Real, Result, debug_assert_bindings, gather,
};
use gridcl_dims::{CellType, Dim, EmbeddedIn, TopoDim};
use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
use gridcl_parallel::{CommDyn, CommExt, IndexMap};

use super::topology::compact;
use super::{fit, plan};
use crate::builder::cell_name;

/// `out[i · gd + g] = nodes[map[i] · gd + g]` for i < len and g < gd.
#[cube(launch_unchecked)]
fn gather_nodes_kernel<F: Float>(
    nodes: &[F],
    map: &[u32],
    out: &mut [F],
    len: u32,
    gd: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let p = map[i] as usize;
            for g in 0..gd as usize {
                out[i * gd as usize + g] = nodes[p * gd as usize + g];
            }
        }
    }
}

/// The family code of `family` in the agreement: 0 equispaced, 1 GLL.
fn family_code(family: NodeFamily) -> u32 {
    match family {
        NodeFamily::Equispaced => 0,
        NodeFamily::Gll => 1,
    }
}

/// The agreed (degree, family) of a child block of type `child` whose children have parent
/// cells in the parent blocks `present` (with their elements `elements`, by parent block):
/// the same element on every rank, also where a rank holds no child of the block.
/// Collective.
///
/// # Errors
///
/// Agreed: `UnsupportedDegree` if the parent cells differ in degree, else
/// `UnsupportedNodeFamily` if they differ in node family.
pub(crate) fn agreed_element(
    comm: &dyn CommDyn,
    child: CellType,
    elements: &[(u32, NodeFamily)],
    present: &[bool],
) -> Result<(u32, NodeFamily)> {
    let mut lo = (u32::MAX, u32::MAX);
    let mut hi = (0u32, 0u32);
    for (&(degree, family), _) in elements.iter().zip(present).filter(|(_, p)| **p) {
        let f = family_code(family);
        lo = (lo.0.min(degree), lo.1.min(f));
        hi = (hi.0.max(degree), hi.1.max(f));
    }
    let lo = (comm.min(lo.0), comm.min(lo.1));
    let hi = (comm.max(hi.0), comm.max(hi.1));
    if lo.0 != hi.0 {
        return Err(GridError::UnsupportedDegree {
            cell: cell_name(child),
            degree: hi.0,
        });
    }
    if lo.1 != hi.1 {
        return Err(GridError::UnsupportedNodeFamily {
            cell: cell_name(child),
            family: "mixed",
        });
    }
    // A block with no children anywhere does not exist (ADR-0039, item 10), so some rank
    // holds a parent cell and the agreed values are real.
    let family = if lo.1 == 1 {
        NodeFamily::Gll
    } else {
        NodeFamily::Equispaced
    };
    Ok((lo.0, family))
}

/// The child geometry of `blocks` (child cell type, degree, family and dofmap rows in parent
/// node indices) over the parent nodes `nodes` (`num_nodes` local nodes): compacted nodes in
/// parent local order, gathered bitwise, with the parent's `origin` and a one-rank node map;
/// also `node_to_parent`. Collective (index map).
pub(crate) fn child_geometry<F, CTD, GD>(
    comm: &dyn CommDyn,
    nodes: &DeviceBuffer<F>,
    num_nodes: usize,
    origin: GD::Array<f64>,
    blocks: Vec<(CellType, u32, NodeFamily, DeviceStrided<u32>)>,
) -> Result<(Geometry<F, CTD, GD>, DeviceBuffer<u32>)>
where
    F: Real,
    CTD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let client = nodes.client();
    let rows: Vec<&DeviceStrided<u32>> = blocks.iter().map(|b| &b.3).collect();
    let (node_to_parent, parent_to_child) = compact(client, num_nodes, &rows)?;
    let n = node_to_parent.len();
    let gd = GD::N;
    let child_nodes =
        DeviceBuffer::<F>::uninit(client, n.checked_mul(gd).ok_or(GridError::TooLarge)?)?;
    if n > 0 {
        fit(child_nodes.len())?;
        let launch = plan(client, n)?;
        debug_assert_bindings(client, 3);
        // SAFETY: the kernel reads map[i] for i < n = map.len() and nodes[map[i] · gd + g],
        // within the parent's num_nodes · gd coordinates since map holds parent local nodes; it
        // writes out[i · gd + g] < n · gd, its length.
        unsafe {
            gather_nodes_kernel::launch_unchecked::<F>(
                client,
                launch.cube_count,
                launch.cube_dim,
                nodes.arg(),
                node_to_parent.arg(),
                child_nodes.arg(),
                fit(n)?,
                fit(gd)?,
                launch.items_per_unit,
            );
        }
    }
    let geometry_blocks = blocks
        .into_iter()
        .map(|(cell, degree, family, rows)| {
            let cell = CTD::Cell::try_from(cell).map_err(|e| GridError::WrongCellDim {
                found: cell_name(e.found),
                expected_tdim: e.expected_tdim,
            })?;
            Ok(GeometryBlock {
                element: CoordinateElement {
                    cell,
                    degree,
                    family,
                },
                dofmap: DeviceStrided {
                    data: gather(&parent_to_child, &rows.data)?,
                    stride: rows.stride,
                },
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let node_map = IndexMap::new(comm, n)?;
    let geometry = Geometry::new(geometry_blocks, child_nodes, node_map, origin)?;
    Ok((geometry, node_to_parent))
}
