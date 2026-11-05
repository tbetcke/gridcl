//! The tags of a child (ADR-0039, item 10): the parent's physical and elementary tags of the
//! dimension the child cells come from, restricted to the parents of child cells and
//! renumbered as child cell tags, with those rows of the name table.
//!
//! Child cells follow their parents' flat order (`select`), so the restricted entries stay
//! sorted by (index, value).

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, ReduceOp, Result, debug_assert_bindings, exclusive_scan, fill, reduce, scatter,
};

use super::{fit, plan};
use crate::{EntityMap, MeshTags, TagRegistry};

/// `keep[i] = 1` where `child[indices[i]]` is a child cell (not `u32::MAX`), for i < len;
/// indices at or past `num_parents` are dropped.
#[cube(launch_unchecked)]
fn keep_kernel(
    indices: &[u32],
    child: &[u32],
    keep: &mut [u32],
    len: u32,
    num_parents: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let p = indices[i];
            let mut k = 0u32;
            if p < num_parents {
                k = u32::cast_from(child[p as usize] != u32::MAX);
            }
            keep[i] = k;
        }
    }
}

/// Kept entry i goes to `pos[i]` with index `child[indices[i]]` and value `values[i]`.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn restrict_kernel(
    indices: &[u32],
    values: &[i32],
    child: &[u32],
    pos: &[u32],
    out_indices: &mut [u32],
    out_values: &mut [i32],
    len: u32,
    num_parents: u32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            let p = indices[i];
            if p < num_parents {
                let c = child[p as usize];
                if c != u32::MAX {
                    let o = pos[i] as usize;
                    out_indices[o] = c;
                    out_values[o] = values[i];
                }
            }
        }
    }
}

/// `flags[i] = keep[i] · (values[i] == value)`.
#[cube(launch_unchecked)]
fn group_kernel(
    keep: &[u32],
    values: &[i32],
    flags: &mut [u32],
    len: u32,
    value: i32,
    items_per_unit: u32,
) {
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = first + it;
        if i < len as usize {
            flags[i] = keep[i] * u32::cast_from(values[i] == value);
        }
    }
}

/// `tags` restricted to the parents of child cells through the dense parent → child map
/// `child` (`u32::MAX` for parents without a child), as child cell tags of dimension
/// `child_dim`, with the entry count of each value in `values` (one count read each, the
/// per-group count reads of the compaction).
fn restrict(
    tags: &MeshTags,
    child: &DeviceBuffer<u32>,
    child_dim: usize,
    values: &[i32],
) -> Result<(MeshTags, Vec<(i32, usize)>)> {
    let client = tags.indices.client();
    let len = tags.indices.len();
    let num_parents = fit(child.len())?;
    let keep = DeviceBuffer::<u32>::zeros(client, len)?;
    if len > 0 {
        let launch = plan(client, len)?;
        debug_assert_bindings(client, 3);
        // SAFETY: entries i < len read indices[i] and write keep[i] (len words each), and read
        // child[p] only for p < num_parents = child.len().
        unsafe {
            keep_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                tags.indices.arg(),
                child.arg(),
                keep.arg(),
                fit(len)?,
                num_parents,
                launch.items_per_unit,
            );
        }
    }
    let pos = exclusive_scan(&keep)?;
    // The count read of the compaction (CLAUDE.md, ADR-0039 item 18).
    let total = if len > 0 {
        reduce(&keep, ReduceOp::Sum)?.to_host()?[0] as usize
    } else {
        0
    };
    let indices = DeviceBuffer::<u32>::uninit(client, total)?;
    let out_values = DeviceBuffer::<i32>::uninit(client, total)?;
    if total > 0 {
        let launch = plan(client, len)?;
        debug_assert_bindings(client, 6);
        // SAFETY: entries i < len read indices[i], values[i] and pos[i] (len and len + 1
        // words) and child[p] for p < num_parents; kept entries write at pos[i] < pos[len] =
        // total, the length of both outputs.
        unsafe {
            restrict_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                tags.indices.arg(),
                tags.values.arg(),
                child.arg(),
                pos.arg(),
                indices.arg(),
                out_values.arg(),
                fit(len)?,
                num_parents,
                launch.items_per_unit,
            );
        }
    }
    let mut sizes = Vec::with_capacity(values.len());
    for &value in values {
        let n = if total > 0 {
            let flags = DeviceBuffer::<u32>::uninit(client, len)?;
            let launch = plan(client, len)?;
            debug_assert_bindings(client, 3);
            // SAFETY: entries i < len read keep[i] and values[i] and write flags[i], all of
            // length len.
            unsafe {
                group_kernel::launch_unchecked(
                    client,
                    launch.cube_count,
                    launch.cube_dim,
                    keep.arg(),
                    tags.values.arg(),
                    flags.arg(),
                    fit(len)?,
                    value,
                    launch.items_per_unit,
                );
            }
            // The per-group count read of the compaction (CLAUDE.md, ADR-0039 item 18).
            reduce(&flags, ReduceOp::Sum)?.to_host()?[0] as usize
        } else {
            0
        };
        sizes.push((value, n));
    }
    Ok((
        MeshTags {
            dim: child_dim,
            indices,
            values: out_values,
        },
        sizes,
    ))
}

/// The child's tag registry: the parent's tags of dimension `map.parent_dim()` restricted to
/// the child cells (dimension `child_dim`), the name rows of that dimension and their group
/// sizes; `num_parents` is the number of local parent entities of that dimension.
pub(crate) fn child_tags(
    parent: &TagRegistry,
    map: &EntityMap,
    num_parents: usize,
    child_dim: usize,
    client: &cubecl::prelude::Client,
) -> Result<TagRegistry> {
    let pdim = map.parent_dim();
    let names: Vec<(String, usize, i32)> = parent
        .names()
        .filter(|&(_, d, _)| d == pdim)
        .map(|(n, _, v)| (n.to_string(), child_dim, v))
        .collect();
    let mut registry = TagRegistry::new(client, child_dim, names);
    let (physical, elementary) = (parent.physical(pdim), parent.elementary(pdim));
    if physical.is_none() && elementary.is_none() {
        return Ok(registry);
    }
    // The dense parent → child map.
    let mut child = DeviceBuffer::<u32>::uninit(client, num_parents)?;
    fill(&mut child, u32::MAX)?;
    let (parents, children) = map.parent_to_child();
    scatter(children, parents, &mut child)?;
    if let Some(t) = physical {
        let mut values: Vec<i32> = parent.group_values(pdim).into_iter().collect();
        values.sort_unstable();
        let (tags, sizes) = restrict(t, &child, child_dim, &values)?;
        registry.set_physical(tags, sizes);
    }
    if let Some(t) = elementary {
        let (tags, _) = restrict(t, &child, child_dim, &[])?;
        registry.set_elementary(tags);
    }
    Ok(registry)
}
