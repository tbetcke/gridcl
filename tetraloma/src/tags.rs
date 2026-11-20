//! `MeshTags`, `TagRegistry` and `Codim` (docs/design.md, "Tags, regions and submeshes";
//! ADR-0022, item 3).

use std::collections::HashMap;

use cubecl::prelude::*;
use tetraloma_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, debug_assert_bindings, exclusive_scan,
    fill,
};

// Design: "Tags, regions and submeshes"; ADR-0022, item 3.
/// Tags of the entities of one dimension, on the device: pairs of an entity index and a tag
/// value.
///
/// The entries are sorted by index and then by value. An index repeats only with distinct
/// values, when an entity belongs to several physical groups. Indices are local entity indices
/// of dimension `dim`; cells (`dim` = TD) are numbered over the cell blocks, block after block.
/// [`TagRegistry::group`] and [`TagRegistry::named`] return one group as a `MeshTags`, and the
/// submesh methods of [`Mesh`](crate::Mesh) take one.
///
/// [Tags and submeshes] explains tags and physical groups.
///
/// [Tags and submeshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // Entity 3 in groups 1 and 2, entity 5 in group 1.
/// let tags = MeshTags {
///     dim: 1,
///     indices: DeviceBuffer::from_host(&client, &[3u32, 3, 5])?,
///     values: DeviceBuffer::from_host(&client, &[1i32, 2, 1])?,
/// };
/// assert_eq!(tags.indices.len(), tags.values.len());
/// # Ok::<(), GridError>(())
/// ```
pub struct MeshTags {
    /// The topological dimension of the tagged entities.
    pub dim: usize,
    /// The flat local entity index of each entry.
    pub indices: DeviceBuffer<u32>,
    /// The tag value of each entry.
    pub values: DeviceBuffer<i32>,
}

impl MeshTags {
    /// A deep copy of both device arrays.
    pub(crate) fn try_clone(&self) -> Result<Self> {
        Ok(MeshTags {
            dim: self.dim,
            indices: self.indices.deep_clone()?,
            values: self.values.deep_clone()?,
        })
    }
}

/// A dimension counted down from the mesh's topological dimension TD, as in NGSolve, so that
/// code that selects regions works for curve, surface and volume meshes alike.
///
/// `Codim::Vol` addresses the cells, `Codim::Bnd` their facets, `Codim::BBnd` the facets of
/// those, and `Codim::BBBnd` one dimension further. [Regions] selects tagged cells and facets
/// with it.
///
/// [Regions]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#regions
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// assert_eq!(Codim::Bnd.dim(3), Some(2));
/// assert_eq!(Codim::BBnd.dim(2), Some(0));
/// assert_eq!(Codim::BBBnd.dim(2), None);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codim {
    /// Dimension TD: the cells.
    Vol,
    /// Dimension TD − 1: the facets.
    Bnd,
    /// Dimension TD − 2.
    BBnd,
    /// Dimension TD − 3.
    BBBnd,
}

impl Codim {
    /// The dimension this codimension addresses in a mesh of topological dimension `tdim`, or
    /// `None` if it lies below 0.
    ///
    /// [Regions] uses codimensions to select tagged entities.
    ///
    /// [Regions]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#regions
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// assert_eq!(Codim::Vol.dim(2), Some(2));
    /// assert_eq!(Codim::BBBnd.dim(3), Some(0));
    /// assert_eq!(Codim::Bnd.dim(0), None);
    /// ```
    pub fn dim(self, tdim: usize) -> Option<usize> {
        let codim = match self {
            Codim::Vol => 0,
            Codim::Bnd => 1,
            Codim::BBnd => 2,
            Codim::BBBnd => 3,
        };
        tdim.checked_sub(codim)
    }
}

// Design: "Tags, regions and submeshes"; ADR-0022, item 3.
/// The tags of a mesh: per dimension one physical and one elementary [`MeshTags`], the name
/// table from (name, dimension) to value, and the size of each physical group.
///
/// The sizes are recorded on the host when the tags are read, so selecting a group needs no
/// read from the device. A named group without entities has size 0. `Default` is empty.
/// [`Mesh::tags`](crate::Mesh::tags) returns the registry of a mesh, and
/// [Tags and submeshes] shows how to use it.
///
/// [Tags and submeshes]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// let tags = TagRegistry::default();
/// assert_eq!(tags.names().count(), 0);
/// assert!(tags.physical(1).is_none());
/// assert!(matches!(tags.named(Codim::Bnd, "wall"), Err(GridError::UnknownTag { .. })));
/// ```
#[derive(Default)]
pub struct TagRegistry {
    /// The client of the mesh, for empty selections; `None` only in `Default`.
    client: Option<Client>,
    /// The topological dimension of the mesh, for [`Codim`].
    tdim: usize,
    /// Physical tags by dimension.
    physical: Vec<Option<MeshTags>>,
    /// Elementary tags by dimension.
    elementary: Vec<Option<MeshTags>>,
    /// (name, dimension, value), in file order.
    names: Vec<(String, usize, i32)>,
    /// The entry count of each physical group, by (dimension, value).
    sizes: HashMap<(usize, i32), usize>,
}

/// flags[i] = 1 where values[i] == value, else 0. Scalar: one entry per item.
#[cube(launch_unchecked)]
fn flag_kernel(values: &[i32], flags: &mut [u32], len: u32, value: i32, items_per_unit: u32) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = begin + it;
        if i < len as usize {
            let mut f = 0u32;
            if values[i] == value {
                f = 1u32;
            }
            flags[i] = f;
        }
    }
}

/// out[offsets[i]] = indices[i] where values[i] == value, for offsets below `size`. Scalar.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn select_kernel(
    indices: &[u32],
    values: &[i32],
    offsets: &[u32],
    out: &mut [u32],
    len: u32,
    size: u32,
    value: i32,
    items_per_unit: u32,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let i = begin + it;
        if i < len as usize {
            let o = offsets[i];
            if values[i] == value && o < size {
                out[o as usize] = indices[i];
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

impl TagRegistry {
    /// An empty registry of a mesh of topological dimension `tdim` with the physical names
    /// `names` ((name, dimension, value)), each group up to dimension `tdim` of size 0 until
    /// set.
    pub(crate) fn new(client: &Client, tdim: usize, names: Vec<(String, usize, i32)>) -> Self {
        let sizes = names
            .iter()
            .filter(|&&(_, d, _)| d <= tdim)
            .map(|&(_, d, v)| ((d, v), 0))
            .collect();
        TagRegistry {
            client: Some(client.clone()),
            tdim,
            physical: (0..=tdim).map(|_| None).collect(),
            elementary: (0..=tdim).map(|_| None).collect(),
            names,
            sizes,
        }
    }

    /// Sets the physical tags of their dimension with each group's entry count.
    ///
    /// # Panics
    ///
    /// If the dimension exceeds tdim.
    pub(crate) fn set_physical(
        &mut self,
        tags: MeshTags,
        sizes: impl IntoIterator<Item = (i32, usize)>,
    ) {
        let dim = tags.dim;
        for (value, n) in sizes {
            self.sizes.insert((dim, value), n);
        }
        self.physical[dim] = Some(tags);
    }

    /// The values of the physical groups of dimension `dim` with a recorded size.
    pub(crate) fn group_values(&self, dim: usize) -> Vec<i32> {
        self.sizes
            .keys()
            .filter(|&&(d, _)| d == dim)
            .map(|&(_, v)| v)
            .collect()
    }

    /// Sets the elementary tags of their dimension.
    ///
    /// # Panics
    ///
    /// If the dimension exceeds tdim.
    pub(crate) fn set_elementary(&mut self, tags: MeshTags) {
        let dim = tags.dim;
        self.elementary[dim] = Some(tags);
    }
}

impl TagRegistry {
    /// A deep copy of every device array.
    pub(crate) fn try_clone(&self) -> Result<Self> {
        let clone = |v: &Vec<Option<MeshTags>>| {
            v.iter()
                .map(|t| t.as_ref().map(MeshTags::try_clone).transpose())
                .collect::<Result<Vec<_>>>()
        };
        Ok(TagRegistry {
            client: self.client.clone(),
            tdim: self.tdim,
            physical: clone(&self.physical)?,
            elementary: clone(&self.elementary)?,
            names: self.names.clone(),
            sizes: self.sizes.clone(),
        })
    }

    /// The allocated bytes of the physical and elementary tag arrays.
    pub(crate) fn footprint_bytes(&self) -> usize {
        self.physical
            .iter()
            .chain(&self.elementary)
            .flatten()
            .map(|t| t.indices.footprint_bytes() + t.values.footprint_bytes())
            .sum()
    }

    /// The physical names as triples (name, dimension, value), in the order of the file or of
    /// [`MeshBuilder::tag_names`](crate::MeshBuilder::tag_names).
    ///
    /// [Physical groups] lists the names of a file.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// let names: Vec<_> = tags.names().collect();
    /// assert_eq!(names.len(), 6);
    /// assert!(names.contains(&("interface", 1, 11)));
    /// // The same name in two dimensions is two groups.
    /// assert!(names.contains(&("boundary", 0, 20)) && names.contains(&("boundary", 1, 10)));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn names(&self) -> impl Iterator<Item = (&str, usize, i32)> {
        self.names.iter().map(|(n, d, v)| (n.as_str(), *d, *v))
    }

    /// The value of the physical group `name` in dimension `dim`, or `None` if dimension `dim`
    /// has no group of this name.
    ///
    /// [Physical groups] looks up a group by name.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// assert_eq!(tags.lookup(2, "triangles"), Some(1));
    /// assert_eq!(tags.lookup(1, "boundary"), Some(10));
    /// assert_eq!(tags.lookup(2, "boundary"), None);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn lookup(&self, dim: usize, name: &str) -> Option<i32> {
        self.names
            .iter()
            .find(|(n, d, _)| *d == dim && n == name)
            .map(|e| e.2)
    }

    /// The physical tags of dimension `dim`: every entity of that dimension in a physical
    /// group, once per group. `None` if the dimension has no physical tags.
    ///
    /// [Physical groups] explains the layout.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// // Every cell is in its region and in "all": 26 + 9 + 35 entries.
    /// let cells = tags.physical(2).unwrap();
    /// assert_eq!((cells.dim, cells.indices.len()), (2, 70));
    /// assert!(tags.physical(3).is_none());
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn physical(&self, dim: usize) -> Option<&MeshTags> {
        self.physical.get(dim)?.as_ref()
    }

    /// The elementary tags of dimension `dim`: the number of the geometric entity of the
    /// mesh generator that each tagged entity belongs to. `None` if the dimension has none.
    ///
    /// [Physical groups] describes elementary tags.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// // One elementary tag per cell.
    /// assert_eq!(tags.elementary(2).unwrap().indices.len(), 35);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn elementary(&self, dim: usize) -> Option<&MeshTags> {
        self.elementary.get(dim)?.as_ref()
    }

    /// The number of entities in the physical group `value` of dimension `dim`, or `None` if
    /// there is no such group. It reads a table on the host and launches no device work.
    ///
    /// [Physical groups] prints the size of every group.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// assert_eq!(tags.group_size(2, 1), Some(26)); // "triangles"
    /// assert_eq!(tags.group_size(1, 11), Some(3)); // "interface"
    /// assert_eq!(tags.group_size(1, 99), None);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn group_size(&self, dim: usize, value: i32) -> Option<usize> {
        self.sizes.get(&(dim, value)).copied()
    }

    /// The entities of the physical group `value` of dimension `dim`, in index order, with
    /// every value equal to `value`.
    ///
    /// The selection runs on the device, into arrays of the recorded group size, so nothing is
    /// read back to the host. [Physical groups] selects a group by value, and [Regions] passes
    /// one to a submesh.
    ///
    /// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
    /// [Regions]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#regions
    ///
    /// # Errors
    ///
    /// `UnknownTag` (with the value as the name) if dimension `dim` has no such group, named or
    /// tagged; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// let quadrilaterals = tags.group(2, 2)?;
    /// assert_eq!((quadrilaterals.dim, quadrilaterals.indices.len()), (2, 9));
    ///
    /// let e = tags.group(2, 5).err().unwrap();
    /// assert_eq!(e.to_string(), "no tag named \"5\" in dimension 2");
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn group(&self, dim: usize, value: i32) -> Result<MeshTags> {
        let unknown = || GridError::UnknownTag {
            name: value.to_string(),
            dim,
        };
        let size = self.group_size(dim, value).ok_or_else(unknown)?;
        let client = self.client.as_ref().ok_or_else(unknown)?;
        let mut values = DeviceBuffer::<i32>::uninit(client, size)?;
        fill(&mut values, value)?;
        let out = DeviceBuffer::<u32>::uninit(client, size)?;
        let tags = match self.physical(dim) {
            Some(tags) if size > 0 && tags.indices.len() > 0 => tags,
            _ if size == 0 => {
                return Ok(MeshTags {
                    dim,
                    indices: out,
                    values,
                });
            }
            _ => return Err(unknown()),
        };
        let len = tags.indices.len();
        let flags = DeviceBuffer::<u32>::uninit(client, len)?;
        let plan = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
        debug_assert_bindings(client, 2);
        // SAFETY: items i < len read values[i] and write flags[i], both of length len.
        unsafe {
            flag_kernel::launch_unchecked(
                client,
                plan.cube_count,
                plan.cube_dim,
                tags.values.arg(),
                flags.arg(),
                fit(len)?,
                value,
                plan.items_per_unit,
            );
        }
        let offsets = exclusive_scan(&flags)?;
        let plan = LaunchPlan::elementwise(client, KernelKind::Scalar, len, 4, 1, 1)?;
        debug_assert_bindings(client, 4);
        // SAFETY: items i < len read indices[i], values[i] (length len) and offsets[i] (an
        // exclusive scan of len flags: at least len entries), and write out[offsets[i]] only
        // when it is below size = out.len().
        unsafe {
            select_kernel::launch_unchecked(
                client,
                plan.cube_count,
                plan.cube_dim,
                tags.indices.arg(),
                tags.values.arg(),
                offsets.arg(),
                out.arg(),
                fit(len)?,
                fit(size)?,
                value,
                plan.items_per_unit,
            );
        }
        Ok(MeshTags {
            dim,
            indices: out,
            values,
        })
    }

    /// The entities of the physical group `name` in the dimension that `codim` addresses, as
    /// [`group`](TagRegistry::group) selects them.
    ///
    /// Because `codim` counts down from the mesh's dimension, the same call selects the
    /// boundary facets of a curve, a surface or a volume mesh. [Regions] selects regions by
    /// name.
    ///
    /// [Regions]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#regions
    ///
    /// # Errors
    ///
    /// `UnknownTag` if that dimension has no group of this name (with `dim` the mesh's TD
    /// when `codim` lies below dimension 0); a named group without entities is empty;
    /// allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// // path: the rectangle [0, 2] × [0, 1] with the physical groups "triangles" (26 cells),
    /// // "quadrilaterals" (9) and "all" (35), the edge groups "boundary" (18) and "interface"
    /// // (3), and the vertex group "boundary" (4); a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let mesh = PlanarMesh::<f32>::try_from(gmsh::read(&client, Serial, path, GmshOptions::default())?)?;
    /// let tags = mesh.tags();
    /// let interface = tags.named(Codim::Bnd, "interface")?;
    /// assert_eq!((interface.dim, interface.indices.len()), (1, 3));
    /// let corners = tags.named(Codim::BBnd, "boundary")?;
    /// assert_eq!((corners.dim, corners.indices.len()), (0, 4));
    ///
    /// // No cell group of this name.
    /// let e = tags.named(Codim::Vol, "interface").err().unwrap();
    /// assert!(matches!(e, GridError::UnknownTag { dim: 2, .. }));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn named(&self, codim: Codim, name: &str) -> Result<MeshTags> {
        let unknown = |dim| GridError::UnknownTag {
            name: name.to_owned(),
            dim,
        };
        let dim = codim.dim(self.tdim).ok_or_else(|| unknown(self.tdim))?;
        let value = self.lookup(dim, name).ok_or_else(|| unknown(dim))?;
        self.group(dim, value)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Cell2, CoordinateElement, D2, Mesh, MeshBuilder, NodeFamily, Real};

    fn client() -> Client {
        tetraloma_testing::test_client()
    }

    fn tags(client: &Client, dim: usize, entries: &[(u32, i32)]) -> MeshTags {
        let (indices, values): (Vec<u32>, Vec<i32>) = entries.iter().copied().unzip();
        MeshTags {
            dim,
            indices: DeviceBuffer::from_host(client, &indices).unwrap(),
            values: DeviceBuffer::from_host(client, &values).unwrap(),
        }
    }

    /// A planar registry: "left" in dimensions 1 and 0, overlapping groups 1 and 2 of
    /// dimension 1 (entities 0 and 5 in both), an empty group, and elementary tags.
    fn registry(client: &Client) -> TagRegistry {
        let mut r = TagRegistry::new(
            client,
            2,
            vec![
                ("left".into(), 1, 1),
                ("walls".into(), 1, 2),
                ("left".into(), 0, 1),
                ("domain".into(), 2, 7),
                ("nothing".into(), 1, 9),
                ("solid".into(), 3, 4),
            ],
        );
        r.set_physical(
            tags(client, 1, &[(0, 1), (0, 2), (3, 1), (5, 1), (5, 2), (8, 2)]),
            [(1, 3), (2, 3), (9, 0)],
        );
        r.set_physical(tags(client, 0, &[(2, 1)]), [(1, 1)]);
        r.set_elementary(tags(client, 1, &[(0, 4), (3, 4), (5, 4), (8, 6)]));
        r
    }

    fn host(t: &MeshTags) -> (usize, Vec<u32>, Vec<i32>) {
        (
            t.dim,
            t.indices.to_host().unwrap(),
            t.values.to_host().unwrap(),
        )
    }

    #[test]
    fn tags_group_and_named() {
        let client = client();
        let r = registry(&client);
        assert_eq!(
            host(&r.group(1, 1).unwrap()),
            (1, vec![0, 3, 5], vec![1; 3])
        );
        assert_eq!(
            host(&r.group(1, 2).unwrap()),
            (1, vec![0, 5, 8], vec![2; 3])
        );
        assert_eq!(host(&r.group(1, 9).unwrap()), (1, vec![], vec![]));
        assert_eq!(
            host(&r.named(Codim::Bnd, "left").unwrap()),
            (1, vec![0, 3, 5], vec![1; 3])
        );
        assert_eq!(
            host(&r.named(Codim::BBnd, "left").unwrap()),
            (0, vec![2], vec![1])
        );
        assert_eq!(r.lookup(1, "walls"), Some(2));
        assert_eq!(r.lookup(2, "walls"), None);
        assert_eq!(r.group_size(1, 2), Some(3));
        assert_eq!(r.names().count(), 6);
        assert!(r.elementary(1).is_some() && r.elementary(0).is_none());
        // "domain" has a name but no tags of dimension 2: an empty group.
        assert_eq!(r.group_size(2, 7), Some(0));
        assert_eq!(
            host(&r.named(Codim::Vol, "domain").unwrap()),
            (2, vec![], vec![])
        );
        for (result, name, dim) in [
            (r.named(Codim::Vol, "left"), "left", 2),
            (r.named(Codim::Bnd, "domain"), "domain", 1),
            (r.group(2, 8), "8", 2),
            (r.named(Codim::BBBnd, "left"), "left", 2),
            (r.group(1, 4), "4", 1),
            (r.group(3, 1), "1", 3),
            // A name above tdim has no group.
            (r.group(3, 4), "4", 3),
        ] {
            match result {
                Err(GridError::UnknownTag { name: n, dim: d }) => {
                    assert_eq!((n.as_str(), d), (name, dim))
                }
                Err(e) => panic!("{name}: {e}"),
                Ok(_) => panic!("{name}: expected UnknownTag"),
            }
        }
    }

    #[test]
    fn tags_codim_dim() {
        let all = [Codim::Vol, Codim::Bnd, Codim::BBnd, Codim::BBBnd];
        for tdim in 0..=3 {
            let dims: Vec<Option<usize>> = all.iter().map(|c| c.dim(tdim)).collect();
            let want: Vec<Option<usize>> = (0..4).map(|k| tdim.checked_sub(k)).collect();
            assert_eq!(dims, want);
        }
    }

    /// A group of 10^5 + 3 entries out of three times as many, against a host selection.
    #[test]
    fn tags_large_group() {
        let client = client();
        let n = 100_003u32;
        let entries: Vec<(u32, i32)> = (0..n)
            .flat_map(|i| [(i, 1 + (i % 3) as i32), (i, 10 + (i % 2) as i32)])
            .collect();
        let mut r = TagRegistry::new(&client, 3, vec![]);
        let size = |v| entries.iter().filter(|e| e.1 == v).count();
        r.set_physical(tags(&client, 2, &entries), [(2, size(2)), (11, size(11))]);
        for v in [2, 11] {
            let want: Vec<u32> = entries.iter().filter(|e| e.1 == v).map(|e| e.0).collect();
            let (_, got, values) = host(&r.group(2, v).unwrap());
            assert_eq!(got, want, "group {v}");
            assert!(values.iter().all(|&x| x == v));
        }
    }

    /// Two unit triangles, as a mesh of `F`.
    fn mesh<F: Real>(client: &Client) -> Mesh<F, D2, D2> {
        let tri = CoordinateElement::<D2> {
            cell: Cell2::Triangle,
            degree: 1,
            family: NodeFamily::Equispaced,
        };
        MeshBuilder::new(client)
            .nodes(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0], None)
            .cell_block(tri, &[0, 1, 2, 3, 2, 1], None)
            .build()
            .unwrap()
    }

    fn keeps_tags<F: Real>(client: &Client) {
        let mut m = mesh::<F>(client);
        assert!(m.tags().physical(1).is_none());
        // A builder mesh's registry knows its tdim: Bnd addresses dimension 1.
        assert!(matches!(
            m.tags().named(Codim::Bnd, "wall"),
            Err(GridError::UnknownTag { dim: 1, .. })
        ));
        m.set_tags(registry(client));
        let copy = m.try_clone().unwrap();
        let (a, b) = (m.tags(), copy.tags());
        assert_eq!(host(a.physical(1).unwrap()), host(b.physical(1).unwrap()));
        assert_eq!(
            host(a.elementary(1).unwrap()),
            host(b.elementary(1).unwrap())
        );
        assert_eq!(a.names().collect::<Vec<_>>(), b.names().collect::<Vec<_>>());
        assert_eq!(b.group_size(1, 1), Some(3));
        assert_eq!(host(&b.named(Codim::Bnd, "walls").unwrap()).1, [0, 5, 8]);
        // A deep copy: overwriting the original leaves the clone as it was.
        let mut original = registry(client);
        let clone = original.try_clone().unwrap();
        let indices = &mut original.physical[1].as_mut().unwrap().indices;
        fill(indices, 0).unwrap();
        assert_eq!(host(clone.physical(1).unwrap()).1, [0, 0, 3, 5, 5, 8]);
    }

    #[test]
    fn tags_try_clone_keeps_tags() {
        let client = client();
        keeps_tags::<f32>(&client);
        if tetraloma_testing::supports::<f64>(&client) {
            keeps_tags::<f64>(&client);
        }
    }
}
