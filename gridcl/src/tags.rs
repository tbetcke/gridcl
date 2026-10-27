//! `MeshTags`, `TagRegistry` and `Codim` (docs/design.md, "Tags, regions and submeshes";
//! ADR-0022, item 3).

use std::collections::HashMap;

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, debug_assert_bindings, exclusive_scan,
    fill,
};

/// Tags of the entities of one dimension, on the device: entries sorted by (index, value), an
/// index repeating only with distinct values (an element in several physical groups). Indices
/// are flat entity indices; cells (dim = tdim) are numbered over the cell blocks in block
/// order.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
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

/// Codimension addressing (NGSolve): a dimension relative to D_TOP, so that region code works
/// for curve, surface and volume meshes alike.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// assert_eq!(Codim::Bnd.dim(3), Some(2));
/// assert_eq!(Codim::BBnd.dim(2), Some(0));
/// assert_eq!(Codim::BBBnd.dim(2), None);
/// ```
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
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
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

/// The tags of a mesh: per dimension one physical and one elementary [`MeshTags`], the name
/// table (name, dimension) → value, and each physical group's size, recorded on the host when
/// the tags were read. `Default` is empty.
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let tags = TagRegistry::default();
/// assert_eq!(tags.names().count(), 0);
/// assert!(tags.physical(1).is_none());
/// assert!(matches!(tags.named(Codim::Bnd, "wall"), Err(GridError::UnknownTag { .. })));
/// ```
#[derive(Default)]
pub struct TagRegistry {
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
    /// `names` ((name, dimension, value)).
    pub(crate) fn new(tdim: usize, names: Vec<(String, usize, i32)>) -> Self {
        TagRegistry {
            tdim,
            physical: (0..=tdim).map(|_| None).collect(),
            elementary: (0..=tdim).map(|_| None).collect(),
            names,
            sizes: HashMap::new(),
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
            tdim: self.tdim,
            physical: clone(&self.physical)?,
            elementary: clone(&self.elementary)?,
            names: self.names.clone(),
            sizes: self.sizes.clone(),
        })
    }

    /// The physical names: (name, dimension, value).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert!(TagRegistry::default().names().next().is_none());
    /// ```
    pub fn names(&self) -> impl Iterator<Item = (&str, usize, i32)> {
        self.names.iter().map(|(n, d, v)| (n.as_str(), *d, *v))
    }

    /// The value of the physical group `name` in dimension `dim`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert_eq!(TagRegistry::default().lookup(2, "domain"), None);
    /// ```
    pub fn lookup(&self, dim: usize, name: &str) -> Option<i32> {
        self.names
            .iter()
            .find(|(n, d, _)| *d == dim && n == name)
            .map(|e| e.2)
    }

    /// The physical tags of dimension `dim`, if any.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert!(TagRegistry::default().physical(0).is_none());
    /// ```
    pub fn physical(&self, dim: usize) -> Option<&MeshTags> {
        self.physical.get(dim)?.as_ref()
    }

    /// The elementary (entity) tags of dimension `dim`, if any.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert!(TagRegistry::default().elementary(3).is_none());
    /// ```
    pub fn elementary(&self, dim: usize) -> Option<&MeshTags> {
        self.elementary.get(dim)?.as_ref()
    }

    /// The number of entities in physical group `value` of dimension `dim`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// assert_eq!(TagRegistry::default().group_size(1, 7), None);
    /// ```
    pub fn group_size(&self, dim: usize, value: i32) -> Option<usize> {
        self.sizes.get(&(dim, value)).copied()
    }

    /// The entities of physical group `value` of dimension `dim`, selected on the device into
    /// arrays of the recorded group size (no count read), in index order; every value is
    /// `value`.
    ///
    /// # Errors
    ///
    /// `UnknownTag` (the value as the name) if dimension `dim` has no such group; allocation
    /// and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let e = TagRegistry::default().group(2, 5).err().unwrap();
    /// assert_eq!(e.to_string(), "no tag named \"5\" in dimension 2");
    /// ```
    pub fn group(&self, dim: usize, value: i32) -> Result<MeshTags> {
        let unknown = || GridError::UnknownTag {
            name: value.to_string(),
            dim,
        };
        let size = self.group_size(dim, value).ok_or_else(unknown)?;
        let tags = self.physical(dim).ok_or_else(unknown)?;
        let client = tags.indices.client();
        let mut values = DeviceBuffer::<i32>::uninit(client, size)?;
        fill(&mut values, value)?;
        let out = DeviceBuffer::<u32>::uninit(client, size)?;
        let len = tags.indices.len();
        if size == 0 || len == 0 {
            return Ok(MeshTags {
                dim,
                indices: out,
                values,
            });
        }
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

    /// The entities of the physical group `name` in the dimension `codim` addresses, as
    /// [`group`](TagRegistry::group) selects them.
    ///
    /// # Errors
    ///
    /// `UnknownTag` if that dimension has no group of this name (with `dim` the mesh's tdim
    /// when `codim` lies below dimension 0); allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let e = TagRegistry::default().named(Codim::Vol, "domain").err().unwrap();
    /// assert!(matches!(e, GridError::UnknownTag { dim: 0, .. }));
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
        gridcl_testing::test_client()
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
            2,
            vec![
                ("left".into(), 1, 1),
                ("walls".into(), 1, 2),
                ("left".into(), 0, 1),
                ("domain".into(), 2, 7),
                ("nothing".into(), 1, 9),
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
        assert_eq!(r.names().count(), 5);
        assert!(r.elementary(1).is_some() && r.elementary(0).is_none());
        // "domain" has a name but no tags of dimension 2.
        for (result, name, dim) in [
            (r.named(Codim::Vol, "left"), "left", 2),
            (r.named(Codim::Bnd, "domain"), "domain", 1),
            (r.named(Codim::Vol, "domain"), "7", 2),
            (r.named(Codim::BBBnd, "left"), "left", 2),
            (r.group(1, 4), "4", 1),
            (r.group(3, 1), "1", 3),
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
        let mut r = TagRegistry::new(3, vec![]);
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
        if gridcl_testing::supports::<f64>(&client) {
            keeps_tags::<f64>(&client);
        }
    }
}
