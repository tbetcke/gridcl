//! Matching of entity keys against the entities of one dimension (ADR-0022, item 4): the
//! entity keys are rebuilt from `EntityBlock::vertices`, put before the given keys, sorted
//! together (stable), grouped, and each given key takes the entity of its group, if any.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, GridError, KernelKind, LaunchPlan, Result, SortKeys, debug_assert_bindings, fill,
    pad_len, radix_sort, unique_by_key,
};
use gridcl_dims::{CellType, TopoDim};

use crate::Topology;

/// The combined keys: item i < `num_entities` is entity i's vertex row sorted ascending, item
/// `num_entities + t` is given key t, items up to the plane's padding are all ones; the
/// payload of each real item is its index. Scalar: one key per item.
///   vertices  [entity][words] local vertex indices
///   tags      the given keys, planar with plane `tag_plane`
///   keys      the combined keys, planar with plane `plane`
///   payload   one index per real key
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn combine_keys_kernel(
    vertices: &[u32],
    tags: &[u32],
    keys: &mut [u32],
    payload: &mut [u32],
    num_entities: u32,
    num_tags: u32,
    tag_plane: u32,
    plane: u32,
    items_per_unit: u32,
    #[comptime] words: usize,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    let total = (num_entities + num_tags) as usize;
    for it in 0..items_per_unit as usize {
        let i = begin + it;
        if i < plane as usize {
            let mut w = Array::<u32>::new(words);
            if i < num_entities as usize {
                #[unroll]
                for j in 0..words {
                    w[j] = vertices[i * words + j];
                }
                // Ascending, by a sorting network.
                #[unroll]
                for pass in 0..comptime!(words - 1) {
                    #[unroll]
                    for j in 0..comptime!(words - 1 - pass) {
                        let (x, y) = (w[j], w[j + 1]);
                        w[j] = min(x, y);
                        w[j + 1] = max(x, y);
                    }
                }
            } else if i < total {
                let t = i - num_entities as usize;
                #[unroll]
                for j in 0..words {
                    w[j] = tags[j * tag_plane as usize + t];
                }
            } else {
                #[unroll]
                for j in 0..words {
                    w[j] = u32::MAX;
                }
            }
            #[unroll]
            for j in 0..words {
                keys[j * plane as usize + i] = w[j];
            }
            if i < total {
                payload[i] = i as u32;
            }
        }
    }
}

/// For each sorted position j < `len` holding given key t = payload[j] − `num_entities`:
/// `out[t]` is `offset` + the entity at the head of its group (the stable sort puts an equal
/// entity key first), or `u32::MAX` if the group has no entity. Scalar: one position per item.
#[cube(launch_unchecked)]
#[allow(clippy::too_many_arguments)]
fn resolve_kernel(
    payload: &[u32],
    group: &[u32],
    first: &[u32],
    out: &mut [u32],
    len: u32,
    num_entities: u32,
    offset: u32,
    items_per_unit: u32,
) {
    let begin = ABSOLUTE_POS * items_per_unit as usize;
    for it in 0..items_per_unit as usize {
        let j = begin + it;
        if j < len as usize {
            let p = payload[j];
            if p >= num_entities {
                let head = first[group[j] as usize];
                // The sum only for an entity head, so a tag payload's head never wraps it.
                let mut r = offset;
                if head < num_entities {
                    r += head;
                } else {
                    r = u32::MAX;
                }
                out[(p - num_entities) as usize] = r;
            }
        }
    }
}

/// u32 of `x`, or `TooLarge`.
fn fit(x: usize) -> Result<u32> {
    u32::try_from(x).map_err(|_| GridError::TooLarge)
}

impl<TD: TopoDim> Topology<TD> {
    /// The flat index of the entity of dimension `dim` whose sorted local vertex indices are
    /// each key of `keys`, or `u32::MAX` where no entity matches. The word count of `keys`
    /// selects the entity type (1: points, 2: intervals, 3: triangles, 4: quadrilaterals); a
    /// type the dimension does not have matches nothing.
    ///
    /// Sorts the entity keys and `keys` together on the device and reads one count back
    /// (`unique_by_key`).
    ///
    /// # Errors
    ///
    /// `EntitiesNotCreated(dim)` if the entities of `dim` have not been created; `TooLarge` if
    /// the keys and entities together reach 2^32; allocation and device errors.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{DeviceBuffer, DeviceStrided, SortKeys};
    /// use gridcl_dims::{Cell2, D2};
    /// use gridcl_parallel::{IndexMap, Serial};
    /// use gridcl_topology::{CellBlock, Topology, encode_origin_ids};
    ///
    /// let client = gridcl_testing::test_client();
    /// // Two triangles sharing the edge (1, 2).
    /// let block = CellBlock::<D2>::new(
    ///     Cell2::Triangle,
    ///     IndexMap::new(&Serial, 2).unwrap(),
    ///     DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1, 2, 3, 2, 1]).unwrap(), stride: 3 },
    ///     DeviceBuffer::from_host(&client, &encode_origin_ids(&[0, 1])).unwrap(),
    ///     DeviceBuffer::from_host(&client, &[0u32, 0]).unwrap(),
    /// );
    /// let mut topology = Topology::new(&client, vec![block], IndexMap::new(&Serial, 4).unwrap(), &[10, 11, 12, 13]).unwrap();
    /// topology.create_entities(&Serial, 1).unwrap();
    /// // The shared edge, an edge given twice, and a pair of vertices without an edge.
    /// let keys = SortKeys::from_host(&client, &[1, 2, 0, 1, 1, 2, 0, 3], 2).unwrap();
    /// let found = topology.match_entities(1, &keys).unwrap().to_host().unwrap();
    /// let edges = topology.entity_blocks(1).unwrap()[0].vertices.data.to_host().unwrap();
    /// let edge = |i: u32| { let mut e = [edges[2 * i as usize], edges[2 * i as usize + 1]]; e.sort(); e };
    /// assert_eq!((edge(found[0]), edge(found[1])), ([1, 2], [0, 1]));
    /// assert_eq!(found[2], found[0]);
    /// assert_eq!(found[3], u32::MAX);
    /// ```
    pub fn match_entities(&self, dim: usize, keys: &SortKeys) -> Result<DeviceBuffer<u32>> {
        let client = self.client();
        let blocks = self.entity_blocks(dim)?;
        let n = keys.len();
        let mut out = DeviceBuffer::<u32>::uninit(client, n)?;
        if n == 0 {
            return Ok(out);
        }
        let words = keys.num_words();
        let entity = match words {
            1 => CellType::Point,
            2 => CellType::Interval,
            3 => CellType::Triangle,
            _ => CellType::Quadrilateral,
        };
        let Some(block) = blocks.iter().find(|b| b.cell == entity) else {
            fill(&mut out, u32::MAX)?;
            return Ok(out);
        };
        let num_entities = block.index_map.num_local();
        debug_assert_eq!(block.vertices.stride as usize, words);
        debug_assert_eq!(block.vertices.data.len(), num_entities * words);
        let total = num_entities.checked_add(n).ok_or(GridError::TooLarge)?;
        fit(total)?;
        let plane = pad_len(total)?;
        let mut combined = SortKeys::uninit(client, total, words)?;
        fit(combined.words().len())?;
        let payload = DeviceBuffer::<u32>::uninit(client, total)?;
        let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, plane, 4, 1, 1)?;
        debug_assert_bindings(client, 4);
        // SAFETY: items i < plane = pad_len(total) read vertex row i (i < num_entities, rows of
        // `words` entries, data.len() = num_entities · words) or word j of given key
        // t = i − num_entities < n at j · pad_len(n) + t, within `keys`' words planes of
        // pad_len(n); they write keys[j · plane + i] < words · plane, the length of the
        // combined planar buffer, and payload[i] only for i < total = payload.len().
        unsafe {
            combine_keys_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                block.vertices.data.arg(),
                keys.words().arg(),
                combined.words_mut().arg(),
                payload.arg(),
                fit(num_entities)?,
                fit(n)?,
                fit(pad_len(n)?)?,
                fit(plane)?,
                launch.items_per_unit,
                words,
            );
        }
        let (sorted, sorted_payload) = radix_sort(&combined, &payload)?;
        let unique = unique_by_key(&sorted, &sorted_payload)?; // the count read
        let launch = LaunchPlan::elementwise(client, KernelKind::Scalar, total, 4, 1, 1)?;
        debug_assert_bindings(client, 4);
        // SAFETY: positions j < total read sorted_payload[j] and group[j] (both of length
        // total) and first[group[j]], a group index below the group count = first.len(). The
        // payload is a permutation of 0..total, so each given key t = p − num_entities < n is
        // written exactly once, within `out` (n words).
        unsafe {
            resolve_kernel::launch_unchecked(
                client,
                launch.cube_count,
                launch.cube_dim,
                sorted_payload.arg(),
                unique.group.arg(),
                unique.first.arg(),
                out.arg(),
                fit(total)?,
                fit(num_entities)?,
                block.offset,
                launch.items_per_unit,
            );
        }
        Ok(out)
    }
}
