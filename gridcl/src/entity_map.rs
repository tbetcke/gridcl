//! `EntityMap`: the map from the cells, vertices and nodes of a submesh to those of its parent
//! (docs/design.md, "Tags, regions and submeshes"; ADR-0039, item 1).

use gridcl_device::{DeviceBuffer, Result};

/// The map from a submesh to its parent, on the device, for every local child cell, vertex and
/// node, owned or ghost (ADR-0039, item 1).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
///
/// let client = gridcl_testing::test_client();
/// let mut mesh: VolumeMesh<f32> = gridcl_testing::cube_box(Cell3::Tetrahedron, 1, 1).build(&client)?;
/// mesh.create_entities(2)?;
/// let (surface, map) = mesh.boundary()?;
/// // Two triangles per face of the cube; the eight corners.
/// assert_eq!(map.child_to_parent().len(), 12);
/// assert_eq!(map.vertex_to_parent().len(), surface.vertex_index_map().num_local());
/// # Ok::<(), GridError>(())
/// ```
pub struct EntityMap {
    parent_dim: usize,
    child_to_parent: DeviceBuffer<u32>,
    parents: DeviceBuffer<u32>,
    children: DeviceBuffer<u32>,
    vertex_to_parent: DeviceBuffer<u32>,
    node_to_parent: DeviceBuffer<u32>,
    /// Facet children: the flat parent cell and the local facet index of each child cell, for
    /// the child geometry and flags; crate-private (ADR-0039, item 1).
    facet_of: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
}

impl std::fmt::Debug for EntityMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EntityMap")
            .field("parent_dim", &self.parent_dim)
            .field("child_cells", &self.child_to_parent.len())
            .field("child_vertices", &self.vertex_to_parent.len())
            .field("child_nodes", &self.node_to_parent.len())
            .finish()
    }
}

/// The parts of an [`EntityMap`], for submesh creation.
pub(crate) struct EntityMapParts {
    pub(crate) parent_dim: usize,
    pub(crate) child_to_parent: DeviceBuffer<u32>,
    pub(crate) parent_to_child: (DeviceBuffer<u32>, DeviceBuffer<u32>),
    pub(crate) vertex_to_parent: DeviceBuffer<u32>,
    pub(crate) node_to_parent: DeviceBuffer<u32>,
    pub(crate) facet_of: Option<(DeviceBuffer<u32>, DeviceBuffer<u32>)>,
}

impl EntityMap {
    /// The map of its parts.
    pub(crate) fn from_parts(p: EntityMapParts) -> Self {
        let (parents, children) = p.parent_to_child;
        EntityMap {
            parent_dim: p.parent_dim,
            child_to_parent: p.child_to_parent,
            parents,
            children,
            vertex_to_parent: p.vertex_to_parent,
            node_to_parent: p.node_to_parent,
            facet_of: p.facet_of,
        }
    }

    /// The dimension of the parent entities the child cells come from: tdim − 1 for `boundary`
    /// and `facet_submesh`, 1 for `edge_submesh`, tdim for `cell_submesh`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// // Boundary edges of the square.
    /// assert_eq!(map.parent_dim(), 1);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn parent_dim(&self) -> usize {
        self.parent_dim
    }

    /// The flat local parent entity of each local child cell, in flat child order (over the
    /// child's cell blocks in block order).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// // Eight boundary edges of the twelve.
    /// let parents = map.child_to_parent().to_host()?;
    /// assert_eq!(parents.len(), 8);
    /// assert!(parents.iter().all(|&e| e < 12));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn child_to_parent(&self) -> &DeviceBuffer<u32> {
        &self.child_to_parent
    }

    /// The sparse inverse of [`child_to_parent`](EntityMap::child_to_parent): the parent flat
    /// indices that have a child, ascending, and the flat child cell of each.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// let (parents, children) = map.parent_to_child();
    /// let (parents, children) = (parents.to_host()?, children.to_host()?);
    /// let c2p = map.child_to_parent().to_host()?;
    /// assert!(parents.windows(2).all(|w| w[0] < w[1]));
    /// assert!(parents.iter().zip(&children).all(|(&p, &c)| c2p[c as usize] == p));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn parent_to_child(&self) -> (&DeviceBuffer<u32>, &DeviceBuffer<u32>) {
        (&self.parents, &self.children)
    }

    /// The parent local vertex of each child local vertex.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// // The eight vertices on the boundary of the nine.
    /// assert_eq!(map.vertex_to_parent().len(), 8);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn vertex_to_parent(&self) -> &DeviceBuffer<u32> {
        &self.vertex_to_parent
    }

    /// The parent geometry node of each child geometry node.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// // Degree 1: the nodes are the vertices.
    /// assert_eq!(map.node_to_parent().len(), 8);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn node_to_parent(&self) -> &DeviceBuffer<u32> {
        &self.node_to_parent
    }

    /// A deep copy of every device array.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let client = gridcl_testing::test_client();
    /// let mut mesh: PlanarMesh<f32> = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 1).build(&client)?;
    /// mesh.create_entities(1)?;
    /// let (_, map) = mesh.boundary()?;
    /// let copy = map.try_clone()?;
    /// assert_eq!(copy.child_to_parent().to_host()?, map.child_to_parent().to_host()?);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn try_clone(&self) -> Result<Self> {
        Ok(EntityMap {
            parent_dim: self.parent_dim,
            child_to_parent: self.child_to_parent.deep_clone()?,
            parents: self.parents.deep_clone()?,
            children: self.children.deep_clone()?,
            vertex_to_parent: self.vertex_to_parent.deep_clone()?,
            node_to_parent: self.node_to_parent.deep_clone()?,
            facet_of: match &self.facet_of {
                Some((c, f)) => Some((c.deep_clone()?, f.deep_clone()?)),
                None => None,
            },
        })
    }
}
