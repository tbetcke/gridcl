//! The file model: the sections of an MSH 4.1 file as stored.

/// One entry of `$PhysicalNames`: the name of a physical group. tetraloma's reader turns physical
/// groups into named tags; see [Physical groups].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $PhysicalNames\n1\n2 7 \"domain\"\n$EndPhysicalNames\n")?;
/// let name = &file.physical_names[0];
/// assert_eq!((name.dim, name.tag, name.name.as_str()), (2, 7, "domain"));
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Physical groups]: https://tetracl.github.io/tetraloma/tutorials/tags-submeshes.html#physical-groups
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalName {
    /// The dimension of the physical group.
    pub dim: usize,
    /// The physical tag.
    pub tag: i32,
    /// The name, without quotes.
    pub name: String,
}

/// A geometric entity of `$Entities` (or the entity part of a `$PartitionedEntities` entry): a
/// point, curve, surface or volume of the CAD model, with its bounding box, physical groups and
/// boundary. See [the Gmsh format].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// // Two points and the curve between them, in physical group 3.
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Entities\n2 1 0 0\n1 0 0 0 0\n2 1 0 0 0\n1 0 0 0 1 0 0 1 3 2 1 -2\n$EndEntities\n")?;
/// let curve = &file.entities.unwrap().curves[0];
/// assert_eq!((curve.tag, curve.min, curve.max), (1, [0.0; 3], [1.0, 0.0, 0.0]));
/// assert_eq!((&curve.physical_tags[..], &curve.boundary[..]), (&[3][..], &[1, -2][..]));
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [the Gmsh format]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh
#[derive(Clone, Debug, PartialEq)]
pub struct Entity {
    /// The entity tag.
    pub tag: i32,
    /// The bounding box's lower corner; a point's coordinates for dimension 0.
    pub min: [f64; 3],
    /// The bounding box's upper corner; equal to `min` for dimension 0.
    pub max: [f64; 3],
    /// The physical tags of the entity.
    pub physical_tags: Vec<i32>,
    /// The signed tags of the bounding entities of one dimension lower; empty for points.
    pub boundary: Vec<i32>,
}

/// `$Entities`: the geometric entities by dimension. See [the Gmsh format].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Entities\n2 1 0 0\n1 0 0 0 0\n2 1 0 0 0\n1 0 0 0 1 0 0 0 2 1 -2\n$EndEntities\n")?;
/// let e = file.entities.unwrap();
/// assert_eq!((e.points.len(), e.curves.len(), e.surfaces.len()), (2, 1, 0));
/// assert_eq!(e.points[1].min, [1.0, 0.0, 0.0]);
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [the Gmsh format]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Entities {
    /// Dimension 0.
    pub points: Vec<Entity>,
    /// Dimension 1.
    pub curves: Vec<Entity>,
    /// Dimension 2.
    pub surfaces: Vec<Entity>,
    /// Dimension 3.
    pub volumes: Vec<Entity>,
}

/// An entity of a partitioned mesh, with its parent in the unpartitioned model. Gmsh writes these
/// for a mesh it has partitioned; see [Reading Gmsh in parallel].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::read_file;
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/partitioned_p1.msh");
/// let point = &read_file(path.as_ref())?.partitioned_entities.unwrap().points[0];
/// // Point 7 of partition 1 is point 1 of the unpartitioned model.
/// assert_eq!((point.entity.tag, point.parent_dim, point.parent_tag), (7, 0, 1));
/// assert_eq!(point.partitions, [1]);
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Reading Gmsh in parallel]: https://tetracl.github.io/tetraloma/tutorials/parallel-io.html#reading-gmsh-in-parallel
#[derive(Clone, Debug, PartialEq)]
pub struct PartitionedEntity {
    /// The partitioned entity itself.
    pub entity: Entity,
    /// The dimension of the parent entity.
    pub parent_dim: usize,
    /// The tag of the parent entity.
    pub parent_tag: i32,
    /// The partitions the entity belongs to.
    pub partitions: Vec<i32>,
}

/// `$PartitionedEntities`: the entities of a mesh that Gmsh has partitioned, and the number of
/// partitions. See [Reading Gmsh in parallel].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::read_file;
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/partitioned_p1.msh");
/// let p = read_file(path.as_ref())?.partitioned_entities.unwrap();
/// assert_eq!((p.num_partitions, p.ghost_entities.len()), (2, 0));
/// assert!(!p.surfaces.is_empty());
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Reading Gmsh in parallel]: https://tetracl.github.io/tetraloma/tutorials/parallel-io.html#reading-gmsh-in-parallel
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PartitionedEntities {
    /// The number of partitions.
    pub num_partitions: usize,
    /// The ghost entities: (entity tag, partition).
    pub ghost_entities: Vec<(i32, i32)>,
    /// Dimension 0.
    pub points: Vec<PartitionedEntity>,
    /// Dimension 1.
    pub curves: Vec<PartitionedEntity>,
    /// Dimension 2.
    pub surfaces: Vec<PartitionedEntity>,
    /// Dimension 3.
    pub volumes: Vec<PartitionedEntity>,
}

/// One block of `$Nodes`: the nodes classified on one entity. Parametric coordinates are read and
/// dropped. See [the Gmsh format].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
///     $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n")?;
/// let block = &file.node_blocks[0];
/// assert_eq!((block.dim, block.entity_tag, &block.tags[..]), (2, 1, &[1, 2, 3][..]));
/// assert_eq!(block.coords[1], [1.0, 0.0, 0.0]);
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [the Gmsh format]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh
#[derive(Clone, Debug, PartialEq)]
pub struct NodeBlock {
    /// The dimension of the entity the nodes are classified on.
    pub dim: usize,
    /// The tag of that entity.
    pub entity_tag: i32,
    /// The node tags (≥ 1).
    pub tags: Vec<i64>,
    /// The coordinates, `[node][3]`.
    pub coords: Vec<[f64; 3]>,
}

/// One block of `$Elements`: elements of one type on one entity, with their node tags in Gmsh's
/// node order. See [the Gmsh format] for the element types and node orders.
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
///     $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n")?;
/// let block = &file.element_blocks[0];
/// // One 3-node triangle (type 2), element tag 1.
/// assert_eq!((block.dim, block.element_type, &block.tags[..]), (2, 2, &[1][..]));
/// assert_eq!(block.nodes, [1, 2, 3]);
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [the Gmsh format]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementBlock {
    /// The dimension of the elements.
    pub dim: usize,
    /// The tag of the entity the elements are classified on.
    pub entity_tag: i32,
    /// The Gmsh element type id; [`element_type`](fn@super::element_type) describes it.
    pub element_type: i32,
    /// The element tags (≥ 1).
    pub tags: Vec<i64>,
    /// The node tags of each element in Gmsh order, `[element][nodes per element]`.
    pub nodes: Vec<i64>,
}

/// One link of `$Periodic`: an entity, its master entity and the affine map between them. tetraloma
/// stores the links but does not apply them. See [the Gmsh format].
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::read_file;
///
/// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/gmsh/periodic_p1.msh");
/// let link = &read_file(path.as_ref())?.periodic[0];
/// // Point 2 is the image of point 1 under a 4 × 4 affine map.
/// assert_eq!((link.dim, link.tag, link.master_tag), (0, 2, 1));
/// assert_eq!((link.affine.len(), &link.nodes[..]), (16, &[(2, 1)][..]));
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [the Gmsh format]: https://tetracl.github.io/tetraloma/elements/file-formats.html#gmsh
#[derive(Clone, Debug, PartialEq)]
pub struct Periodic {
    /// The dimension of the entities.
    pub dim: usize,
    /// The entity tag.
    pub tag: i32,
    /// The tag of its master entity.
    pub master_tag: i32,
    /// The affine transformation (16 values, row-major), or empty.
    pub affine: Vec<f64>,
    /// (node tag, master node tag) pairs.
    pub nodes: Vec<(i64, i64)>,
}

/// An MSH 4.1 file: the sections tetraloma reads, as stored. The parser skips unknown sections. The
/// version, byte order and `size_t` width belong to the encoding, not to the model.
/// [`to_gmsh_mesh`](super::to_gmsh_mesh) makes a mesh of it; the tutorial
/// [Reading and writing files] shows tetraloma's reader, which does this for you.
///
/// # Example
///
/// ```
/// use tetraloma_io::gmsh::parse;
///
/// let file = parse(b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
///     $Nodes\n1 3 1 3\n2 1 0 3\n1\n2\n3\n0 0 0\n1 0 0\n0 1 0\n$EndNodes\n\
///     $Elements\n1 1 1 1\n2 1 2 1\n1 1 2 3\n$EndElements\n")?;
/// assert_eq!((file.node_blocks.len(), file.element_blocks.len()), (1, 1));
/// assert!(file.entities.is_none() && file.physical_names.is_empty());
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#reading-gmsh
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MshFile {
    /// `$PhysicalNames`, in file order.
    pub physical_names: Vec<PhysicalName>,
    /// `$Entities`, if present.
    pub entities: Option<Entities>,
    /// `$PartitionedEntities`, if present.
    pub partitioned_entities: Option<PartitionedEntities>,
    /// The blocks of `$Nodes`, in file order.
    pub node_blocks: Vec<NodeBlock>,
    /// The blocks of `$Elements`, in file order.
    pub element_blocks: Vec<ElementBlock>,
    /// The links of `$Periodic`, in file order.
    pub periodic: Vec<Periodic>,
}
