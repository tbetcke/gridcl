//! The file model: the sections of an MSH 4.1 file as stored.

/// One entry of `$PhysicalNames`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhysicalName {
    /// The dimension of the physical group.
    pub dim: usize,
    /// The physical tag.
    pub tag: i32,
    /// The name, without quotes.
    pub name: String,
}

/// A geometric entity of `$Entities` (or the entity part of a `$PartitionedEntities` entry).
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

/// `$Entities`: the geometric entities by dimension.
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

/// An entity of a partitioned mesh, with its parent in the unpartitioned model.
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

/// `$PartitionedEntities`.
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

/// One block of `$Nodes`. Parametric coordinates are read and dropped.
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

/// One block of `$Elements`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ElementBlock {
    /// The dimension of the elements.
    pub dim: usize,
    /// The tag of the entity the elements are classified on.
    pub entity_tag: i32,
    /// The Gmsh element type (A.7).
    pub element_type: i32,
    /// The element tags (≥ 1).
    pub tags: Vec<i64>,
    /// The node tags of each element in Gmsh order, `[element][nodes per element]`.
    pub nodes: Vec<i64>,
}

/// One link of `$Periodic`: stored, not applied.
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

/// An MSH 4.1 file: the sections gridcl reads, as stored. Unknown sections are skipped by the
/// parser; the version, byte order and `size_t` width are properties of the encoding, not of
/// the model.
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
