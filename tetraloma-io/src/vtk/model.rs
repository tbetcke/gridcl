//! The file model and the host mesh description.

use tetraloma_dims::CellType;

/// The values of one data array, in one of the VTK types tetraloma reads and writes. See
/// [Writing VTU].
///
/// # Example
///
/// ```
/// use tetraloma_dims::CellType;
/// use tetraloma_io::vtk::{DataValues, VtkCellBlock, VtkMesh, to_vtu};
///
/// let mesh = VtkMesh {
///     gdim: 1,
///     nodes: vec![0.0, 1.0],
///     blocks: vec![VtkCellBlock { cell: CellType::Interval, degree: 1, nodes: vec![0, 1] }],
/// };
/// // Points are Float64, or Float32 on request.
/// let (file, _) = to_vtu(&mesh, true, true)?;
/// assert_eq!(file.points, DataValues::F32(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0]));
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Writing VTU]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#writing-vtu
#[derive(Clone, Debug, PartialEq)]
pub enum DataValues {
    /// `Float32`.
    F32(Vec<f32>),
    /// `Float64`.
    F64(Vec<f64>),
    /// `Int32`.
    I32(Vec<i32>),
    /// `UInt32`.
    U32(Vec<u32>),
    /// `Int64`.
    I64(Vec<i64>),
    /// `UInt8`.
    U8(Vec<u8>),
}

/// A named point or cell data array, `[entry][components]`: one entry per point or per cell, each
/// of `components` values. See [Writing VTU].
///
/// # Example
///
/// ```
/// use tetraloma_io::vtk::{DataArray, DataValues};
///
/// // A vector field on two points.
/// let velocity = DataArray {
///     name: "velocity".into(),
///     components: 3,
///     values: DataValues::F64(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
/// };
/// assert_eq!(velocity.components, 3);
/// ```
///
/// [Writing VTU]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#writing-vtu
#[derive(Clone, Debug, PartialEq)]
pub struct DataArray {
    /// The `Name` attribute.
    pub name: String,
    /// `NumberOfComponents`, at least 1.
    pub components: usize,
    /// The values.
    pub values: DataValues,
}

/// The arrays of one `UnstructuredGrid` piece, as stored: points, cells with their VTK cell types
/// and VTK node order, and point and cell data. [`to_vtu`](super::to_vtu) makes it from a
/// [`VtkMesh`]; tetraloma's `VtkWriter` adds the data arrays. See [Writing VTU].
///
/// # Example
///
/// ```
/// use tetraloma_dims::CellType;
/// use tetraloma_io::vtk::{DataValues, VtkCellBlock, VtkMesh, to_vtu};
///
/// // Two triangles of a unit square.
/// let mesh = VtkMesh {
///     gdim: 2,
///     nodes: vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
///     blocks: vec![VtkCellBlock {
///         cell: CellType::Triangle,
///         degree: 1,
///         nodes: vec![0, 1, 2, 1, 3, 2],
///     }],
/// };
/// let (file, _) = to_vtu(&mesh, true, false)?;
/// assert_eq!(file.connectivity, [0, 1, 2, 1, 3, 2]);
/// assert_eq!(file.offsets, [3, 6]);
/// assert_eq!(file.types, [5, 5]);
/// let DataValues::F64(points) = &file.points else { unreachable!() };
/// assert_eq!(points[9..], [1.0, 1.0, 0.0]);
/// assert!(file.point_data.is_empty() && file.cell_data.is_empty());
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Writing VTU]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#writing-vtu
#[derive(Clone, Debug, PartialEq)]
pub struct VtuFile {
    /// Point coordinates `[point][3]`, `F32` or `F64`.
    pub points: DataValues,
    /// The point indices of every cell, in VTK node order.
    pub connectivity: Vec<i64>,
    /// The end of each cell in `connectivity`.
    pub offsets: Vec<i64>,
    /// The VTK cell type of each cell, as [`vtk_cell_type`](super::vtk_cell_type) gives it.
    pub types: Vec<u8>,
    /// Point data, one entry per point.
    pub point_data: Vec<DataArray>,
    /// Cell data, one entry per cell.
    pub cell_data: Vec<DataArray>,
}

/// The mesh a VTU file describes, on the host: node coordinates with the origin added, and cell
/// blocks in tetraloma's node order ([Lagrange nodes]). tetraloma's `VtkWriter` downloads a mesh
/// into it; see [Writing VTU].
///
/// # Example
///
/// ```
/// use tetraloma_dims::CellType;
/// use tetraloma_io::vtk::{VtkCellBlock, VtkMesh, to_vtu};
///
/// // One P2 interval on the x axis: vertices first, then the midpoint.
/// let mesh = VtkMesh {
///     gdim: 1,
///     nodes: vec![0.0, 1.0, 0.5],
///     blocks: vec![VtkCellBlock { cell: CellType::Interval, degree: 2, nodes: vec![0, 1, 2] }],
/// };
/// let (file, _) = to_vtu(&mesh, true, false)?;
/// assert_eq!((file.types[0], &file.connectivity[..]), (68, &[0, 1, 2][..]));
/// # Ok::<(), tetraloma_device::GridError>(())
/// ```
///
/// [Lagrange nodes]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-ordering
/// [Writing VTU]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#writing-vtu
#[derive(Clone, Debug, PartialEq)]
pub struct VtkMesh {
    /// The geometric dimension, 1–3.
    pub gdim: usize,
    /// Node coordinates `[node][gdim]`, with the origin added.
    pub nodes: Vec<f64>,
    /// The cell blocks, written in order.
    pub blocks: Vec<VtkCellBlock>,
}

/// One block of a [`VtkMesh`]: equispaced Lagrange cells of one type and degree, with their nodes
/// in tetraloma's node order ([Lagrange nodes]).
///
/// # Example
///
/// ```
/// use tetraloma_dims::CellType;
/// use tetraloma_io::vtk::VtkCellBlock;
///
/// // One P2 triangle: three vertices, then the edges (0,1), (0,2), (1,2).
/// let block = VtkCellBlock { cell: CellType::Triangle, degree: 2, nodes: vec![0, 1, 2, 3, 4, 5] };
/// assert_eq!(block.nodes.len(), tetraloma_refcell::num_lagrange_nodes(block.cell, block.degree));
/// ```
///
/// [Lagrange nodes]: https://tetracl.github.io/tetraloma/elements/lagrange.html#node-ordering
#[derive(Clone, Debug, PartialEq)]
pub struct VtkCellBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// Cell → node lists in tetraloma's node order, `num_lagrange_nodes(cell, degree)` per
    /// cell.
    pub nodes: Vec<u32>,
}
