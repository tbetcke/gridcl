//! The file model and the host mesh description.

use gridcl_dims::CellType;

/// The values of one data array, in one of the VTK types gridcl reads and writes.
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

/// A named point or cell data array, `[entry][components]`.
#[derive(Clone, Debug, PartialEq)]
pub struct DataArray {
    /// The `Name` attribute.
    pub name: String,
    /// `NumberOfComponents`, at least 1.
    pub components: usize,
    /// The values.
    pub values: DataValues,
}

/// The arrays of one `UnstructuredGrid` piece, as stored.
#[derive(Clone, Debug, PartialEq)]
pub struct VtuFile {
    /// Point coordinates `[point][3]`, `F32` or `F64`.
    pub points: DataValues,
    /// The point indices of every cell, in VTK node order.
    pub connectivity: Vec<i64>,
    /// The end of each cell in `connectivity`.
    pub offsets: Vec<i64>,
    /// The VTK cell type of each cell (A.8).
    pub types: Vec<u8>,
    /// Point data, one entry per point.
    pub point_data: Vec<DataArray>,
    /// Cell data, one entry per cell.
    pub cell_data: Vec<DataArray>,
}

/// The mesh a VTU file describes: nodes with the origin added, and cell blocks in internal
/// (A.5) node order.
#[derive(Clone, Debug, PartialEq)]
pub struct VtkMesh {
    /// The geometric dimension, 1–3.
    pub gdim: usize,
    /// Node coordinates `[node][gdim]`, with the origin added.
    pub nodes: Vec<f64>,
    /// The cell blocks, written in order.
    pub blocks: Vec<VtkCellBlock>,
}

/// One block of equispaced Lagrange cells of one type and degree.
#[derive(Clone, Debug, PartialEq)]
pub struct VtkCellBlock {
    /// The cell type.
    pub cell: CellType,
    /// The degree p ≥ 1.
    pub degree: u32,
    /// Cell → node lists in A.5 order, `num_lagrange_nodes(cell, degree)` per cell.
    pub nodes: Vec<u32>,
}
