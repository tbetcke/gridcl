//! VTKHDF output (docs/design.md, "VTK", VTKHDF; ADR-0035, rule 8): one `UnstructuredGrid` file
//! with one partition per rank.
//!
//! The layout is the one VTK 9.7.1's `vtkHDFWriter` writes for a partitioned data set, with
//! contiguous datasets of fixed size instead of its chunked, extendable ones:
//!
//! - group `/VTKHDF` with the attributes `Version` (Int64 `[2, 8]`) and `Type` (the 16-byte
//!   ASCII string `UnstructuredGrid`);
//! - `NumberOfPoints`, `NumberOfCells`, `NumberOfConnectivityIds`: Int64, one per partition;
//! - `Points` `[points][3]` (Float32 or Float64), `Connectivity` (Int64, partition-local point
//!   ids), `Offsets` (Int64, `cells + 1` per partition, each starting at 0), `Types` (UInt8),
//!   the partitions one after the other in rank order;
//! - groups `PointData`, `CellData` (one dataset per array, `[entries]` with one component,
//!   else `[entries][components]`) and an empty `FieldData`.
//!
//! The file-format bounds are v18 to v18, so any HDF5 from 1.8 on, the one in the VTK wheel
//! included, reads the file.

use std::path::Path;

use gridcl_device::{GridError, Result};
use gridcl_parallel::{CommDyn, CommExt, CommScalar};
use hdf5_metno::file::LibraryVersion;
use hdf5_metno::types::FixedAscii;
use hdf5_metno::{Dataset, File, Group, H5Type};
use ndarray::{ArrayView2, s};

use super::write::type_name;
use super::{DataArray, DataValues, VtuFile};

/// The `Version` attribute that VTK 9.7.1's `vtkHDFWriter` writes.
const VERSION: [i64; 2] = [2, 8];

/// The `Type` attribute.
const TYPE: &str = "UnstructuredGrid";

/// The format name in `GridError::Parse`.
const FORMAT: &str = "VTKHDF";

/// Applies `$body` to the values of a [`DataValues`], bound to `$x`.
macro_rules! with_values {
    ($v:expr, $x:ident => $body:expr) => {
        match $v {
            DataValues::F32($x) => $body,
            DataValues::F64($x) => $body,
            DataValues::I32($x) => $body,
            DataValues::U32($x) => $body,
            DataValues::I64($x) => $body,
            DataValues::U8($x) => $body,
        }
    };
}

/// The sizes of one partition.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Sizes {
    points: usize,
    cells: usize,
    connectivity: usize,
}

impl Sizes {
    fn of(piece: &VtuFile, points: usize) -> Sizes {
        Sizes {
            points,
            cells: piece.types.len(),
            connectivity: piece.connectivity.len(),
        }
    }

    fn add(self, o: Sizes) -> Result<Sizes> {
        let sum = |a: usize, b: usize| a.checked_add(b).ok_or(GridError::TooLarge);
        Ok(Sizes {
            points: sum(self.points, o.points)?,
            cells: sum(self.cells, o.cells)?,
            connectivity: sum(self.connectivity, o.connectivity)?,
        })
    }
}

/// Where every partition goes: `starts[r]` is the sum of the sizes of the partitions before
/// rank r, `total` that of all.
#[derive(Clone, Debug)]
struct Layout {
    sizes: Vec<Sizes>,
    starts: Vec<Sizes>,
    total: Sizes,
}

impl Layout {
    fn new(sizes: Vec<Sizes>) -> Result<Layout> {
        let mut starts = Vec::with_capacity(sizes.len());
        let mut total = Sizes::default();
        for &s in &sizes {
            starts.push(total);
            total = total.add(s)?;
        }
        Ok(Layout {
            sizes,
            starts,
            total,
        })
    }

    /// The length of `Offsets`: `cells + 1` per partition.
    fn offsets_len(&self) -> Result<usize> {
        self.total
            .cells
            .checked_add(self.sizes.len())
            .ok_or(GridError::TooLarge)
    }
}

/// Writes this rank's `piece` as partition `comm.rank()` of the VTKHDF file at `path`,
/// collectively over `comm`: every rank calls it, and the partitions follow in rank order. The
/// file is created or truncated.
///
/// With an MPI communicator (`comm.raw()` is `Some`, feature `mpi`) every rank writes its
/// partition through HDF5's MPI-IO driver; a rank with an empty piece still takes part in every
/// collective call. Otherwise the pieces are gathered and rank 0 writes the file with HDF5's
/// default driver. Both give the same datasets and attributes (module documentation).
///
/// # Errors
///
/// - `LengthMismatch` as [`VtuFile::write_to`] gives it, on the rank whose piece does not fit
///   together, and `RemoteFailure` on the others;
/// - `Parse` (offset 0) on every rank if the pieces differ in their point type or in the
///   names, components or types of their point or cell data arrays, which must be the same on
///   every rank;
/// - `TooLarge` if the total sizes overflow;
/// - `Io` naming `path` on the rank where HDF5 fails.
///
/// A rank without a local error returns `RemoteFailure` with the lowest failing rank.
///
/// # Example
///
/// ```
/// use gridcl_io::vtk::{DataValues, VtuFile, write_vtkhdf};
/// use gridcl_parallel::Serial;
///
/// // One triangle, on one rank.
/// let piece = VtuFile {
///     points: DataValues::F64(vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
///     connectivity: vec![0, 1, 2],
///     offsets: vec![3],
///     types: vec![5],
///     point_data: vec![],
///     cell_data: vec![],
/// };
/// let path = std::env::temp_dir().join(format!("gridcl-doc-{}.vtkhdf", std::process::id()));
/// write_vtkhdf(&path, &piece, &Serial)?;
/// assert!(std::fs::metadata(&path)?.len() > 0);
/// std::fs::remove_file(&path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_vtkhdf(path: &Path, piece: &VtuFile, comm: &dyn CommDyn) -> Result<()> {
    let points = agree(comm, piece.check())?;
    let layout = Layout::new(all_gather_sizes(comm, Sizes::of(piece, points))?)?;
    check_schema(comm, piece)?;
    #[cfg(feature = "mpi")]
    if comm.raw().is_some() {
        return collective(path, piece, &layout, comm);
    }
    gathered(path, piece, &layout, comm)
}

/// Every rank's sizes, in rank order. Collective.
fn all_gather_sizes(comm: &dyn CommDyn, mine: Sizes) -> Result<Vec<Sizes>> {
    let gather = |x: usize| comm.all_gather(x as u64);
    let (p, c, k) = (
        gather(mine.points),
        gather(mine.cells),
        gather(mine.connectivity),
    );
    let size = |x: u64| usize::try_from(x).map_err(|_| GridError::TooLarge);
    (0..comm.size())
        .map(|r| {
            Ok(Sizes {
                points: size(p[r])?,
                cells: size(c[r])?,
                connectivity: size(k[r])?,
            })
        })
        .collect()
}

/// The point type and the declared data arrays of a piece, as text.
fn schema(piece: &VtuFile) -> String {
    let arrays = |kind: &str, arrays: &[DataArray]| {
        arrays
            .iter()
            .map(|a| {
                format!(
                    "{kind} {:?} {} {}\n",
                    a.name,
                    a.components,
                    type_name(&a.values)
                )
            })
            .collect::<String>()
    };
    format!(
        "points {}\n{}{}",
        type_name(&piece.points),
        arrays("point", &piece.point_data),
        arrays("cell", &piece.cell_data)
    )
}

/// `Parse` on every rank unless every rank's [`schema`] is the same (by an FNV-1a hash, whose
/// minimum and maximum over the ranks must agree). Collective.
fn check_schema(comm: &dyn CommDyn, piece: &VtuFile) -> Result<()> {
    let hash = schema(piece)
        .bytes()
        .fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
            (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
    if comm.min(hash) != comm.max(hash) {
        return Err(GridError::Parse {
            format: FORMAT,
            offset: 0,
            message: "(no byte offset: a model being written) the pieces differ in their point \
                      type or in their point or cell data arrays"
                .into(),
        });
    }
    Ok(())
}

/// The agreed status of a collective step (ADR-0026, item 15): a failing rank returns its own
/// error, every other rank `RemoteFailure` with the lowest failing rank. Collective.
fn agree<T>(comm: &dyn CommDyn, local: Result<T>) -> Result<T> {
    let lowest = comm.min(if local.is_err() {
        comm.rank() as u64
    } else {
        u64::MAX
    });
    match local {
        Err(e) => Err(e),
        Ok(_) if lowest != u64::MAX => Err(GridError::RemoteFailure {
            rank: lowest as usize,
        }),
        ok => ok,
    }
}

/// `Io` naming `path` and the HDF5 error.
fn io(path: &Path) -> impl Fn(hdf5_metno::Error) -> GridError + '_ {
    move |e| GridError::Io(format!("{}: {e}", path.display()))
}

/// The open datasets of a file, created alike on every rank that writes.
struct Datasets {
    // Groups and datasets stay open until the file closes: with MPI-IO, every rank opens and
    // closes the same objects in the same order (HDF5's collective calling requirements).
    _groups: Vec<Group>,
    number_of_points: Dataset,
    number_of_cells: Dataset,
    number_of_connectivity_ids: Dataset,
    points: Dataset,
    connectivity: Dataset,
    offsets: Dataset,
    types: Dataset,
    point_data: Vec<Dataset>,
    cell_data: Vec<Dataset>,
}

/// A dataset of `entries` rows of `components` values of `T` (one dimension if `components`
/// is 1).
fn new_dataset<T: H5Type>(
    group: &Group,
    _like: &[T],
    name: &str,
    entries: usize,
    components: usize,
) -> hdf5_metno::Result<Dataset> {
    if components == 1 {
        group.new_dataset::<T>().shape(entries).create(name)
    } else {
        group
            .new_dataset::<T>()
            .shape((entries, components))
            .create(name)
    }
}

/// The datasets of the data arrays `like`, `entries` rows each, in a new group `name`.
fn data_group(
    root: &Group,
    name: &str,
    like: &[DataArray],
    entries: usize,
) -> hdf5_metno::Result<(Group, Vec<Dataset>)> {
    let group = root.create_group(name)?;
    let sets = like
        .iter()
        .map(|a| with_values!(&a.values, x => new_dataset(&group, x, &a.name, entries, a.components)))
        .collect::<hdf5_metno::Result<_>>()?;
    Ok((group, sets))
}

/// Creates the groups, attributes and datasets of a file holding `layout`'s partitions, with
/// the arrays declared by `like`. Collective with MPI-IO.
fn create(file: &File, like: &VtuFile, layout: &Layout) -> hdf5_metno::Result<Datasets> {
    let root = file.create_group("VTKHDF")?;
    root.new_attr::<i64>()
        .shape(VERSION.len())
        .create("Version")?
        .write(&VERSION)?;
    let kind =
        FixedAscii::<16>::from_ascii(TYPE).map_err(|e| hdf5_metno::Error::from(e.to_string()))?;
    root.new_attr::<FixedAscii<16>>()
        .create("Type")?
        .write_scalar(&kind)?;
    let partitions = layout.sizes.len();
    let count = |name: &str| root.new_dataset::<i64>().shape(partitions).create(name);
    let number_of_points = count("NumberOfPoints")?;
    let number_of_cells = count("NumberOfCells")?;
    let number_of_connectivity_ids = count("NumberOfConnectivityIds")?;
    let total = layout.total;
    let points = with_values!(&like.points, x => new_dataset(&root, x, "Points", total.points, 3))?;
    let connectivity = root
        .new_dataset::<i64>()
        .shape(total.connectivity)
        .create("Connectivity")?;
    let offsets_len = layout
        .offsets_len()
        .map_err(|e| hdf5_metno::Error::from(e.to_string()))?;
    let offsets = root
        .new_dataset::<i64>()
        .shape(offsets_len)
        .create("Offsets")?;
    let types = root
        .new_dataset::<u8>()
        .shape(total.cells)
        .create("Types")?;
    let (point_group, point_data) = data_group(&root, "PointData", &like.point_data, total.points)?;
    let (cell_group, cell_data) = data_group(&root, "CellData", &like.cell_data, total.cells)?;
    let field_group = root.create_group("FieldData")?;
    Ok(Datasets {
        _groups: vec![root, point_group, cell_group, field_group],
        number_of_points,
        number_of_cells,
        number_of_connectivity_ids,
        points,
        connectivity,
        offsets,
        types,
        point_data,
        cell_data,
    })
}

/// Writes `values`, rows of `components`, into `set` from row `start` on; nothing if empty.
fn write_rows<T: H5Type>(
    set: &Dataset,
    values: &[T],
    components: usize,
    start: usize,
) -> hdf5_metno::Result<()> {
    let rows = values.len() / components;
    if rows == 0 {
        return Ok(());
    }
    if set.ndim() == 1 {
        set.write_slice(values, s![start..start + rows])
    } else {
        let view = ArrayView2::from_shape((rows, components), values)
            .map_err(|e| hdf5_metno::Error::from(e.to_string()))?;
        set.write_slice(view, s![start..start + rows, ..])
    }
}

/// Writes the per-partition counts (by one rank).
fn write_counts(sets: &Datasets, layout: &Layout) -> hdf5_metno::Result<()> {
    let column =
        |f: fn(&Sizes) -> usize| layout.sizes.iter().map(|s| f(s) as i64).collect::<Vec<_>>();
    sets.number_of_points.write(&column(|s| s.points))?;
    sets.number_of_cells.write(&column(|s| s.cells))?;
    sets.number_of_connectivity_ids
        .write(&column(|s| s.connectivity))
}

/// Writes `piece` as partition `rank` of `layout`.
fn write_piece(
    sets: &Datasets,
    piece: &VtuFile,
    layout: &Layout,
    rank: usize,
) -> hdf5_metno::Result<()> {
    let start = layout.starts[rank];
    with_values!(&piece.points, x => write_rows(&sets.points, x, 3, start.points))?;
    write_rows(
        &sets.connectivity,
        &piece.connectivity,
        1,
        start.connectivity,
    )?;
    let offsets: Vec<i64> = std::iter::once(0)
        .chain(piece.offsets.iter().copied())
        .collect();
    write_rows(&sets.offsets, &offsets, 1, start.cells + rank)?;
    write_rows(&sets.types, &piece.types, 1, start.cells)?;
    for (set, a) in sets.point_data.iter().zip(&piece.point_data) {
        with_values!(&a.values, x => write_rows(set, x, a.components, start.points))?;
    }
    for (set, a) in sets.cell_data.iter().zip(&piece.cell_data) {
        with_values!(&a.values, x => write_rows(set, x, a.components, start.cells))?;
    }
    Ok(())
}

/// A new file at `path` (truncated if it exists) with the format bounds v18–v18 and, with
/// `mpio`, the MPI-IO driver on that communicator's handle. Collective with MPI-IO.
fn new_file(path: &Path, mpio: Option<&dyn CommDyn>) -> hdf5_metno::Result<File> {
    File::with_options()
        .with_fapl(|p| {
            #[cfg(feature = "mpi")]
            if let Some(raw) = mpio.and_then(|c| c.raw()) {
                p.mpio(raw, None);
            }
            #[cfg(not(feature = "mpi"))]
            let _ = mpio;
            p.libver_bounds(LibraryVersion::V18, LibraryVersion::V18)
        })
        .create(path)
}

/// The MPI-IO path: every rank creates the same objects, writes its own partition (rank 0
/// also the counts), and closes the file.
#[cfg(feature = "mpi")]
fn collective(path: &Path, piece: &VtuFile, layout: &Layout, comm: &dyn CommDyn) -> Result<()> {
    let file = agree(comm, new_file(path, Some(comm)).map_err(io(path)))?;
    let sets = agree(comm, create(&file, piece, layout).map_err(io(path)))?;
    let written = (|| {
        if comm.rank() == 0 {
            write_counts(&sets, layout)?;
        }
        write_piece(&sets, piece, layout, comm.rank())
    })();
    agree(comm, written.map_err(io(path)))?;
    drop(sets);
    agree(comm, file.close().map_err(io(path)))
}

/// The gather path: rank 0 receives every piece and writes them all.
fn gathered(path: &Path, piece: &VtuFile, layout: &Layout, comm: &dyn CommDyn) -> Result<()> {
    let pieces = agree(comm, gather_pieces(comm, piece))?;
    let written = match pieces {
        Some(pieces) => (|| {
            let file = new_file(path, None)?;
            let sets = create(&file, piece, layout)?;
            write_counts(&sets, layout)?;
            for (rank, p) in pieces.iter().enumerate() {
                write_piece(&sets, p, layout, rank)?;
            }
            drop(sets);
            file.close()
        })()
        .map_err(io(path)),
        None => Ok(()),
    };
    agree(comm, written)
}

/// The values of every rank on rank 0 (empty elsewhere), in rank order. Collective.
fn gather<T: CommScalar>(comm: &dyn CommDyn, mine: &[T]) -> Result<Vec<Vec<T>>> {
    let mut counts = vec![0; comm.size()];
    counts[0] = mine.len();
    let (values, from) = comm.all_to_all_v(mine, &counts)?;
    let mut rest = values.as_slice();
    Ok(from
        .into_iter()
        .map(|n| {
            let (head, tail) = rest.split_at(n);
            rest = tail;
            head.to_vec()
        })
        .collect())
}

/// [`gather`] of a [`DataValues`] (UInt8 through `u32`).
fn gather_values(comm: &dyn CommDyn, mine: &DataValues) -> Result<Vec<DataValues>> {
    Ok(match mine {
        DataValues::F32(x) => gather(comm, x)?.into_iter().map(DataValues::F32).collect(),
        DataValues::F64(x) => gather(comm, x)?.into_iter().map(DataValues::F64).collect(),
        DataValues::I32(x) => gather(comm, x)?.into_iter().map(DataValues::I32).collect(),
        DataValues::U32(x) => gather(comm, x)?.into_iter().map(DataValues::U32).collect(),
        DataValues::I64(x) => gather(comm, x)?.into_iter().map(DataValues::I64).collect(),
        DataValues::U8(x) => gather_u8(comm, x)?
            .into_iter()
            .map(DataValues::U8)
            .collect(),
    })
}

/// [`gather`] of bytes, through `u32`.
fn gather_u8(comm: &dyn CommDyn, mine: &[u8]) -> Result<Vec<Vec<u8>>> {
    let wide: Vec<u32> = mine.iter().map(|&b| u32::from(b)).collect();
    Ok(gather(comm, &wide)?
        .into_iter()
        // Every value came from a u8.
        .map(|x| x.into_iter().map(|w| w as u8).collect())
        .collect())
}

/// [`gather`] of data arrays declared alike on every rank (checked by [`check_schema`]).
fn gather_arrays(comm: &dyn CommDyn, mine: &[DataArray]) -> Result<Vec<Vec<DataArray>>> {
    let mut out = vec![Vec::with_capacity(mine.len()); comm.size()];
    for a in mine {
        for (r, values) in gather_values(comm, &a.values)?.into_iter().enumerate() {
            out[r].push(DataArray {
                name: a.name.clone(),
                components: a.components,
                values,
            });
        }
    }
    Ok(out)
}

/// Every rank's piece on rank 0, `None` elsewhere. Collective.
fn gather_pieces(comm: &dyn CommDyn, piece: &VtuFile) -> Result<Option<Vec<VtuFile>>> {
    let points = gather_values(comm, &piece.points)?;
    let connectivity = gather(comm, &piece.connectivity)?;
    let offsets = gather(comm, &piece.offsets)?;
    let types = gather_u8(comm, &piece.types)?;
    let point_data = gather_arrays(comm, &piece.point_data)?;
    let cell_data = gather_arrays(comm, &piece.cell_data)?;
    if comm.rank() != 0 {
        return Ok(None);
    }
    Ok(Some(
        points
            .into_iter()
            .zip(connectivity)
            .zip(offsets)
            .zip(types)
            .zip(point_data.into_iter().zip(cell_data))
            .map(
                |((((points, connectivity), offsets), types), (point_data, cell_data))| VtuFile {
                    points,
                    connectivity,
                    offsets,
                    types,
                    point_data,
                    cell_data,
                },
            )
            .collect(),
    ))
}
