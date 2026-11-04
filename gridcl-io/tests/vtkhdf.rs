//! The VTKHDF writer (docs/plans/M8.md, T9; ADR-0035, rule 8): the layout of the file, the
//! gather path on one rank and on 1–4 threads, VTK's `vtkHDFReader` reading what a PVTU of the
//! same pieces holds, and agreed errors. Feature `hdf5`.

#![cfg(feature = "hdf5")]

use std::fs;
use std::path::{Path, PathBuf};

use gridcl_device::GridError;
use gridcl_io::vtk::{Binary, DataArray, DataValues, VtuFile, write_pvtu, write_vtkhdf};
use gridcl_parallel::{CommDyn, Serial};
use gridcl_testing::ThreadComm;
use gridcl_testing::oracle::Oracle;
use hdf5_metno::File;
use hdf5_metno::types::FixedAscii;

/// A directory for the files of `name`, new and empty, unique per process.
fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("gridcl-m8-vtkhdf-{}-{name}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    dir
}

/// The piece of rank `rank`: `rank + 1` triangles in a strip (none on rank 1 of 3 or more
/// ranks), points shifted by the rank, every data type, a two- and a three-component array,
/// and `vtkGhostType` 1 on the last cell of ranks above 0. Points in f32 if `single`.
fn piece(rank: usize, size: usize, single: bool) -> VtuFile {
    let cells = if rank == 1 && size >= 3 { 0 } else { rank + 1 };
    let points = if cells == 0 { 0 } else { cells + 2 };
    let x0 = 10.0 * rank as f64;
    let coords: Vec<f64> = (0..points)
        .flat_map(|i| [x0 + (i / 2) as f64 + 0.1, (i % 2) as f64 / 3.0, 0.0])
        .collect();
    let connectivity: Vec<i64> = (0..cells)
        .flat_map(|c| {
            let c = c as i64;
            if c % 2 == 0 {
                [c, c + 1, c + 2]
            } else {
                [c, c + 2, c + 1]
            }
        })
        .collect();
    let id = |k: usize| (100 * rank + k) as i64;
    VtuFile {
        points: if single {
            DataValues::F32(coords.iter().map(|&x| x as f32).collect())
        } else {
            DataValues::F64(coords)
        },
        offsets: (1..=cells).map(|c| 3 * c as i64).collect(),
        connectivity,
        types: vec![5; cells],
        point_data: vec![
            DataArray {
                name: "u".into(),
                components: 1,
                values: DataValues::F64((0..points).map(|i| id(i) as f64 / 7.0).collect()),
            },
            DataArray {
                name: "velocity".into(),
                components: 3,
                values: DataValues::F32((0..3 * points).map(|i| id(i) as f32 / 3.0).collect()),
            },
        ],
        cell_data: vec![
            DataArray {
                name: "level".into(),
                components: 1,
                values: DataValues::I32((0..cells).map(|c| -(id(c) as i32)).collect()),
            },
            DataArray {
                name: "pair".into(),
                components: 2,
                values: DataValues::U32((0..2 * cells).map(|c| id(c) as u32).collect()),
            },
            DataArray {
                name: "OriginId".into(),
                components: 1,
                values: DataValues::I64((0..cells).map(id).collect()),
            },
            DataArray {
                name: "vtkGhostType".into(),
                components: 1,
                values: DataValues::U8(
                    (0..cells)
                        .map(|c| u8::from(rank > 0 && c + 1 == cells))
                        .collect(),
                ),
            },
        ],
    }
}

/// Writes `pieces` as `<dir>/<stem>.pvtu` with one VTU per piece (ghost level 1).
fn write_reference(dir: &Path, stem: &str, pieces: &[VtuFile]) -> PathBuf {
    let names: Vec<String> = (0..pieces.len())
        .map(|r| format!("{stem}_{r}.vtu"))
        .collect();
    for (p, name) in pieces.iter().zip(&names) {
        let mut f = fs::File::create(dir.join(name)).unwrap();
        p.write_to(&mut f, Binary::Appended, false).unwrap();
    }
    let path = dir.join(format!("{stem}.pvtu"));
    let mut f = fs::File::create(&path).unwrap();
    write_pvtu(&mut f, &names, &pieces[0], 1).unwrap();
    path
}

/// Writes the pieces of `size` threads (or of `Serial` if `size` is 0) with `write_vtkhdf`.
fn write_on(size: usize, path: &Path, single: bool) -> Vec<VtuFile> {
    if size == 0 {
        let p = piece(0, 1, single);
        write_vtkhdf(path, &p, &Serial).unwrap();
        return vec![p];
    }
    ThreadComm::run(size, |comm| {
        let p = piece(comm.rank(), size, single);
        write_vtkhdf(path, &p, &comm).unwrap_or_else(|e| panic!("rank {}: {e}", comm.rank()));
        p
    })
}

/// Opens `path` without HDF5's file lock, which an oracle process spawned by a concurrent test
/// may hold (HDF5 opens files without close-on-exec).
fn open(path: &Path) -> File {
    File::with_options()
        .with_fapl(|p| p.file_locking(false))
        .open(path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

fn i64s(file: &File, name: &str) -> Vec<i64> {
    file.dataset(name).unwrap().read_raw::<i64>().unwrap()
}

#[test]
fn layout_is_vtkhdf_with_one_partition_per_rank() {
    let dir = temp_dir("layout");
    let path = dir.join("strip.vtkhdf");
    let pieces = write_on(3, &path, false);
    let file = open(&path);
    let root = file.group("VTKHDF").unwrap();
    assert_eq!(
        root.attr("Version").unwrap().read_raw::<i64>().unwrap(),
        [2, 8]
    );
    let kind: FixedAscii<16> = root.attr("Type").unwrap().read_scalar().unwrap();
    assert_eq!(kind.as_str(), "UnstructuredGrid");
    assert_eq!(i64s(&file, "VTKHDF/NumberOfPoints"), [3, 0, 5]);
    assert_eq!(i64s(&file, "VTKHDF/NumberOfCells"), [1, 0, 3]);
    assert_eq!(i64s(&file, "VTKHDF/NumberOfConnectivityIds"), [3, 0, 9]);
    // Offsets per partition, each from 0; connectivity partition-local.
    assert_eq!(i64s(&file, "VTKHDF/Offsets"), [0, 3, 0, 0, 3, 6, 9]);
    let connectivity: Vec<i64> = pieces.iter().flat_map(|p| p.connectivity.clone()).collect();
    assert_eq!(i64s(&file, "VTKHDF/Connectivity"), connectivity);
    let points = file.dataset("VTKHDF/Points").unwrap();
    assert_eq!(points.shape(), [8, 3]);
    let DataValues::F64(want) = &pieces[2].points else {
        unreachable!()
    };
    assert_eq!(&points.read_raw::<f64>().unwrap()[9..], &want[..]);
    assert_eq!(
        file.dataset("VTKHDF/Types")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        [5; 4]
    );
    let shape = |name: &str| file.dataset(name).unwrap().shape();
    assert_eq!(shape("VTKHDF/PointData/u"), [8]);
    assert_eq!(shape("VTKHDF/PointData/velocity"), [8, 3]);
    assert_eq!(shape("VTKHDF/CellData/pair"), [4, 2]);
    assert_eq!(
        file.dataset("VTKHDF/CellData/vtkGhostType")
            .unwrap()
            .read_raw::<u8>()
            .unwrap(),
        [0, 0, 0, 1]
    );
    assert!(
        file.group("VTKHDF/FieldData")
            .unwrap()
            .member_names()
            .unwrap()
            .is_empty()
    );
    drop(file);
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn file_bounds_are_v18() {
    let dir = temp_dir("bounds");
    let path = dir.join("one.vtkhdf");
    write_on(0, &path, true);
    let file = open(&path);
    let bounds = file.fapl().unwrap().libver_bounds();
    drop(file);
    // The superblock written with low = v18 is version 2 (HDF5's file-format specification).
    let bytes = fs::read(&path).unwrap();
    assert_eq!(&bytes[..8], b"\x89HDF\r\n\x1a\n");
    assert_eq!(
        bytes[8], 2,
        "superblock version (bounds read back: {bounds:?})"
    );
    fs::remove_dir_all(&dir).unwrap();
}

/// VTK reads the VTKHDF file of `size` threads (0: `Serial`) as the PVTU of the same pieces,
/// array by array and bitwise, with the same per-partition counts.
fn reads_as_pvtu(size: usize, single: bool) {
    let test = format!("vtkhdf_reads_as_pvtu({size}, f32 points: {single})");
    let Some(oracle) = Oracle::find(&test) else {
        return;
    };
    let dir = temp_dir(&format!("pvtu-{size}-{single}"));
    let path = dir.join("strip.vtkhdf");
    let pieces = write_on(size, &path, single);
    let reference = write_reference(&dir, "strip", &pieces);
    let read = &oracle.run_files("read", std::slice::from_ref(&path))[0];
    assert_eq!(read["errors"], serde_json::json!([]), "{test}: {read}");
    let total: usize = pieces.iter().map(|p| p.types.len()).sum();
    assert_eq!(read["num_cells"], total, "{test}: {read}");
    let out = oracle.run(
        "compare",
        &[path.clone().into_os_string(), reference.into_os_string()],
    );
    assert_eq!(out["equal"], true, "{test}: {out}");
    assert_eq!(out["pieces"], pieces.len(), "{test}: {out}");
    let ghosts = &oracle.run_files("remove-ghosts", &[path])[0];
    let owned = pieces
        .iter()
        .enumerate()
        .map(|(r, p)| p.types.len() - usize::from(r > 0 && !p.types.is_empty()))
        .sum::<usize>();
    assert_eq!(ghosts["num_cells"], owned, "{test}: {ghosts}");
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn vtkhdf_reads_as_pvtu_on_one_rank() {
    reads_as_pvtu(0, false);
    reads_as_pvtu(0, true);
}

#[test]
fn vtkhdf_reads_as_pvtu_gathered_on_threads() {
    for size in 1..=4 {
        reads_as_pvtu(size, size % 2 == 0);
    }
}

#[test]
fn piece_that_does_not_fit_fails_on_its_rank() {
    let dir = temp_dir("bad-piece");
    let path = dir.join("bad.vtkhdf");
    let got = ThreadComm::run(3, |comm| {
        let mut p = piece(comm.rank(), 3, false);
        if comm.rank() == 2 {
            p.offsets.pop();
        }
        write_vtkhdf(&path, &p, &comm)
    });
    assert!(
        matches!(got[2], Err(GridError::LengthMismatch { .. })),
        "{got:?}"
    );
    for r in [0, 1] {
        assert!(
            matches!(got[r], Err(GridError::RemoteFailure { rank: 2 })),
            "{got:?}"
        );
    }
    assert!(!path.exists());
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn pieces_with_different_arrays_fail_everywhere() {
    let dir = temp_dir("schema");
    let path = dir.join("schema.vtkhdf");
    let got = ThreadComm::run(2, |comm| {
        let mut p = piece(comm.rank(), 2, false);
        if comm.rank() == 1 {
            p.cell_data[0].name = "other".into();
        }
        write_vtkhdf(&path, &p, &comm)
    });
    for r in &got {
        assert!(
            matches!(
                r,
                Err(GridError::Parse {
                    format: "VTKHDF",
                    offset: 0,
                    ..
                })
            ),
            "{got:?}"
        );
    }
    fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn unwritable_path_gives_io_and_remote_failure() {
    let path = Path::new("/nonexistent-gridcl-dir/out.vtkhdf");
    let got = ThreadComm::run(2, |comm| {
        write_vtkhdf(path, &piece(comm.rank(), 2, false), &comm)
    });
    // Rank 0 writes on the gather path; rank 1 hears of its failure (ADR-0026, item 15).
    let Err(GridError::Io(message)) = &got[0] else {
        panic!("{got:?}");
    };
    assert!(
        message.contains("/nonexistent-gridcl-dir/out.vtkhdf"),
        "{message}"
    );
    assert!(
        matches!(got[1], Err(GridError::RemoteFailure { rank: 0 })),
        "{got:?}"
    );
    let one = write_vtkhdf(path, &piece(0, 1, false), &Serial);
    assert!(matches!(one, Err(GridError::Io(_))), "{one:?}");
}
