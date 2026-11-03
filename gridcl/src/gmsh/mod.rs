//! Gmsh MSH 4.1 meshes: [`read`] and [`GmshWriter`] (docs/design.md, "Import and export /
//! Gmsh MSH 4.1"; ADR-0022, ADR-0035). The file model and the conversion to a host mesh
//! description are in `gridcl_io::gmsh`.

mod options;
mod read;
mod write;

pub use options::GmshOptions;
pub use read::read;
pub use write::GmshWriter;
#[cfg(test)]
pub(crate) use write::gathered;
pub(crate) use write::{elementary_of, values_by_index};

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::thread::ThreadId;

    use cubecl::prelude::CubeElement;
    use gridcl_parallel::Serial;

    use std::path::{Path, PathBuf};

    use gridcl_device::{GridError, Real, Result};
    use gridcl_geometry::DeviceGrid;
    use gridcl_topology::decode_origin_ids;

    use super::*;
    use crate::{Codim, CoordinateElement, D2, GmshOptions, MeshBuilder, NodeFamily, PlanarMesh};

    fn fixture(name: &str) -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../gridcl-io/tests/fixtures/gmsh")
            .join(name)
    }

    /// Warnings by thread, so that tests running in parallel see only their own.
    static RECORDS: Mutex<Vec<(ThreadId, String)>> = Mutex::new(Vec::new());

    struct Capture;

    impl log::Log for Capture {
        fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
            metadata.level() <= log::Level::Warn
        }
        fn log(&self, record: &log::Record<'_>) {
            if self.enabled(record.metadata()) {
                let mut records = RECORDS.lock().unwrap();
                records.push((std::thread::current().id(), record.args().to_string()));
            }
        }
        fn flush(&self) {}
    }

    static LOGGER: Capture = Capture;

    fn my_warnings() -> Vec<String> {
        let me = std::thread::current().id();
        let records = RECORDS.lock().unwrap();
        records
            .iter()
            .filter(|r| r.0 == me)
            .map(|r| r.1.clone())
            .collect()
    }

    /// One `log::warn!` per f32 read, naming the file; none for f64.
    #[test]
    fn gmsh_read_warns_once_per_f32_file() {
        // Another test binary thread may have set it already; either way it is ours.
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Warn);
        let client = gridcl_testing::test_client();
        let path = fixture("triangle_p2.msh");
        read::<f32, _>(&client, Serial, &path, GmshOptions::default()).unwrap();
        let warnings = my_warnings();
        assert_eq!(warnings.len(), 1, "{warnings:?}");
        assert!(warnings[0].contains("triangle_p2.msh"), "{}", warnings[0]);
        if gridcl_testing::supports::<f64>(&client) {
            read::<f64, _>(&client, Serial, &path, GmshOptions::default()).unwrap();
            assert_eq!(my_warnings().len(), 1);
        }
        read::<f32, _>(
            &client,
            Serial,
            fixture("hybrid_p1.msh"),
            GmshOptions::default(),
        )
        .unwrap();
        assert_eq!(my_warnings().len(), 2);
    }

    /// On two threads only rank 0 logs: the f32 warning and, without `metis` (threads have no
    /// MPI handle for PT-Scotch), the `Block` fallback; an explicit partitioner has none.
    #[test]
    fn gmsh_read_warns_once_on_rank_0() {
        let _ = log::set_logger(&LOGGER);
        log::set_max_level(log::LevelFilter::Warn);
        let path = fixture("triangle_p2.msh");
        let fallback = usize::from(!cfg!(feature = "metis"));
        let got = gridcl_testing::ThreadComm::run(2, |comm| {
            let client = gridcl_testing::test_client();
            read::<f32, _>(&client, comm.clone(), &path, GmshOptions::default()).unwrap();
            let first = my_warnings();
            let options = GmshOptions::default().partitioner(crate::Block);
            read::<f32, _>(&client, comm, &path, options).unwrap();
            (first, my_warnings())
        });
        let (first, second) = &got[0];
        assert_eq!(first.len(), 1 + fallback, "{first:?}");
        assert!(
            first.iter().all(|w| w.contains("triangle_p2.msh")),
            "{first:?}"
        );
        assert_eq!(second.len(), 2 + fallback, "{second:?}");
        assert_eq!(got[1], (vec![], vec![]));
    }

    /// A device without f64 gives `UnsupportedPrecision` before the file is read.
    #[test]
    fn gmsh_read_f64_needs_device_support() {
        let client = gridcl_testing::test_client();
        let mut props = client.properties().clone();
        props.features.types.elem.remove(&f64::cube_type());
        gridcl_device::testing::with_properties(props, || {
            let r = read::<f64, _>(
                &client,
                Serial,
                fixture("triangle_p1.msh"),
                GmshOptions::default(),
            );
            assert!(matches!(
                r,
                Err(GridError::UnsupportedPrecision { scalar: "f64", .. })
            ));
        });
    }

    /// A tag element whose corners are vertices but not an edge of the mesh (the other diagonal
    /// of a square cut into two triangles); a missing file.
    #[test]
    fn gmsh_read_unmatched_tag_element_and_io() {
        let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
            $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
            $Elements\n2 3 1 3\n2 1 2 2\n1 1 2 3\n2 1 3 4\n1 1 1 1\n3 2 4\n$EndElements\n";
        let path = std::env::temp_dir().join(format!("gridcl-t11-{}.msh", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let client = gridcl_testing::test_client();
        let r = read::<f32, _>(&client, Serial, &path, GmshOptions::default());
        let _ = std::fs::remove_file(&path);
        assert!(
            matches!(r, Err(GridError::UnmatchedTagElement { dim: 1, tag: 3 })),
            "{:?}",
            r.err()
        );
        let r = read::<f32, _>(
            &client,
            Serial,
            fixture("no-such-file.msh"),
            GmshOptions::default(),
        );
        assert!(matches!(r, Err(GridError::Io(_))));
    }

    /// Physical names without elements, in D_TOP and below: empty groups of size 0.
    #[test]
    fn gmsh_read_named_groups_without_elements() {
        let text = "$MeshFormat\n4.1 0 8\n$EndMeshFormat\n\
            $PhysicalNames\n2\n2 1 \"domain\"\n1 5 \"empty\"\n$EndPhysicalNames\n\
            $Nodes\n1 4 1 4\n2 1 0 4\n1\n2\n3\n4\n0 0 0\n1 0 0\n1 1 0\n0 1 0\n$EndNodes\n\
            $Elements\n1 2 1 2\n2 1 2 2\n1 1 2 3\n2 1 3 4\n$EndElements\n";
        let path = std::env::temp_dir().join(format!("gridcl-m4r-{}.msh", std::process::id()));
        std::fs::write(&path, text).unwrap();
        let client = gridcl_testing::test_client();
        let r = read::<f32, _>(&client, Serial, &path, GmshOptions::default());
        let _ = std::fs::remove_file(&path);
        let mesh = PlanarMesh::<f32>::try_from(r.unwrap()).unwrap();
        let tags = mesh.tags();
        for (codim, name, dim, value) in [(Codim::Vol, "domain", 2, 1), (Codim::Bnd, "empty", 1, 5)]
        {
            assert_eq!(tags.group_size(dim, value), Some(0), "{name}");
            let t = tags.named(codim, name).unwrap();
            assert_eq!(
                (t.dim, t.indices.len(), t.values.len()),
                (dim, 0, 0),
                "{name}"
            );
        }
    }

    /// Cells are tagged in flat numbering over the blocks; tag elements through the entities.
    #[test]
    fn gmsh_read_tags_cells_and_entities() {
        let client = gridcl_testing::test_client();
        let any = read::<f32, _>(
            &client,
            Serial,
            fixture("two_regions_p1.msh"),
            GmshOptions::default(),
        )
        .unwrap();
        let mesh = PlanarMesh::<f32>::try_from(any).unwrap();
        let tags = mesh.tags();
        // 26 triangles in block 0, 9 quadrilaterals in block 1.
        let quads = tags.named(Codim::Vol, "quadrilaterals").unwrap();
        assert_eq!(
            quads.indices.to_host().unwrap(),
            (26..35).collect::<Vec<u32>>()
        );
        assert_eq!(tags.named(Codim::Vol, "all").unwrap().indices.len(), 35);
        // "boundary" in dimensions 1 and 0.
        assert_eq!(
            tags.named(Codim::Bnd, "boundary").unwrap().indices.len(),
            18
        );
        assert_eq!(
            tags.named(Codim::BBnd, "boundary").unwrap().indices.len(),
            4
        );
        assert_eq!(tags.lookup(1, "interface"), Some(11));
        // Every cell and every tag element has an elementary tag.
        assert_eq!(tags.elementary(2).unwrap().indices.len(), 35);
        assert_eq!(tags.elementary(1).unwrap().indices.len(), 21);
        assert_eq!(tags.elementary(0).unwrap().indices.len(), 6);
    }

    /// On one rank the numbered file is M4's: `from_gmsh_mesh`, which numbers the non-vertex
    /// nodes and the tag elements itself and shifts by its own minima, writes the same file
    /// from the same host mesh, for every mesh fixture.
    #[test]
    fn gmsh_write_one_rank_equals_m4() {
        use gridcl_io::gmsh::{FileNumbering, from_gmsh_mesh, from_numbered_gmsh_mesh};
        let client = gridcl_testing::test_client();
        let shift = |min: Option<i64>| min.filter(|&m| m < 1).map_or(0, |m| 1 - m);
        let dir = fixture("");
        let mut names: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name().into_string().unwrap())
            .filter(|n| n.ends_with(".msh") && !n.starts_with("duplicate_"))
            .collect();
        names.sort();
        for name in names {
            let any = read::<f32, _>(&client, Serial, fixture(&name), GmshOptions::default())
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let (m4, numbered) = crate::with_mesh!(any.as_ref(), m => {
                let n = crate::io_download::gmsh_numbering(m).unwrap();
                let host = gathered(m, &n).unwrap();
                let numbering = FileNumbering {
                    node_shift: shift(n.node_min),
                    element_shift: shift(n.cell_min),
                    partition: None,
                };
                (from_gmsh_mesh(&host).unwrap(), from_numbered_gmsh_mesh(&host, &numbering).unwrap())
            });
            assert_eq!(numbered, m4, "{name}");
        }
        let mesh = builder_mesh::<f32>(NodeFamily::Equispaced, 2).unwrap();
        let n = crate::io_download::gmsh_numbering(&mesh).unwrap();
        let host = gathered(&mesh, &n).unwrap();
        let numbering = FileNumbering {
            node_shift: shift(n.node_min),
            element_shift: shift(n.cell_min),
            partition: None,
        };
        assert_eq!(numbering.node_shift, 1);
        assert_eq!(
            from_numbered_gmsh_mesh(&host, &numbering).unwrap(),
            from_gmsh_mesh(&host).unwrap()
        );
    }

    /// A builder mesh of `family`: a 2 × 2 square of quadrilaterals of degree `p`, ids from 0.
    fn builder_mesh<F: Real>(family: NodeFamily, p: u32) -> Result<PlanarMesh<F>> {
        let client = gridcl_testing::test_client();
        let h = gridcl_testing::square_box(crate::Cell2::Quadrilateral, 2, p);
        let b = &h.blocks[0];
        let element = CoordinateElement::<D2> {
            cell: crate::Cell2::Quadrilateral,
            degree: b.degree,
            family,
        };
        MeshBuilder::new(&client)
            .nodes(&h.nodes, None)
            .cell_block(element, &b.cells, None)
            .build()
    }

    /// Without tags and with origin ids from 0: written with ids shifted by one, read back with
    /// the same cells, coordinates and one elementary tag 1.
    #[test]
    fn gmsh_write_untagged_builder_mesh() {
        let client = gridcl_testing::test_client();
        let mesh = builder_mesh::<f32>(NodeFamily::Equispaced, 2).unwrap();
        let path = std::env::temp_dir().join(format!("gridcl-t13-{}.msh", std::process::id()));
        GmshWriter::new(&path).write(&mesh).unwrap();
        let back = PlanarMesh::<f32>::try_from(
            read::<f32, _>(&client, Serial, &path, GmshOptions::default()).unwrap(),
        )
        .unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let ids =
            |m: &PlanarMesh<f32>| decode_origin_ids(&m.vertex_origin_ids().to_host().unwrap());
        let cells = |m: &PlanarMesh<f32>| {
            decode_origin_ids(&m.cell_blocks()[0].origin_ids.to_host().unwrap())
        };
        let shifted: Vec<i64> = ids(&mesh).iter().map(|i| i + 1).collect();
        assert_eq!(ids(&back), shifted);
        assert_eq!(
            cells(&back),
            cells(&mesh).iter().map(|i| i + 1).collect::<Vec<_>>()
        );
        assert_eq!(
            mesh.nodes().to_host().unwrap(),
            back.nodes().to_host().unwrap()
        );
        let elementary = back.tags().elementary(2).unwrap();
        assert!(elementary.values.to_host().unwrap().iter().all(|&v| v == 1));
        assert!(back.tags().physical(2).is_none() && back.tags().names().next().is_none());
        // One entity, tag 1, holding the node block.
        assert!(text.contains("$Entities\n0 0 1 0\n1 "), "{text}");
    }

    /// A P3 GLL mesh (the box's equispaced nodes taken as GLL nodes) is written as its
    /// equispaced interpolant: read back, every node equals `equispaced_nodes` with the origin
    /// added, to the precision's tolerance (ADR-0022, item 9).
    fn writes_gll_as_equispaced<F: Real>(name: &str, tol: f64) {
        let client = gridcl_testing::test_client();
        let mesh = builder_mesh::<F>(NodeFamily::Gll, 3).unwrap();
        let path =
            std::env::temp_dir().join(format!("gridcl-m5-gll-{name}-{}.msh", std::process::id()));
        GmshWriter::new(&path).write(&mesh).unwrap();
        let back = PlanarMesh::<F>::try_from(
            read::<F, _>(&client, Serial, &path, GmshOptions::default()).unwrap(),
        )
        .unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            back.geometry_block(0).element.family,
            NodeFamily::Equispaced,
            "{name}"
        );
        let absolute = |m: &PlanarMesh<F>, x: Vec<F>| -> Vec<f64> {
            let origin = m.origin();
            x.iter()
                .enumerate()
                .map(|(i, v)| v.widen() + origin.as_ref()[i % 2])
                .collect()
        };
        let want = absolute(
            &mesh,
            gridcl_geometry::equispaced_nodes(&mesh)
                .unwrap()
                .to_host()
                .unwrap(),
        );
        let got = absolute(&back, back.nodes().to_host().unwrap());
        let raw = absolute(&mesh, mesh.nodes().to_host().unwrap());
        assert_eq!(got.len(), want.len(), "{name}");
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!(
                (a - b).abs() <= tol * b.abs().max(1.0),
                "{name}: coordinate {i}: {a} vs {b}"
            );
        }
        assert!(
            raw.iter().zip(&want).any(|(a, b)| (a - b).abs() > 1e-3),
            "{name}: the conversion moves nodes"
        );
    }

    #[test]
    fn gmsh_write_converts_gll() {
        writes_gll_as_equispaced::<f32>("f32", 1e-5);
        if gridcl_testing::supports::<f64>(&gridcl_testing::test_client()) {
            writes_gll_as_equispaced::<f64>("f64", 1e-12);
        }
    }
}
