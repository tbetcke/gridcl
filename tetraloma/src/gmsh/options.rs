//! The options of [`read`](super::read) (ADR-0022, item 5; ADR-0035, items 1 and 2).

use std::fmt;
use std::rc::Rc;

use tetraloma_device::Result;
use tetraloma_parallel::{Comm, CommDyn};
use tetraloma_partition::{DistGraph, Partitioner};

use crate::{CellOrdering, GhostMode};

// Design: "Import and export / Gmsh MSH 4.1"; ADR-0022, item 5; ADR-0035, rules 1, 2 and 6.
/// The options of [`read`](super::read), set with builder methods.
///
/// `Default` infers the geometric dimension GD from the coordinates, picks the default
/// partitioner on more than one rank (see [`partitioner`](GmshOptions::partitioner)), adds no
/// ghost cells and keeps the cell order of the file. On one rank only
/// [`gdim`](GmshOptions::gdim) and [`ordering`](GmshOptions::ordering) change the result.
///
/// `GmshOptions` holds its partitioner behind an `Rc`, so it is `Clone` but neither `Send` nor
/// `Sync`, like [`Mesh`](crate::Mesh).
///
/// The tutorial [Parallel I/O] explains each option on several ranks, and [Distributed meshes]
/// the pipeline that the options steer.
///
/// [Parallel I/O]: https://tetracl.github.io/tetraloma/tutorials/parallel-io.html#reading-gmsh-in-parallel
/// [Distributed meshes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html
///
/// # Example
///
/// ```
/// use tetraloma::prelude::*;
///
/// // Read a planar mesh into three dimensions, with one layer of ghost cells across facets.
/// let options = GmshOptions::default().gdim(3).ghosts(GhostMode::SharedFacet);
/// # let _ = options;
/// ```
#[derive(Clone)]
pub struct GmshOptions {
    pub(super) gdim: Option<usize>,
    pub(super) partitioner: Option<Rc<dyn Partitioner>>,
    pub(super) ghosts: GhostMode,
    pub(super) ordering: CellOrdering,
}

impl Default for GmshOptions {
    fn default() -> Self {
        GmshOptions {
            gdim: None,
            partitioner: None,
            ghosts: GhostMode::None,
            ordering: CellOrdering::Input,
        }
    }
}

impl fmt::Debug for GmshOptions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GmshOptions")
            .field("gdim", &self.gdim)
            .field("partitioner", &self.partitioner.as_ref().map(|p| p.name()))
            .field("ghosts", &self.ghosts)
            .field("ordering", &self.ordering)
            .finish()
    }
}

impl GmshOptions {
    /// Fixes the geometric dimension GD of the mesh that [`read`](super::read) returns.
    ///
    /// Without it, the reader infers GD from the nodes: 3 if any node has a z coordinate other
    /// than 0, else 2 if any has a y coordinate other than 0, else 1, and at least the
    /// topological dimension TD. A coordinate counts as 0 if its magnitude is below 10⁻¹² times
    /// the diameter of the mesh's bounding box. A file of triangles in the plane z = 0 thus
    /// reads as a planar mesh; with `gdim(3)` it reads as a surface mesh in three dimensions.
    /// `gdim` must be 1, 2 or 3 and at least the inferred dimension, or the read fails with
    /// `DimensionMismatch`. [Reading and writing files] shows how the file decides the
    /// dimensions.
    ///
    /// [Reading and writing files]: https://tetracl.github.io/tetraloma/tutorials/read-write.html#reading-gmsh
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // path: a planar mesh of triangles, a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/triangle_p1.msh");
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let any = gmsh::read::<f32, _>(&client, Serial, path, GmshOptions::default().gdim(3))?;
    /// assert_eq!((any.tdim(), any.gdim()), (2, 3));
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn gdim(self, gdim: usize) -> Self {
        GmshOptions {
            gdim: Some(gdim),
            ..self
        }
    }

    /// The partitioner that distributes the cells of a read on more than one rank.
    ///
    /// Without one, a read on more than one rank uses PT-Scotch with the feature `ptscotch` and
    /// an MPI communicator, otherwise METIS on rank 0 with the feature `metis`, otherwise
    /// [`Block`](crate::Block); in the last case rank 0 logs a warning. A read on one rank
    /// calls no partitioner. With [`Block`](crate::Block), a partitioned file whose partition
    /// count equals the rank count keeps its partitions: each cell goes to the rank of its
    /// partition.
    ///
    /// [Partitioners] compares the partitioners, and [Cargo features] lists the features that
    /// add them.
    ///
    /// [Partitioners]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#partitioners
    /// [Cargo features]: https://tetracl.github.io/tetraloma/features.html#partitioners
    ///
    /// # Examples
    ///
    /// `Block` on two ranks:
    ///
    /// ```
    /// use tetraloma::prelude::*;
    /// // ThreadComm: a test communicator whose ranks are threads of this process; an MPI
    /// // program uses `Mpi` instead.
    /// # use tetraloma_testing::ThreadComm;
    ///
    /// // path: the rectangle [0, 2] × [0, 1] in two regions, a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let areas = ThreadComm::run(2, |comm| {
    ///     // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    ///     # let client = tetraloma_testing::test_client();
    ///     let options = GmshOptions::default().partitioner(Block).ghosts(GhostMode::SharedFacet);
    ///     let mesh = PlanarMesh::<f32, ThreadComm>::try_from(gmsh::read(&client, comm, path, options)?)?;
    ///     mesh.measure(2)
    /// });
    /// for area in areas {
    ///     assert!((area? - 2.0).abs() < 1e-5);
    /// }
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// PT-Scotch on MPI, with the feature `ptscotch`, launched by `mpirun`:
    ///
    /// ```no_run
    /// # #[cfg(feature = "ptscotch")]
    /// # fn main() -> tetraloma::Result<()> {
    /// use tetraloma::prelude::*;
    ///
    /// let universe = mpi::initialize().expect("MPI is not initialised yet");
    /// let comm = Mpi::world(&universe);
    /// // client: the client of this rank's compute device.
    /// # let client = tetraloma_testing::test_client();
    /// let any: AnyMesh<f64, Mpi> = gmsh::read(&client, comm.clone(), "head.msh",
    ///     GmshOptions::default().partitioner(PtScotch::default()).ghosts(GhostMode::SharedFacet))?;
    /// let head: VolumeMesh<f64, Mpi> = any.try_into()?;
    /// let owned: usize = head.cell_blocks().iter().map(|b| b.index_map.num_owned()).sum();
    /// println!("{owned} owned cells on rank {}", comm.rank());
    /// # Ok(())
    /// # }
    /// # #[cfg(not(feature = "ptscotch"))]
    /// # fn main() {}
    /// ```
    pub fn partitioner(self, p: impl Partitioner + 'static) -> Self {
        GmshOptions {
            partitioner: Some(Rc::new(p)),
            ..self
        }
    }

    /// The ghost cells that each rank adds to its owned cells; default [`GhostMode::None`].
    ///
    /// [Ghost modes] explains the modes. On one rank there are no ghost cells, whatever the
    /// mode.
    ///
    /// [Ghost modes]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#ghost-modes
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// let options = GmshOptions::default().ghosts(GhostMode::SharedVertex { layers: 1 });
    /// # let _ = options;
    /// ```
    pub fn ghosts(self, mode: GhostMode) -> Self {
        GmshOptions {
            ghosts: mode,
            ..self
        }
    }

    /// The order of the owned cells in each cell block; default [`CellOrdering::Input`], which
    /// on one rank is the order of the file.
    ///
    /// [Cell ordering] explains the orders and why a space-filling curve helps the memory
    /// access of kernels.
    ///
    /// [Cell ordering]: https://tetracl.github.io/tetraloma/tutorials/distributed.html#cell-ordering
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma::prelude::*;
    ///
    /// // path: a box of mixed cell types, a test file of tetraloma-io.
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../tetraloma-io/tests/fixtures/gmsh/hybrid_p1.msh");
    /// // client: a device client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let options = GmshOptions::default().ordering(CellOrdering::Hilbert);
    /// let mesh = VolumeMesh::<f32>::try_from(gmsh::read(&client, Serial, path, options)?)?;
    /// assert!(mesh.measure(3)? > 0.0);
    /// # Ok::<(), GridError>(())
    /// ```
    pub fn ordering(self, o: CellOrdering) -> Self {
        GmshOptions {
            ordering: o,
            ..self
        }
    }

    /// The partitioner of a read on `comm` (rule 6), and whether it is the `Block` fallback,
    /// which rank 0 announces with one warning.
    pub(super) fn partitioner_on<C: Comm>(&self, comm: &C) -> (Shared, bool) {
        if let Some(p) = &self.partitioner {
            return (Shared(Rc::clone(p)), false);
        }
        #[cfg(feature = "ptscotch")]
        if (comm as &dyn CommDyn).raw().is_some() {
            return (
                Shared(Rc::new(tetraloma_partition::PtScotch::default())),
                false,
            );
        }
        #[cfg(feature = "metis")]
        return (
            Shared(Rc::new(tetraloma_partition::Metis::default())),
            false,
        );
        #[cfg(not(feature = "metis"))]
        {
            let _ = comm;
            (Shared(Rc::new(tetraloma_partition::Block)), true)
        }
    }
}

/// The partitioner of [`GmshOptions`] as the builder's `Partitioner + 'static`.
pub(super) struct Shared(pub(super) Rc<dyn Partitioner>);

impl Partitioner for Shared {
    fn name(&self) -> &str {
        self.0.name()
    }

    fn partition(&self, comm: &dyn CommDyn, g: &DistGraph, nparts: usize) -> Result<Vec<i32>> {
        self.0.partition(comm, g, nparts)
    }
}
