//! The options of [`read`](super::read) (ADR-0022, item 5; ADR-0035, items 1 and 2).

use std::fmt;
use std::rc::Rc;

use gridcl_device::Result;
use gridcl_parallel::{Comm, CommDyn};
use gridcl_partition::{DistGraph, Partitioner};

use crate::{CellOrdering, GhostMode};

/// Options of [`read`](super::read). `Default` infers D_GEO, uses the default partitioner of
/// ADR-0035 (rule 6) on more than one rank, no ghost cells and the input cell order.
///
/// `GmshOptions` holds its partitioner behind an `Rc`, so it is `Clone` but neither `Send` nor
/// `Sync`, like [`Mesh`](crate::Mesh).
///
/// # Example
///
/// ```
/// use gridcl::prelude::*;
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
    /// Fixes D_GEO: 1–3, at least D_TOP and at least the inferred D_GEO.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/triangle_p1.msh");
    /// let client = gridcl_testing::test_client();
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

    /// The partitioner of a read on more than one rank (pipeline stage 3). Without one, the
    /// read uses PT-Scotch with the feature `ptscotch` and an MPI communicator, else METIS on
    /// rank 0 with `metis`, else [`Block`](crate::Block) with a warning (ADR-0035, rule 6). One rank calls no
    /// partitioner. [`Block`](crate::Block) keeps the partitions of a partitioned file whose partition
    /// count equals the rank count (ADR-0035, rule 5).
    ///
    /// # Examples
    ///
    /// `Block` on two threads:
    ///
    /// ```
    /// use gridcl::prelude::*;
    /// use gridcl_testing::ThreadComm;
    ///
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/two_regions_p1.msh");
    /// let areas = ThreadComm::run(2, |comm| {
    ///     let client = gridcl_testing::test_client();
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
    /// PT-Scotch on MPI (Listing B8), with the feature `ptscotch`, launched by `mpirun`:
    ///
    /// ```no_run
    /// # #[cfg(feature = "ptscotch")]
    /// # fn main() -> gridcl::Result<()> {
    /// use gridcl::prelude::*;
    ///
    /// let universe = mpi::initialize().expect("MPI is not initialised yet");
    /// let comm = Mpi::world(&universe);
    /// let client = gridcl_testing::test_client();
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

    /// The ghost cells of each rank; default [`GhostMode::None`].
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
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

    /// The order of the owned cells; default [`CellOrdering::Input`], which on one rank is
    /// file order within each cell block (ADR-0035, item 1).
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl::prelude::*;
    ///
    /// let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../gridcl-io/tests/fixtures/gmsh/hybrid_p1.msh");
    /// let client = gridcl_testing::test_client();
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
                Shared(Rc::new(gridcl_partition::PtScotch::default())),
                false,
            );
        }
        #[cfg(feature = "metis")]
        return (Shared(Rc::new(gridcl_partition::Metis::default())), false);
        #[cfg(not(feature = "metis"))]
        {
            let _ = comm;
            (Shared(Rc::new(gridcl_partition::Block)), true)
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
