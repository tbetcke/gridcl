//! `Geometry<F, TD, GD>` (ADR-0012): geometry blocks, the node L-vector, the node `IndexMap`,
//! the origin and the E-vector cache.

use core::cell::OnceCell;

use tetraloma_device::{DeviceBuffer, GridError, Real, Result};
use tetraloma_dims::{Dim, EmbeddedIn, TopoDim};
use tetraloma_parallel::{CommDyn, IndexMap};

use crate::strategy::StrategyFamily;
use crate::{EVector, GeometryBlock};

// Design: "Geometry", "Precision policy"; ADR-0012.
/// The geometry of a mesh: one [`GeometryBlock`] per cell block, the node coordinates, the
/// node index map and the origin, with a cache of the blocks' [`EVector`]s.
///
/// The node coordinates are stored relative to the origin, the centre of the mesh's bounding
/// box, so that f32 meshes far from zero keep their accuracy ([Tolerances]). `Mesh` holds one
/// and exposes it through [`DeviceGrid`](crate::DeviceGrid); a grid type of your own can
/// hold one too. [Lagrange geometry] explains the map that the blocks define.
///
/// # Example
///
/// ```
/// use tetraloma_device::{DeviceBuffer, DeviceStrided};
/// use tetraloma_dims::{Cell1, D1, D2};
/// use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
/// use tetraloma_parallel::{IndexMap, Serial};
///
/// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
/// # let client = tetraloma_testing::test_client();
/// // Two nodes of a segment in the plane, relative to the origin (1, 2).
/// let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
/// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
/// let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
/// let geometry = Geometry::<f32, D1, D2>::new(
///     vec![GeometryBlock { element, dofmap }],
///     nodes,
///     IndexMap::new(&Serial, 2).unwrap(),
///     [1.0, 2.0],
/// )
/// .unwrap();
/// assert_eq!(geometry.origin(), [1.0, 2.0]);
/// assert_eq!(geometry.nodes().len(), 4);
/// ```
///
/// [Tolerances]: https://tetracl.github.io/tetraloma/concepts/precision.html#tolerances
/// [Lagrange geometry]: https://tetracl.github.io/tetraloma/concepts/geometry.html#coordinate-elements
pub struct Geometry<F: Real, TD: TopoDim, GD: Dim> {
    blocks: Vec<GeometryBlock<TD>>,
    nodes: DeviceBuffer<F>,
    node_index_map: IndexMap,
    origin: GD::Array<f64>,
    /// The origin narrowed to `F`, uploaded at creation; the factor kernels add it back.
    origin_device: DeviceBuffer<F>,
    evectors: Vec<OnceCell<EVector<F>>>,
}

impl<F, TD, GD> Geometry<F, TD, GD>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    /// The geometry of `blocks` over `nodes`: `gdim` coordinates per local node, node after
    /// node, relative to `origin`, with `node_index_map` numbering the owned and ghost nodes.
    ///
    /// The E-vectors are built later, on first use. See [Lagrange geometry].
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `nodes` does not hold `gdim` values per local node, if a dofmap's
    /// stride is not its element's node count or its length not a whole number of rows, or if
    /// there are no nodes but a dofmap is not empty (`expected: 0`); allocation and device
    /// errors. Node indices themselves are not checked: the E-vector kernel clamps an index
    /// past the local nodes to the last node.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided, GridError};
    /// use tetraloma_dims::{Cell1, D1, D2};
    /// use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// use tetraloma_parallel::{IndexMap, Serial};
    ///
    /// // `client`: a CubeCL client, for example `cubecl::Device::cpu()?.client()`.
    /// # let client = tetraloma_testing::test_client();
    /// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let block = || GeometryBlock {
    ///     element,
    ///     dofmap: DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 },
    /// };
    /// // Two nodes in the plane need four coordinates.
    /// let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0])?;
    /// let geometry = Geometry::<f32, D1, D2>::new(vec![block()], nodes, IndexMap::new(&Serial, 2)?, [1.0, 2.0])?;
    /// assert_eq!(geometry.blocks().len(), 1);
    /// let three = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5])?;
    /// assert!(matches!(
    ///     Geometry::<f32, D1, D2>::new(vec![block()], three, IndexMap::new(&Serial, 2)?, [1.0, 2.0]),
    ///     Err(GridError::LengthMismatch { expected: 4, found: 3 })
    /// ));
    /// # Ok::<(), GridError>(())
    /// ```
    ///
    /// [Lagrange geometry]: https://tetracl.github.io/tetraloma/concepts/geometry.html#coordinate-elements
    pub fn new(
        blocks: Vec<GeometryBlock<TD>>,
        nodes: DeviceBuffer<F>,
        node_index_map: IndexMap,
        origin: GD::Array<f64>,
    ) -> Result<Self> {
        let expected = node_index_map
            .num_local()
            .checked_mul(GD::N)
            .ok_or(GridError::TooLarge)?;
        if nodes.len() != expected {
            return Err(GridError::LengthMismatch {
                expected,
                found: nodes.len(),
            });
        }
        for block in &blocks {
            let k = block.nodes_per_cell();
            if block.dofmap.stride as usize != k || !block.dofmap.data.len().is_multiple_of(k) {
                return Err(GridError::LengthMismatch {
                    expected: k,
                    found: block.dofmap.stride as usize,
                });
            }
            // Cells without nodes would read only the padding of the L-vector, which `uninit`
            // leaves unspecified, so their E-vectors would not be deterministic.
            if expected == 0 && block.dofmap.data.len() != 0 {
                return Err(GridError::LengthMismatch {
                    expected: 0,
                    found: block.dofmap.data.len(),
                });
            }
        }
        let evectors = blocks.iter().map(|_| OnceCell::new()).collect();
        let narrowed: Vec<F> = origin.as_ref().iter().map(|&x| F::narrow(x)).collect();
        let origin_device = DeviceBuffer::from_host(nodes.client(), &narrowed)?;
        Ok(Geometry {
            blocks,
            nodes,
            node_index_map,
            origin,
            origin_device,
            evectors,
        })
    }

    /// The geometry blocks, one per cell block.
    ///
    /// See [Lagrange geometry].
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// assert_eq!(geometry.blocks().len(), 1);
    /// assert_eq!(geometry.blocks()[0].num_cells(), 1);
    /// ```
    ///
    /// [Lagrange geometry]: https://tetracl.github.io/tetraloma/concepts/geometry.html#coordinate-elements
    pub fn blocks(&self) -> &[GeometryBlock<TD>] {
        &self.blocks
    }

    /// The node coordinates on the device: `gdim` values per local node, node after node,
    /// relative to the [`origin`](Geometry::origin).
    ///
    /// See [Tolerances] for why the coordinates are relative.
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// // A copy to the host, for inspection.
    /// assert_eq!(geometry.nodes().to_host().unwrap(), [-0.5, 0.0, 0.5, 0.0]);
    /// ```
    ///
    /// [Tolerances]: https://tetracl.github.io/tetraloma/concepts/precision.html#tolerances
    pub fn nodes(&self) -> &DeviceBuffer<F> {
        &self.nodes
    }

    /// The index map of the nodes: the owned nodes first, then the ghosts.
    ///
    /// See [Index maps].
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// let map = geometry.node_index_map();
    /// assert_eq!((map.num_owned(), map.num_ghosts()), (2, 0));
    /// ```
    ///
    /// [Index maps]: https://tetracl.github.io/tetraloma/tutorials/halo-exchange.html#index-maps
    pub fn node_index_map(&self) -> &IndexMap {
        &self.node_index_map
    }

    /// The origin of the node coordinates, in f64: the centre of the mesh's bounding box for
    /// a mesh from `MeshBuilder` or a reader.
    ///
    /// See [Tolerances].
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// assert_eq!(geometry.origin(), [1.0, 2.0]);
    /// ```
    ///
    /// [Tolerances]: https://tetracl.github.io/tetraloma/concepts/precision.html#tolerances
    pub fn origin(&self) -> GD::Array<f64> {
        self.origin
    }

    /// The E-vector of block `b`: built on the device on first use, then cached.
    ///
    /// See [`EVector`] and [Writing kernels].
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the first build.
    ///
    /// # Panics
    ///
    /// If there is no block `b`.
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// let xe = geometry.evector(0).unwrap();
    /// assert_eq!(xe.num_cells(), 1);
    /// // Later calls return the cached E-vector.
    /// assert!(std::ptr::eq(xe, geometry.evector(0).unwrap()));
    /// ```
    ///
    /// [Writing kernels]: https://tetracl.github.io/tetraloma/tutorials/kernels.html#geometryblock
    pub fn evector(&self, b: usize) -> Result<&EVector<F>> {
        let cache = &self.evectors[b];
        if let Some(xe) = cache.get() {
            return Ok(xe);
        }
        let block = &self.blocks[b];
        let s = crate::strategy::choose(
            self.nodes.client(),
            StrategyFamily::EVector,
            block.element.cell.into(),
            block.element.degree,
            size_of::<F>(),
            false,
            // A gather: no table for `shared` to stage.
            || Ok(None),
        )?;
        let xe = crate::evector::build(
            &self.nodes,
            &block.dofmap.data,
            block.num_cells(),
            block.nodes_per_cell(),
            GD::N,
            self.origin_device.share(),
            s,
        )?;
        Ok(cache.get_or_init(|| xe))
    }

    /// The allocated bytes of every device array (dofmaps, nodes, the uploaded origin, the
    /// E-vectors built so far and their cached basis coefficients) plus the
    /// [`footprint_bytes`](IndexMap::footprint_bytes) of the node index map. For memory
    /// accounting in tests (`Mesh::footprint_bytes`, ADR-0035); not part of the stable API.
    ///
    /// # Example
    ///
    /// ```
    /// use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// use tetraloma_dims::{Cell1, D1, D2};
    /// use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// use tetraloma_parallel::{IndexMap, Serial};
    ///
    /// let client = tetraloma_testing::test_client();
    /// let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// let geometry = Geometry::<f32, D1, D2>::new(
    ///     vec![GeometryBlock { element, dofmap }],
    ///     nodes,
    ///     IndexMap::new(&Serial, 2).unwrap(),
    ///     [1.0, 2.0],
    /// )
    /// .unwrap();
    /// // At least the nodes and the dofmap.
    /// let arrays = geometry.nodes().footprint_bytes() + geometry.blocks()[0].dofmap.data.footprint_bytes();
    /// assert!(geometry.footprint_bytes() > arrays);
    /// ```
    #[doc(hidden)]
    pub fn footprint_bytes(&self) -> usize {
        let dofmaps: usize = self
            .blocks
            .iter()
            .map(|b| b.dofmap.data.footprint_bytes())
            .sum();
        // An E-vector's origin is a share of `origin_device`.
        let evectors: usize = self
            .evectors
            .iter()
            .filter_map(OnceCell::get)
            .map(|xe| {
                let chunks: usize = xe.chunks().iter().map(|c| c.data.footprint_bytes()).sum();
                chunks + xe.coefficients_bytes()
            })
            .sum();
        dofmaps
            + evectors
            + self.nodes.footprint_bytes()
            + self.origin_device.footprint_bytes()
            + self.node_index_map.footprint_bytes()
    }

    /// A deep copy of every device array; the E-vector cache of the copy starts empty.
    ///
    /// The node index map, ghosts included, is cloned and shares its neighbourhoods with the
    /// original, so the copy is not collective and `comm` is unused. See [Lagrange geometry].
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the copies.
    ///
    /// # Example
    ///
    /// ```
    /// # use tetraloma_device::{DeviceBuffer, DeviceStrided};
    /// # use tetraloma_dims::{Cell1, D1, D2};
    /// # use tetraloma_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
    /// # use tetraloma_parallel::{IndexMap, Serial};
    /// // `geometry`: one segment in the plane from (0.5, 2) to (1.5, 2), its nodes relative to
    /// // the origin (1, 2), as in the example of `Geometry::new`.
    /// # let client = tetraloma_testing::test_client();
    /// # let nodes = DeviceBuffer::from_host(&client, &[-0.5f32, 0.0, 0.5, 0.0]).unwrap();
    /// # let element = CoordinateElement::<D1> { cell: Cell1::Interval, degree: 1, family: NodeFamily::Equispaced };
    /// # let dofmap = DeviceStrided { data: DeviceBuffer::from_host(&client, &[0u32, 1]).unwrap(), stride: 2 };
    /// # let geometry = Geometry::<f32, D1, D2>::new(
    /// #     vec![GeometryBlock { element, dofmap }],
    /// #     nodes,
    /// #     IndexMap::new(&Serial, 2).unwrap(),
    /// #     [1.0, 2.0],
    /// # )
    /// # .unwrap();
    /// // `Serial`: the communicator of a program on one process.
    /// let copy = geometry.try_clone(&Serial).unwrap();
    /// assert_eq!(copy.origin(), geometry.origin());
    /// assert_eq!(copy.nodes().to_host().unwrap(), geometry.nodes().to_host().unwrap());
    /// ```
    ///
    /// [Lagrange geometry]: https://tetracl.github.io/tetraloma/concepts/geometry.html#coordinate-elements
    pub fn try_clone(&self, comm: &dyn CommDyn) -> Result<Self> {
        let _ = comm;
        let blocks: Vec<GeometryBlock<TD>> = self
            .blocks
            .iter()
            .map(GeometryBlock::try_clone)
            .collect::<Result<_>>()?;
        let evectors = blocks.iter().map(|_| OnceCell::new()).collect();
        Ok(Geometry {
            blocks,
            nodes: self.nodes.deep_clone()?,
            node_index_map: self.node_index_map.clone(),
            origin: self.origin,
            origin_device: self.origin_device.deep_clone()?,
            evectors,
        })
    }
}
