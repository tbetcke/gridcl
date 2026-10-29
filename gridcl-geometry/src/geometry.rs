//! `Geometry<F, TD, GD>` (ADR-0012): geometry blocks, the node L-vector, the node `IndexMap`,
//! the origin and the E-vector cache.

use core::cell::OnceCell;

use gridcl_device::{DeviceBuffer, GridError, Real, Result};
use gridcl_dims::{Dim, EmbeddedIn, TopoDim};
use gridcl_parallel::{CommDyn, IndexMap};

use crate::{EVector, GeometryBlock};

/// The geometry of a mesh (ADR-0012).
///
/// # Example
///
/// ```
/// use gridcl_device::{DeviceBuffer, DeviceStrided};
/// use gridcl_dims::{Cell1, D1, D2};
/// use gridcl_geometry::{CoordinateElement, Geometry, GeometryBlock, NodeFamily};
/// use gridcl_parallel::{IndexMap, Serial};
///
/// let client = gridcl_testing::test_client();
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
    /// The geometry of `blocks` over `nodes` (`[node][gdim]`, relative to `origin`).
    ///
    /// # Errors
    ///
    /// `LengthMismatch` if `nodes` does not hold `gdim` values per local node, if a dofmap's
    /// stride is not its element's node count or its length not a whole number of rows, or if
    /// there are no nodes but a dofmap is not empty (`expected: 0`). Node indices themselves
    /// are not checked: the E-vector kernel clamps an index past the local nodes to the last
    /// node.
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
    pub fn blocks(&self) -> &[GeometryBlock<TD>] {
        &self.blocks
    }

    /// The L-vector `[node][gdim]`, relative to [`origin`](Geometry::origin).
    pub fn nodes(&self) -> &DeviceBuffer<F> {
        &self.nodes
    }

    /// The [owned | ghost] nodes.
    pub fn node_index_map(&self) -> &IndexMap {
        &self.node_index_map
    }

    /// The centre of the bounding box, in f64.
    pub fn origin(&self) -> GD::Array<f64> {
        self.origin
    }

    /// The E-vector of block `b`, built on first use and cached.
    ///
    /// # Errors
    ///
    /// Allocation and device errors of the first build.
    pub fn evector(&self, b: usize) -> Result<&EVector<F>> {
        let cache = &self.evectors[b];
        if let Some(xe) = cache.get() {
            return Ok(xe);
        }
        let block = &self.blocks[b];
        let xe = crate::evector::build(
            &self.nodes,
            &block.dofmap.data,
            block.num_cells(),
            block.nodes_per_cell(),
            GD::N,
            self.origin_device.share(),
        )?;
        Ok(cache.get_or_init(|| xe))
    }

    /// A deep copy of every device array; the E-vector cache starts empty. The node index
    /// map, ghosts included, is cloned and shares its neighbourhoods with the original, so the
    /// copy is not collective and `comm` is unused.
    ///
    /// # Errors
    ///
    /// Errors of `DeviceBuffer::deep_clone`.
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
