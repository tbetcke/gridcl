//! Mesh families, with cells as items (docs/plans/M3s.md, question 9), each timing the smallest
//! call whose launches all take the family's plan ("One timed region per family"):
//!
//! - TopologyKeys (u32): `gridcl_topology::bench::emit_keys` for edges and faces of a P1
//!   tetrahedron box, the cell-fastest copy included;
//! - PermutationInfo (u32): `gridcl_topology::bench::permutation_info` on the same box;
//! - EVector (f32, f64): `DeviceGrid::evector` of a P2 tetrahedron box, on a fresh `try_clone`
//!   made outside the timing;
//! - GeomFactors (f32, f64): the four block functions X, J, det J and K, on a P2 tetrahedron box
//!   and a P2 triangle surface in 3D of about half the cells each, with a degree-4 rule. Each
//!   field is dropped right after its call, so at most one field is alive.
//!
//! The boxes have the box side nearest to the nominal size. Vertex origin ids are a random
//! permutation of the node indices, so the orientation words vary and fit u32. Checks compare a
//! sample of cells (every cell up to [`SAMPLE`], else an odd stride, so every lane position is
//! seen) and the padding with host references in f64.

use std::any::Any;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use cubecl::prelude::Client;
use gridcl::{
    Cell2, Cell3, CellBlock, CellType, D2, D3, DeviceBuffer, DeviceGrid, DeviceStrided, EmbeddedIn,
    FieldShape, IndexMap, Mesh, QuadField, QuadRule, Real, Serial, TopoDim, VECTOR_PAD, detj_block,
    jacobians_block, physical_points_block, pseudo_inverse_block,
};
use gridcl_device::KernelKind;
use gridcl_refcell::{gauss_rule, orientation_bits, sub_entities};
use gridcl_testing::HostMesh;
use gridcl_topology::{Topology, bench, encode_origin_ids};

use super::device::SplitMix;
use crate::protocol::Workload;
use crate::record::Scalar;

/// Seed of the vertex origin ids.
const SEED: u64 = 0x6772_6964_636c_0011;
/// Cells compared in full; above this, about this many at an odd stride.
pub const SAMPLE: usize = 20_000;
/// Degree of exactness of the GeomFactors rule.
const RULE_DEGREE: u32 = 4;
/// Degree of the EVector and GeomFactors meshes.
const P: u32 = 2;

/// The workload of `kind` in `scalar` at about `n` cells, or `None` if this module does not
/// measure it.
pub fn workload(
    client: &Client,
    kind: KernelKind,
    scalar: Scalar,
    n: usize,
) -> Option<Result<Box<dyn Workload>, String>> {
    fn boxed<W: Workload + 'static>(w: Result<W, String>) -> Result<Box<dyn Workload>, String> {
        w.map(|w| Box::new(w) as Box<dyn Workload>)
    }
    Some(match (kind, scalar) {
        (KernelKind::TopologyKeys, Scalar::U32) => boxed(Keys::new(client, n)),
        (KernelKind::PermutationInfo, Scalar::U32) => boxed(Bits::new(client, n)),
        (KernelKind::EVector, Scalar::F32) => boxed(EVectors::<f32>::new(client, n)),
        (KernelKind::EVector, Scalar::F64) => boxed(EVectors::<f64>::new(client, n)),
        (KernelKind::GeomFactors, Scalar::F32) => boxed(Factors::<f32>::new(client, n)),
        (KernelKind::GeomFactors, Scalar::F64) => boxed(Factors::<f64>::new(client, n)),
        _ => return None,
    })
}

/// The side s of the tetrahedron box (6 s³ cells) nearest to `cells`.
pub fn tetrahedron_side(cells: usize) -> usize {
    nearest_side(cells, |s| 6 * s * s * s)
}

/// The side s of the hexahedron box (s³ cells) nearest to `cells`.
pub fn hexahedron_side(cells: usize) -> usize {
    nearest_side(cells, |s| s * s * s)
}

/// The side s of the triangle box (2 s² cells) nearest to `cells`.
pub fn triangle_side(cells: usize) -> usize {
    nearest_side(cells, |s| 2 * s * s)
}

/// The cells of the GeomFactors workload at nominal size `n`: a tetrahedron box and a triangle
/// surface of about n / 2 cells each.
pub fn factor_cells(n: usize) -> usize {
    let s = tetrahedron_side(n / 2);
    let t = triangle_side(n / 2);
    6 * s * s * s + 2 * t * t
}

/// The smallest s ≥ 1 whose count is nearest to `cells`; ties go to the smaller box.
fn nearest_side(cells: usize, count: impl Fn(usize) -> usize) -> usize {
    let mut s = 1;
    while count(s + 1) <= cells {
        s += 1;
    }
    if count(s) < cells && count(s + 1) - cells < cells - count(s) {
        s + 1
    } else {
        s
    }
}

/// The meshes of the workloads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Generator {
    /// `cube_box` of tetrahedra, 6 s³ cells.
    Tetrahedra,
    /// `square_box` of triangles, 2 s² cells, lifted into 3D.
    Surface,
}

/// Host meshes by (generator, side, degree).
type HostMeshes = HashMap<(Generator, usize, u32), Rc<HostMesh>>;

thread_local! {
    /// Shared by the scalar types of a sweep.
    static HOST_MESHES: RefCell<HostMeshes> = RefCell::new(HashMap::new());
}

/// The host mesh of `generator` at `side` and degree `p`, generated once per thread.
fn host_mesh(generator: Generator, side: usize, p: u32) -> Rc<HostMesh> {
    HOST_MESHES.with(|cache| {
        cache
            .borrow_mut()
            .entry((generator, side, p))
            .or_insert_with(|| {
                Rc::new(match generator {
                    Generator::Tetrahedra => gridcl_testing::cube_box(Cell3::Tetrahedron, side, p),
                    Generator::Surface => {
                        lifted(&gridcl_testing::square_box(Cell2::Triangle, side, p))
                    }
                })
            })
            .clone()
    })
}

/// The planar mesh `h` on the plane z = (x + 2y) / 4 in 3D, so that no Jacobian entry is 0.
fn lifted(h: &HostMesh) -> HostMesh {
    let nodes = h
        .nodes
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|&[x, y]| [x, y, 0.25 * (x + 2.0 * y)])
        .collect();
    HostMesh {
        gdim: 3,
        nodes,
        ..h.clone()
    }
}

fn device_error(e: gridcl::GridError) -> String {
    e.to_string()
}

/// The check tolerance of `F`: 1e-5 relative in f32, 1e-12 in f64 (CLAUDE.md).
fn tolerance<F: Real>() -> f64 {
    if size_of::<F>() == 4 { 1e-5 } else { 1e-12 }
}

/// The cells a check compares: all below [`SAMPLE`] cells, else every s-th cell for an odd s
/// (odd, so the sample meets every position within a vector) and the last cell.
pub fn sample(cells: usize) -> Vec<usize> {
    if cells <= SAMPLE {
        return (0..cells).collect();
    }
    let stride = (cells / SAMPLE) | 1;
    let mut out: Vec<usize> = (0..cells).step_by(stride).collect();
    if out.last() != Some(&(cells - 1)) {
        out.push(cells - 1);
    }
    out
}

/// A random permutation of 0..n as origin ids.
fn permuted_ids(n: usize) -> Vec<i64> {
    let mut ids: Vec<i64> = (0..n as i64).collect();
    let mut rng = SplitMix(SEED);
    for i in (1..n).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        ids.swap(i, j);
    }
    ids
}

/// A P1 tetrahedron box as a `Topology`, with node indices as vertex indices and
/// [`permuted_ids`] as vertex origin ids, with the host cells and ids.
struct TetTopology {
    topology: Topology<D3>,
    cells: Vec<u32>,
    ids: Vec<i64>,
}

impl TetTopology {
    fn new(client: &Client, n: usize) -> Result<Self, String> {
        let h = host_mesh(Generator::Tetrahedra, tetrahedron_side(n), 1);
        let cells = h.blocks[0].cells.clone();
        let num_cells = cells.len() / 4;
        let ids = permuted_ids(h.num_nodes());
        let cell_ids: Vec<i64> = (0..num_cells as i64).collect();
        let block = CellBlock::<D3>::new(
            Cell3::Tetrahedron,
            IndexMap::new(&Serial, num_cells).map_err(device_error)?,
            DeviceStrided {
                data: DeviceBuffer::from_host(client, &cells).map_err(device_error)?,
                stride: 4,
            },
            DeviceBuffer::from_host(client, &encode_origin_ids(&cell_ids)).map_err(device_error)?,
            DeviceBuffer::zeros(client, num_cells).map_err(device_error)?,
        );
        let topology = Topology::new(
            client,
            vec![block],
            IndexMap::new(&Serial, ids.len()).map_err(device_error)?,
            &ids,
        )
        .map_err(device_error)?;
        Ok(TetTopology {
            topology,
            cells,
            ids,
        })
    }

    fn num_cells(&self) -> usize {
        self.cells.len() / 4
    }
}

/// TopologyKeys: the keys and payloads of edges (dimension 1) and faces (dimension 2).
struct Keys {
    t: TetTopology,
}

impl Keys {
    fn new(client: &Client, n: usize) -> Result<Self, String> {
        Ok(Keys {
            t: TetTopology::new(client, n)?,
        })
    }
}

impl Workload for Keys {
    fn items(&self) -> usize {
        self.t.num_cells()
    }

    /// Reads the four vertices per dimension; writes, per edge and face, its words and payload.
    fn bytes(&self) -> Option<usize> {
        let words: usize = (1..3)
            .flat_map(|dim| sub_entities(CellType::Tetrahedron, dim))
            .map(|e| e.len() + 1)
            .sum();
        Some(self.items() * (2 * 4 + words) * size_of::<u32>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let edges = bench::emit_keys(&self.t.topology, 1).map_err(device_error)?;
        let faces = bench::emit_keys(&self.t.topology, 2).map_err(device_error)?;
        Ok(Box::new([edges, faces]))
    }

    /// Key i · padded + c holds the sorted vertices of sub-entity i of cell c, word w at
    /// w · plane + e, with payload e; padding cells hold all-ones keys.
    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let out = out
            .downcast::<[bench::BenchOutput; 2]>()
            .map_err(|_| "keys: unexpected output".to_owned())?;
        let num_cells = self.t.num_cells();
        let padded = num_cells.next_multiple_of(VECTOR_PAD);
        let cells = sample(num_cells);
        for (dim, out) in [1, 2].into_iter().zip(out.iter()) {
            let entities = sub_entities(CellType::Tetrahedron, dim);
            let words = entities[0].len();
            let host = out.to_host().map_err(device_error)?;
            let [keys, payload] = host.as_slice() else {
                return Err(format!("keys of dimension {dim}: {} buffers", host.len()));
            };
            let total = entities.len() * padded;
            let plane = keys.len() / words;
            if payload.len() != total || plane < total || keys.len() != plane * words {
                return Err(format!(
                    "keys of dimension {dim}: {} words and {} payloads for {total} keys",
                    keys.len(),
                    payload.len()
                ));
            }
            let positions = cells.iter().copied().chain(num_cells..padded);
            for c in positions {
                for (i, entity) in entities.iter().enumerate() {
                    let e = i * padded + c;
                    let mut want: Vec<u32> = if c < num_cells {
                        entity.iter().map(|&v| self.t.cells[4 * c + v]).collect()
                    } else {
                        vec![u32::MAX; words]
                    };
                    want.sort_unstable();
                    let got: Vec<u32> = (0..words).map(|w| keys[w * plane + e]).collect();
                    if got != want || payload[e] != e as u32 {
                        return Err(format!(
                            "keys of dimension {dim}: cell {c}, entity {i}: key {got:?}, \
                             payload {}; want {want:?}, {e}",
                            payload[e]
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// PermutationInfo: the orientation word of every cell.
struct Bits {
    t: TetTopology,
}

impl Bits {
    fn new(client: &Client, n: usize) -> Result<Self, String> {
        Ok(Bits {
            t: TetTopology::new(client, n)?,
        })
    }
}

impl Workload for Bits {
    fn items(&self) -> usize {
        self.t.num_cells()
    }

    /// Reads four vertices and their low id words, writes one word, per cell.
    fn bytes(&self) -> Option<usize> {
        Some(self.items() * 9 * size_of::<u32>())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let words = bench::permutation_info(&self.t.topology, 0).map_err(device_error)?;
        Ok(Box::new(words))
    }

    /// Against `gridcl_refcell::orientation_bits` of the cell's vertex origin ids.
    fn check(&mut self, out: Box<dyn Any>) -> Result<(), String> {
        let out = out
            .downcast::<bench::BenchOutput>()
            .map_err(|_| "permutation info: unexpected output".to_owned())?;
        let host = out.to_host().map_err(device_error)?;
        let [words] = host.as_slice() else {
            return Err(format!("permutation info: {} buffers", host.len()));
        };
        let num_cells = self.t.num_cells();
        if words.len() != num_cells {
            return Err(format!(
                "permutation info: {} words for {num_cells} cells",
                words.len()
            ));
        }
        for c in sample(num_cells) {
            let ids: Vec<i64> = (0..4)
                .map(|v| self.t.ids[self.t.cells[4 * c + v] as usize])
                .collect();
            let want = orientation_bits(CellType::Tetrahedron, &ids);
            if words[c] != want {
                return Err(format!(
                    "permutation info: cell {c}: {:#x}, want {want:#x}",
                    words[c]
                ));
            }
        }
        Ok(())
    }
}

/// The device's nodes (relative to the origin) and dofmap of block 0, read once for the
/// checks.
struct HostGeometry<F> {
    nodes: Vec<F>,
    dofmap: Vec<u32>,
    nodes_per_cell: usize,
    num_cells: usize,
}

impl<F: Real> HostGeometry<F> {
    fn new<TD: TopoDim + EmbeddedIn<D3>>(mesh: &Mesh<F, TD, D3>) -> Result<Self, String> {
        let dofmap = &mesh.geometry_block(0).dofmap;
        Ok(HostGeometry {
            nodes: mesh.nodes().to_host().map_err(device_error)?,
            dofmap: dofmap.data.to_host().map_err(device_error)?,
            nodes_per_cell: dofmap.stride as usize,
            num_cells: mesh.cell_blocks()[0].index_map.num_local(),
        })
    }

    /// Coordinate g of node k of cell c, relative to the origin.
    fn node(&self, c: usize, k: usize, g: usize) -> F {
        self.nodes[self.dofmap[c * self.nodes_per_cell + k] as usize * 3 + g]
    }
}

/// EVector: the E-vector of a P2 tetrahedron box, built on a fresh copy of the mesh.
struct EVectors<F: Real> {
    mesh: Mesh<F, D3, D3>,
    fresh: Option<Mesh<F, D3, D3>>,
    host: HostGeometry<F>,
}

impl<F: Real> EVectors<F> {
    fn new(client: &Client, n: usize) -> Result<Self, String> {
        let h = host_mesh(Generator::Tetrahedra, tetrahedron_side(n), P);
        let mesh: Mesh<F, D3, D3> = h.build(client).map_err(device_error)?;
        Ok(EVectors {
            host: HostGeometry::new(&mesh)?,
            mesh,
            fresh: None,
        })
    }
}

impl<F: Real> Workload for EVectors<F> {
    fn items(&self) -> usize {
        self.host.num_cells
    }

    /// Reads the dofmap and gathers three coordinates per node; writes them.
    fn bytes(&self) -> Option<usize> {
        let per_node = size_of::<u32>() + 2 * 3 * size_of::<F>();
        Some(self.items() * self.host.nodes_per_cell * per_node)
    }

    /// A copy of the mesh with an empty E-vector cache.
    fn prepare(&mut self) -> Result<(), String> {
        self.fresh = None;
        self.fresh = Some(self.mesh.try_clone().map_err(device_error)?);
        Ok(())
    }

    /// The E-vector stays in the copy until the next `prepare`.
    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        let fresh = self.fresh.as_ref().ok_or("evector: no fresh mesh")?;
        fresh.evector(0).map_err(device_error)?;
        Ok(Box::new(()))
    }

    /// Bitwise against the gather on the host, padding cells included (copies of the last).
    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        let fresh = self.fresh.as_ref().ok_or("evector: no fresh mesh")?;
        let xe = fresh.evector(0).map_err(device_error)?;
        let h = &self.host;
        let cells = sample(h.num_cells);
        let mut covered = 0;
        for chunk in xe.chunks() {
            let data = chunk.data.to_host().map_err(device_error)?;
            let stride = chunk.num_cells.next_multiple_of(VECTOR_PAD);
            covered += chunk.num_cells;
            let local = cells
                .iter()
                .filter(|&&c| (chunk.first_cell..chunk.first_cell + chunk.num_cells).contains(&c))
                .map(|&c| c - chunk.first_cell)
                .chain(chunk.num_cells..stride);
            for c in local {
                let cell = (chunk.first_cell + c).min(h.num_cells - 1);
                for k in 0..h.nodes_per_cell {
                    for g in 0..3 {
                        let got = data[(k * 3 + g) * stride + c];
                        let want = h.node(cell, k, g);
                        if got != want {
                            return Err(format!(
                                "evector: cell {}, node {k}, coordinate {g}: {got:?}, \
                                 want {want:?}",
                                chunk.first_cell + c
                            ));
                        }
                    }
                }
            }
        }
        if covered != h.num_cells {
            return Err(format!(
                "evector: chunks hold {covered} cells, not {}",
                h.num_cells
            ));
        }
        Ok(())
    }
}

/// One mesh of the GeomFactors workload, with its point set and host data.
struct FactorMesh<F: Real, TD: TopoDim + EmbeddedIn<D3>> {
    mesh: Mesh<F, TD, D3>,
    rule: QuadRule<F, TD>,
    cell: TD::Cell,
    host: HostGeometry<F>,
    /// The reference points, `[point][tdim]`.
    points: Vec<f64>,
    /// The box side, which bounds the cancellation in J (see [`FactorMesh::check`]).
    side: usize,
}

impl<F: Real, TD: TopoDim + EmbeddedIn<D3>> FactorMesh<F, TD> {
    fn new(client: &Client, h: &HostMesh, side: usize) -> Result<Self, String> {
        let mesh: Mesh<F, TD, D3> = h.build(client).map_err(device_error)?;
        let rule = QuadRule::for_grid(&mesh, RULE_DEGREE).map_err(device_error)?;
        let cell = mesh.cell_blocks()[0].cell;
        // The E-vector is built once here, outside the timing; the factor kernels read it.
        mesh.evector(0).map_err(device_error)?;
        Ok(FactorMesh {
            host: HostGeometry::new(&mesh)?,
            points: gauss_rule(cell.into(), RULE_DEGREE).points,
            mesh,
            rule,
            cell,
            side,
        })
    }

    fn num_points(&self) -> usize {
        self.points.len() / TD::N
    }

    /// Reads the E-vector and writes X (3), J (3 · tdim), det J (1) and K (tdim · 3) values per
    /// point.
    fn bytes(&self) -> usize {
        let read = 4 * self.host.nodes_per_cell * 3;
        let written = self.num_points() * (3 + 3 * TD::N + 1 + TD::N * 3);
        self.host.num_cells * (read + written) * size_of::<F>()
    }

    /// The four block functions, each field dropped after its call.
    fn run(&self) -> Result<(), String> {
        let pts = self.rule.get(self.cell).map_err(device_error)?;
        drop(physical_points_block(&self.mesh, 0, pts).map_err(device_error)?);
        drop(jacobians_block(&self.mesh, 0, pts).map_err(device_error)?);
        drop(detj_block(&self.mesh, 0, pts).map_err(device_error)?);
        drop(pseudo_inverse_block(&self.mesh, 0, pts).map_err(device_error)?);
        Ok(())
    }

    /// Repeats the four calls one at a time and compares each field with the affine map of the
    /// cell's vertices (the meshes are affine), in f64 from the device's nodes.
    ///
    /// X is compared within the tolerance relative to max(1, |X|). J = Σ_k x_k ∇φ_k sums
    /// coordinates of size up to 1 into entries of size 1/side, so J, det J and K are compared
    /// relative to their largest entry within the tolerance times κ = 4 · nodes per cell · side
    /// (4 bounds the P2 basis gradients), and det J and K within a further factor 4 for the
    /// conditioning of J.
    fn check(&self, label: &str) -> Result<(), String> {
        let pts = self.rule.get(self.cell).map_err(device_error)?;
        let tol = tolerance::<F>();
        let kappa = 4.0 * self.host.nodes_per_cell as f64 * self.side as f64;
        let origin = self.mesh.origin();
        let what = |f: &str| format!("{f} of the {label}");
        let field = physical_points_block(&self.mesh, 0, pts).map_err(device_error)?;
        self.compare(&what("X"), &field, |a, q, out| {
            for (g, x) in out.iter_mut().enumerate() {
                *x = origin[g] + a.at(&self.points[q * TD::N..(q + 1) * TD::N], g);
            }
            tol * out.iter().fold(1.0f64, |m, x| m.max(x.abs()))
        })?;
        drop(field);
        let field = jacobians_block(&self.mesh, 0, pts).map_err(device_error)?;
        self.compare(&what("J"), &field, |a, _, out| {
            for t in 0..TD::N {
                out[t * 3..(t + 1) * 3].copy_from_slice(&a.j[t]);
            }
            tol * kappa * max_abs(out)
        })?;
        drop(field);
        let field = detj_block(&self.mesh, 0, pts).map_err(device_error)?;
        self.compare(&what("det J"), &field, |a, _, out| {
            out[0] = a.det();
            tol * 4.0 * kappa * out[0].abs()
        })?;
        drop(field);
        let field = pseudo_inverse_block(&self.mesh, 0, pts).map_err(device_error)?;
        self.compare(&what("K"), &field, |a, _, out| {
            a.pseudo_inverse(out);
            tol * 4.0 * kappa * max_abs(out)
        })?;
        Ok(())
    }

    /// Compares the sampled cells of `field` with `want(cell map, point, values)`, which fills
    /// the reference values and returns the absolute tolerance.
    fn compare<S: FieldShape>(
        &self,
        what: &str,
        field: &QuadField<F, S>,
        want: impl Fn(&Affine, usize, &mut [f64]) -> f64,
    ) -> Result<(), String> {
        let (tile, q_len, len) = (field.tile(), field.num_points(), S::LEN);
        if field.num_cells() != self.host.num_cells || q_len != self.num_points() {
            return Err(format!(
                "{what}: {} cells of {q_len} points, want {} of {}",
                field.num_cells(),
                self.host.num_cells,
                self.num_points()
            ));
        }
        let cells = sample(self.host.num_cells);
        let mut values = vec![0.0; len];
        for chunk in field.chunks() {
            let data = chunk.data.to_host().map_err(device_error)?;
            let range = chunk.first_cell..chunk.first_cell + chunk.num_cells;
            for &cell in cells.iter().filter(|c| range.contains(c)) {
                let a = Affine::new::<F>(&self.host, cell, TD::N);
                let c = cell - chunk.first_cell;
                for q in 0..q_len {
                    let bound = want(&a, q, &mut values);
                    for (k, &w) in values.iter().enumerate() {
                        let got = data[((c / tile) * q_len * len + q * len + k) * tile + c % tile];
                        if (got.widen() - w).abs() > bound {
                            return Err(format!(
                                "{what}: cell {cell}, point {q}, component {k}: {}, want {w} \
                                 (tolerance {bound:e})",
                                got.widen()
                            ));
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0f64, |m, v| m.max(v.abs()))
}

/// The affine map x = v0 + Σ_t ξ_t (v_{t+1} − v0) of a simplex cell, relative to the origin
/// (Appendix A.1: the axes of triangles and tetrahedra are the vertices 1, 2 and 3).
struct Affine {
    v0: [f64; 3],
    /// Column t of J: v_{t+1} − v0.
    j: Vec<[f64; 3]>,
}

impl Affine {
    fn new<F: Real>(h: &HostGeometry<F>, c: usize, tdim: usize) -> Affine {
        let v = |k: usize| -> [f64; 3] { std::array::from_fn(|g| h.node(c, k, g).widen()) };
        let v0 = v(0);
        let j = (1..=tdim)
            .map(|k| {
                let vk = v(k);
                std::array::from_fn(|g| vk[g] - v0[g])
            })
            .collect();
        Affine { v0, j }
    }

    /// Coordinate g at the reference point `xi`.
    fn at(&self, xi: &[f64], g: usize) -> f64 {
        self.v0[g]
            + self
                .j
                .iter()
                .zip(xi)
                .map(|(col, x)| x * col[g])
                .sum::<f64>()
    }

    /// G = JᵀJ.
    fn gram(&self) -> Vec<Vec<f64>> {
        let dot = |a: &[f64; 3], b: &[f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        self.j
            .iter()
            .map(|a| self.j.iter().map(|b| dot(a, b)).collect())
            .collect()
    }

    /// det J for tdim = 3, sqrt(det JᵀJ) for tdim = 2.
    fn det(&self) -> f64 {
        match self.j.len() {
            3 => {
                let [a, b, c] = [self.j[0], self.j[1], self.j[2]];
                a[0] * (b[1] * c[2] - b[2] * c[1]) - b[0] * (a[1] * c[2] - a[2] * c[1])
                    + c[0] * (a[1] * b[2] - a[2] * b[1])
            }
            2 => {
                let g = self.gram();
                (g[0][0] * g[1][1] - g[0][1] * g[1][0]).sqrt()
            }
            n => unreachable!("no affine factors for tdim {n}"),
        }
    }

    /// K = (JᵀJ)⁻¹Jᵀ (tdim × 3), column-major: `out[g · tdim + t]`.
    fn pseudo_inverse(&self, out: &mut [f64]) {
        let td = self.j.len();
        let g = self.gram();
        let inv: Vec<Vec<f64>> = match td {
            2 => {
                let d = g[0][0] * g[1][1] - g[0][1] * g[1][0];
                vec![
                    vec![g[1][1] / d, -g[0][1] / d],
                    vec![-g[1][0] / d, g[0][0] / d],
                ]
            }
            3 => {
                let m = |r: usize, c: usize| g[r % 3][c % 3];
                let d = self.det().powi(2);
                (0..3)
                    .map(|r| {
                        (0..3)
                            .map(|c| {
                                (m(c + 1, r + 1) * m(c + 2, r + 2)
                                    - m(c + 1, r + 2) * m(c + 2, r + 1))
                                    / d
                            })
                            .collect()
                    })
                    .collect()
            }
            n => unreachable!("no affine factors for tdim {n}"),
        };
        for gd in 0..3 {
            for t in 0..td {
                out[gd * td + t] = (0..td).map(|s| inv[t][s] * self.j[s][gd]).sum();
            }
        }
    }
}

/// GeomFactors: X, J, det J and K on a P2 tetrahedron box and a P2 triangle surface in 3D.
struct Factors<F: Real> {
    volume: FactorMesh<F, D3>,
    surface: FactorMesh<F, D2>,
}

impl<F: Real> Factors<F> {
    fn new(client: &Client, n: usize) -> Result<Self, String> {
        let (s, t) = (tetrahedron_side(n / 2), triangle_side(n / 2));
        Ok(Factors {
            volume: FactorMesh::new(client, &host_mesh(Generator::Tetrahedra, s, P), s)?,
            surface: FactorMesh::new(client, &host_mesh(Generator::Surface, t, P), t)?,
        })
    }
}

impl<F: Real> Workload for Factors<F> {
    fn items(&self) -> usize {
        self.volume.host.num_cells + self.surface.host.num_cells
    }

    fn bytes(&self) -> Option<usize> {
        Some(self.volume.bytes() + self.surface.bytes())
    }

    fn run(&mut self) -> Result<Box<dyn Any>, String> {
        self.volume.run()?;
        self.surface.run()?;
        Ok(Box::new(()))
    }

    /// The fields are not kept (at 10^7 cells in f64 they would need about 30 GB), so the
    /// check repeats the calls at the same configuration, one field at a time.
    fn check(&mut self, _: Box<dyn Any>) -> Result<(), String> {
        self.volume.check("tetrahedron box")?;
        self.surface.check("triangle surface")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sides_are_nearest() {
        assert_eq!(tetrahedron_side(1), 1);
        assert_eq!(tetrahedron_side(1000), 5); // 750 is nearer than 1296
        assert_eq!(tetrahedron_side(100_000), 26); // 105 456
        assert_eq!(tetrahedron_side(10_000_000), 119); // 10 110 954
        assert_eq!(hexahedron_side(1000), 10);
        assert_eq!(hexahedron_side(1_000_000), 100);
        assert_eq!(triangle_side(500), 16); // 512
        assert_eq!(triangle_side(5_000_000), 1581);
        assert_eq!(factor_cells(1000), 384 + 512);
    }

    #[test]
    fn samples_meet_every_lane_and_the_last_cell() {
        assert_eq!(sample(5), [0, 1, 2, 3, 4]);
        let s = sample(10 * SAMPLE + 7);
        assert_eq!(*s.last().unwrap(), 10 * SAMPLE + 6);
        let lanes: std::collections::BTreeSet<usize> = s.iter().map(|c| c % 16).collect();
        assert_eq!(lanes.len(), 16);
        assert!(s.len() <= SAMPLE + 2);
    }

    #[test]
    fn permuted_ids_are_a_permutation() {
        let mut ids = permuted_ids(1000);
        assert_ne!(ids, (0..1000).collect::<Vec<i64>>());
        ids.sort_unstable();
        assert_eq!(ids, (0..1000).collect::<Vec<i64>>());
    }

    #[test]
    fn affine_factors_of_a_scaled_triangle() {
        // v0 = 0, v1 = (2, 0, 0), v2 = (0, 3, 0): det = 6, K = diag(1/2, 1/3) in the plane.
        let a = Affine {
            v0: [0.0; 3],
            j: vec![[2.0, 0.0, 0.0], [0.0, 3.0, 0.0]],
        };
        assert_eq!(a.det(), 6.0);
        let mut k = [0.0; 6];
        a.pseudo_inverse(&mut k);
        assert_eq!(k, [0.5, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0]);
        assert_eq!(a.at(&[0.5, 0.5], 1), 1.5);
    }

    #[test]
    fn affine_inverse_of_a_tetrahedron() {
        let a = Affine {
            v0: [0.0; 3],
            j: vec![[1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 1.0, 4.0]],
        };
        assert_eq!(a.det(), 8.0);
        let mut k = [0.0; 9];
        a.pseudo_inverse(&mut k);
        // K J = I, with K[t][g] at g · 3 + t and J[g][s] = j[s][g].
        for t in 0..3 {
            for s in 0..3 {
                let ks: f64 = (0..3).map(|g| k[g * 3 + t] * a.j[s][g]).sum();
                assert!((ks - f64::from(t == s)).abs() < 1e-14, "({t}, {s}): {ks}");
            }
        }
    }
}
