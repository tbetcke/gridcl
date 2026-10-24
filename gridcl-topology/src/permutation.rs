//! The PermutationInfo kernel (docs/design.md, "Permutation info"; A.6): one orientation word
//! per cell from its vertices' origin ids, generic over the vector size.
//!
//! The lanes of a vector are consecutive cells of one block. At vector size 1 the kernel
//! gathers each cell's (hi, lo) id words through the cell-major rows; above it, a `Scalar`
//! copy first gathers them into cell-fastest planes (`transpose`), which the kernel reads with
//! vector loads. Every comparison is an arithmetic mask (0 or 1 per lane), so no lane branches
//! on its values. The sub-entity tables are comptime constants, and when every id
//! fits in u32 the kernel compares the lo words only.

use cubecl::prelude::*;
use gridcl_device::{
    DeviceBuffer, DeviceStrided, GridError, KernelKind, LaunchPlan, Result, VECTOR_PAD,
    debug_assert_bindings, pad_len,
};
use gridcl_dims::CellType;
use gridcl_refcell::{sub_entities, sub_entity_types, tdim, vertices};

use crate::transpose;

/// The comptime tables of one cell type: its edges, and its faces as vertex cycles (A.6).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct PermTables {
    /// Vertices per cell.
    pub nv: usize,
    /// Edges as (first, second) local vertices.
    pub edges: Vec<(usize, usize)>,
    /// Faces as local vertices in cycle order: list order for triangles, 0–1–3–2 for
    /// quadrilaterals.
    pub faces: Vec<Vec<usize>>,
    /// Whether every id fits in u32: compare the lo words only.
    pub fast: bool,
}

impl PermTables {
    /// The tables of `cell` (dimension ≥ 2).
    pub fn new(cell: CellType, fast: bool) -> Self {
        let td = tdim(cell);
        let edges = sub_entities(cell, 1).iter().map(|e| (e[0], e[1])).collect();
        let faces = if td == 3 {
            sub_entities(cell, 2)
                .iter()
                .zip(sub_entity_types(cell, 2))
                .map(|(f, ty)| match ty {
                    CellType::Triangle => vec![f[0], f[1], f[2]],
                    _ => vec![f[0], f[1], f[3], f[2]],
                })
                .collect()
        } else {
            Vec::new()
        };
        PermTables {
            nv: vertices(cell).len(),
            edges,
            faces,
            fast,
        }
    }

    /// Bit position of the first face field.
    fn face_base(&self, f: usize) -> u32 {
        (self.edges.len() + 3 * f) as u32
    }
}

/// 1 in each lane where id a < id b, else 0. Ids are (hi, lo) pairs; with `fast` only the lo
/// words are compared.
#[cube]
#[allow(unused_assignments)] // both comptime branches assign `r`
fn less<N: Size>(
    a_hi: Vector<u32, N>,
    a_lo: Vector<u32, N>,
    b_hi: Vector<u32, N>,
    b_lo: Vector<u32, N>,
    #[comptime] fast: bool,
) -> Vector<u32, N> {
    let lo_lt = Vector::<u32, N>::cast_from(a_lo.less_than(&b_lo));
    let mut r = lo_lt;
    if comptime!(!fast) {
        let hi_lt = Vector::<u32, N>::cast_from(a_hi.less_than(&b_hi));
        let hi_eq = Vector::<u32, N>::cast_from(a_hi.equal(&b_hi));
        r = hi_lt + hi_eq * lo_lt;
    }
    r
}

/// The orientation word of each lane from the (hi, lo) id words of the cell's vertices.
#[cube]
fn orientation_word<N: Size>(
    hi: &Array<Vector<u32, N>>,
    lo: &Array<Vector<u32, N>>,
    #[comptime] t: PermTables,
) -> Vector<u32, N> {
    let mut bits = Vector::<u32, N>::broadcast(0u32);
    // One bit per edge: set when the first vertex has the higher id.
    #[unroll]
    for e in 0..comptime!(t.edges.len()) {
        let a = comptime!(t.edges[e].0);
        let b = comptime!(t.edges[e].1);
        let set = less::<N>(hi[b], lo[b], hi[a], lo[a], t.fast);
        bits |= set << Vector::<u32, N>::broadcast(comptime!(e as u32));
    }
    // Three bits per face: the rotation (the cycle position of the lowest vertex) and the
    // reflection (its successor is higher than its predecessor).
    #[unroll]
    for f in 0..comptime!(t.faces.len()) {
        let m = comptime!(t.faces[f].len());
        let mut rotation = Vector::<u32, N>::broadcast(0u32);
        let mut reflection = Vector::<u32, N>::broadcast(0u32);
        #[unroll]
        for k in 0..m {
            let a = comptime!(t.faces[f][k]);
            let mut is_min = Vector::<u32, N>::broadcast(1u32);
            #[unroll]
            for j in 0..m {
                if comptime!(j != k) {
                    let b = comptime!(t.faces[f][j]);
                    is_min *= less::<N>(hi[a], lo[a], hi[b], lo[b], t.fast);
                }
            }
            let succ = comptime!(t.faces[f][(k + 1) % m]);
            let pred = comptime!(t.faces[f][(k + m - 1) % m]);
            let up = less::<N>(hi[pred], lo[pred], hi[succ], lo[succ], t.fast);
            rotation += is_min * Vector::<u32, N>::broadcast(comptime!(k as u32));
            reflection += is_min * up;
        }
        let field = rotation | (reflection << Vector::<u32, N>::broadcast(2u32));
        bits |= field << Vector::<u32, N>::broadcast(comptime!(t.face_base(f)));
    }
    bits
}

/// The orientation word of every cell vector `v < num_vectors` (Listing B5 loop), from the
/// cell-major rows: each lane gathers its vertices' ids. Lanes past the last cell recompute the
/// last cell, so they read in bounds. Each vertex index is clamped to `last_vertex` lane by lane
/// (an arithmetic `min`, as in `gather`), so an index past the local vertices reads the last
/// vertex's ids instead of leaving `ids`. Launched at vector size 1, where the cell-fastest
/// copy would not pay.
///   vertices  [cell][nv] local vertex indices
///   ids       interleaved (hi, lo) origin id words per local vertex
///   out       one word per cell
#[cube(launch_unchecked)]
fn permutation_info_kernel<N: Size>(
    vertices: &[u32],
    ids: &[u32],
    out: &mut [Vector<u32, N>],
    num_cells: u32,
    last_vertex: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] t: PermTables,
) {
    let nv = t.nv;
    let last = num_cells - 1;
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let mut hi = Array::<Vector<u32, N>>::new(comptime!(t.nv));
            let mut lo = Array::<Vector<u32, N>>::new(comptime!(t.nv));
            #[unroll]
            for j in 0..nv {
                let mut h = Vector::<u32, N>::broadcast(0u32);
                let mut l = Vector::<u32, N>::broadcast(0u32);
                #[unroll]
                for lane in 0..N::value() {
                    let c = min((v * N::value() + lane) as u32, last) as usize;
                    let vertex = min(vertices[c * nv + j], last_vertex) as usize;
                    h.insert(lane, ids[2 * vertex]);
                    l.insert(lane, ids[2 * vertex + 1]);
                }
                hi[j] = h;
                lo[j] = l;
            }
            out[v] = orientation_word::<N>(&hi, &lo, comptime!(t.clone()));
        }
    }
}

/// [`permutation_info_kernel`] on the cell-fastest copy of the vertices' id words
/// (`transpose::origin_ids`): the lo words of vertex j at plane j, the hi words (unless
/// `t.fast`) at plane nv + j, each of `padded_cells` words. Every read is a vector load; the
/// copy's padding cells repeat the last cell.
///   words  the id planes
///   out    one word per cell
#[cube(launch_unchecked)]
fn permutation_info_planes_kernel<N: Size>(
    words: &[Vector<u32, N>],
    out: &mut [Vector<u32, N>],
    padded_cells: u32,
    num_vectors: u32,
    items_per_unit: u32,
    #[comptime] t: PermTables,
) {
    let nv = t.nv;
    let stride = padded_cells as usize / N::value();
    let first = ABSOLUTE_POS * items_per_unit as usize;
    for i in 0..items_per_unit as usize {
        let v = first + i;
        if v < num_vectors as usize {
            let mut hi = Array::<Vector<u32, N>>::new(comptime!(t.nv));
            let mut lo = Array::<Vector<u32, N>>::new(comptime!(t.nv));
            #[unroll]
            for j in 0..nv {
                lo[j] = words[j * stride + v];
                if comptime!(t.fast) {
                    hi[j] = Vector::<u32, N>::broadcast(0u32);
                } else {
                    hi[j] = words[(nv + j) * stride + v];
                }
            }
            out[v] = orientation_word::<N>(&hi, &lo, comptime!(t.clone()));
        }
    }
}

/// The orientation word of each of the `num_cells` cells of `cell` in `vertices`, from the
/// interleaved vertex origin ids `ids`; `fast` when every id fits in u32. `None` for cells of
/// dimension below 2, which have no bits. Above vector size 1 the launch reads a cell-fastest
/// copy of the id words, made right before it and dropped after it.
pub(crate) fn permutation_info(
    cell: CellType,
    vertices: &DeviceStrided<u32>,
    ids: &DeviceBuffer<u32>,
    fast: bool,
) -> Result<Option<DeviceBuffer<u32>>> {
    if tdim(cell) < 2 {
        return Ok(None);
    }
    let client = vertices.data.client();
    let tables = PermTables::new(cell, fast);
    let num_cells = vertices.data.len() / tables.nv;
    let out = DeviceBuffer::<u32>::uninit(client, num_cells)?;
    if num_cells == 0 {
        return Ok(Some(out));
    }
    let cells = u32::try_from(num_cells).map_err(|_| GridError::TooLarge)?;
    // The largest vertex index the kernel reads; 0 when there are no vertices (then it reads
    // the zeroed padding words 0 and 1 of `ids`).
    let last_vertex = u32::try_from(ids.len() / 2)
        .map_err(|_| GridError::TooLarge)?
        .saturating_sub(1);
    // The copy's plan comes first, so `record_plans` lists it right before the kernel's.
    let copy_plan = transpose::plan(client, num_cells)?;
    let plan = LaunchPlan::elementwise(
        client,
        KernelKind::PermutationInfo,
        num_cells,
        size_of::<u32>(),
        VECTOR_PAD,
        2 * tables.nv + 4,
    )?;
    let n = plan.vector_size;
    if n > 1 {
        let words = transpose::origin_ids(&copy_plan, vertices, num_cells, ids, last_vertex, fast)?;
        let padded = u32::try_from(pad_len(num_cells)?).map_err(|_| GridError::TooLarge)?;
        debug_assert_bindings(client, 2);
        // SAFETY: `out` holds pad_len(num_cells) = padded words, plan.num_vectors · n of them,
        // and the kernel writes only vectors below num_vectors. It reads `words` at
        // p · padded / n + v with p < 2 · nv (p < nv when fast) and v < num_vectors =
        // padded / n, within the copy's planes of padded words each; padded is a multiple of
        // VECTOR_PAD, which n divides.
        unsafe {
            permutation_info_planes_kernel::launch_unchecked(
                client,
                plan.cube_count,
                plan.cube_dim,
                n,
                words.vector_arg(n),
                out.vector_arg(n),
                padded,
                plan.num_vectors,
                plan.items_per_unit,
                tables,
            );
        }
        return Ok(Some(out));
    }
    debug_assert_bindings(client, 3);
    // SAFETY: `out` holds pad_len(num_cells) words, plan.num_vectors · n of them, and the kernel
    // writes only vectors below num_vectors. It reads `vertices` at c · nv + j with
    // c ≤ num_cells − 1 and j < nv, within its num_cells · nv entries (num_cells is
    // vertices.len() / nv). It reads `ids` at 2 · vertex and 2 · vertex + 1 with
    // vertex = min(vertices[..], last_vertex): with ids.len() = 2 · num_vertices ≥ 2 that is
    // at most ids.len() − 1; with no vertices it reads words 0 and 1, inside the allocation of
    // at least VECTOR_PAD words. No vertex index is trusted: Topology::new checks lengths only.
    unsafe {
        permutation_info_kernel::launch_unchecked(
            client,
            plan.cube_count,
            plan.cube_dim,
            n,
            vertices.data.arg(),
            ids.arg(),
            out.vector_arg(n),
            cells,
            last_vertex,
            plan.num_vectors,
            plan.items_per_unit,
            tables,
        );
    }
    Ok(Some(out))
}
