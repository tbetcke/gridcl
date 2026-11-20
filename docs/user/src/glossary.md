# Glossary

The terms that this book uses, in alphabetical order. Each entry gives a short definition and
links to the chapter that explains the term in full. Names in code font are types, functions or
fields of tetraloma or CubeCL; the [API documentation](api/tetraloma/index.html) describes them
in detail.

Affine map
: A map of the form $X(\xi) = x_0 + J \xi$ with a constant Jacobian $J$. Straight-sided
intervals, triangles and tetrahedra, and parallelograms, have affine maps.
See [The reference-to-physical map](concepts/reference-cells.md#the-reference-to-physical-map).

`AnyMesh`
: An enum with one variant per valid pair of topological and geometric dimension, each holding a
typed mesh. File readers return it, because the file decides the dimensions.
See [AnyMesh](concepts/typed-dimensions.md#anymesh).

Basix
: The finite element library of the FEniCS project. It uses tetraloma's reference cells but numbers
the edges and faces of triangles and tetrahedra in reverse.
See [Basix and DOLFINx](elements/basix-dolfinx.md).

`Block` partitioner
: The built-in default partitioner. It splits the cells, in input order, into contiguous pieces of
equal size, one per rank, and ignores the dual graph's edges.
See [Partitioners](tutorials/distributed.md#partitioners).

Block size
: The number of values per index in an array that a scatterer exchanges; the array holds
`block_size * num_local()` values.
See [Scatterers](tutorials/halo-exchange.md#scatterers).

`BlockFields`
: The result of a geometric-factor method: one `QuadField` per cell block, with one value (a
number, vector or matrix) per cell and quadrature point, kept on the device.
See [BlockFields](tutorials/kernels.md#blockfields).

Boundary mesh
: The facets of a mesh that belong to exactly one cell, returned by `boundary()` as a mesh one
dimension lower, with each cell's normal pointing out of its parent cell.
See [Boundary meshes](tutorials/tags-submeshes.md#boundary-meshes).

Cell
: One piece of a mesh: an interval, triangle, quadrilateral, tetrahedron, hexahedron, prism or
pyramid. All cells of a mesh have the mesh's topological dimension.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Cell block
: The cells of one cell type in a mesh, stored together so that every array of the block has a
fixed row length. A `CellBlock` holds the cell type, an index map, the vertices, origin ids and
flags of its cells.
See [Cell blocks](concepts/meshes.md#cell-blocks).

Cell-major order
: The order $[\text{cell}][\text{point}]$ in which `to_host_cell_major` returns a field at
quadrature points, cell by cell and point by point. On the device the field is stored in tiles
instead.
See [Geometric data](tutorials/geometric-data.md).

Cell ordering
: The order of a rank's owned cells, which is the order in which kernels visit them: input order,
Hilbert order or reverse Cuthill–McKee order, chosen with `CellOrdering`.
See [Cell ordering](tutorials/distributed.md#cell-ordering).

Chunk
: A piece of a field at quadrature points that fits into one device allocation. A `QuadChunk`
holds `num_cells` consecutive cells from `first_cell` on.
See [BlockFields](tutorials/kernels.md#blockfields).

Clamp box
: The box $[-0.1, 1.1]^t$ to which the pull-back confines its iterates. It holds every reference
cell with a margin of 0.1; a point whose preimage lies outside it does not converge.
See [The iteration](concepts/pull-back.md#the-iteration).

Client
: The CubeCL handle of one device, which owns its memory and launches its kernels. You get it
with `cubecl::Device::cpu()?.client()` or the constructor of another backend.
See [Opening a device](getting-started/first-program.md#opening-a-device).

Codimension
: The geometric dimension minus the topological dimension, $g - t$. Codimension 0 covers planar
and volume meshes, codimension 1 surfaces in 3D and curves in the plane, codimension 2 curves in
3D. The enum `Codim` of the tag methods counts down from the topological dimension instead.
See [Codimension](concepts/typed-dimensions.md#codimension).

Collapsed Gauss–Jacobi rule
: The Gauss rule of triangles, tetrahedra and pyramids: a tensor rule on the unit square or cube,
mapped onto the cell.
See [Quadrature rules](concepts/quadrature.md#quadrature-rules).

Collective call
: A call that every rank of a communicator makes, in the same order and with matching arguments,
such as `build_on`, `measure`, `orient`, a halo exchange or a write. A rank that skips one leaves
the others waiting.
See [Building on a communicator](tutorials/distributed.md#building-on-a-communicator).

Column-major
: The storage order of Jacobians and pseudo-inverses: column by column. Entry $(\gamma, d)$ of
$J$ is component $d\,g + \gamma$, and on the host `j[d][γ]` is $J_{\gamma d}$.
See [The Jacobian](concepts/geometric-factors.md#the-jacobian).

Communicator
: The set of processes a mesh lives on, the last type parameter `C` of a mesh: `Serial` for one
process, `Mpi` for the ranks of an MPI program.
See [Distributed meshes](tutorials/distributed.md).

Compressed sparse row (CSR)
: The storage of a table whose rows have different lengths: an array of all entries, row by row,
and an array of offsets where each row starts. Upward connectivity uses it.
See [Connectivity](concepts/meshes.md#connectivity).

Comptime parameter
: A parameter of a CubeCL kernel whose value is fixed when the kernel is compiled, so that CubeCL
compiles one kernel per combination of values and knows loop bounds in advance.
See [A kernel that integrates](tutorials/kernels.md#a-kernel-that-integrates).

Connectivity
: Which entities of one dimension touch which entities of another, named by a `ConnKey`.
Downward connectivity $(t, d)$ lists each cell's entities of dimension $d$, with a fixed row
length; upward connectivity $(d, t)$ lists the cells of each entity, in compressed sparse row
form.
See [Connectivity](concepts/meshes.md#connectivity).

Coordinate element
: The triple of cell type, degree $p$ and node family that defines the nodes and Lagrange basis
functions of a cell block's geometry (`CoordinateElement`).
See [Coordinate elements](concepts/geometry.md#coordinate-elements).

Cube, unit, cube count, cube dimension
: CubeCL's names for the parts of a kernel launch. A unit is one invocation of the kernel (a
thread); a cube is a group of units, whose number is the cube dimension; the cube count is the
number of cubes of the launch.
See [The launch shape](tutorials/kernels.md#the-launch-shape).

CubeCL
: The Rust language extension and runtime in which tetraloma's kernels are written once and
compiled for each backend.
See [Installation](getting-started/installation.md#runtime-features-of-cubecl).

Degree
: Of a coordinate element, the polynomial degree $p$ of the map from the reference cell; of a
quadrature rule, the degree $k$ up to which it integrates polynomials exactly.
See [Coordinate elements](concepts/geometry.md#coordinate-elements) and
[Quadrature rules](concepts/quadrature.md#quadrature-rules).

DefElement, ndelement
: DefElement is an encyclopedia of finite element definitions, and ndelement a library that
implements them. tetraloma follows their reference cells, numbering, node layout and orientation
rules.
See [Reference cell definitions](elements/reference-cells.md).

Determinant
: $\det J$, the factor by which the map from the reference cell scales volume in codimension 0.
It is negative where the map mirrors the reference cell.
See [The determinant](concepts/geometric-factors.md#the-determinant).

Determinism
: The property that the same program on the same device gives the same bits on every run. tetraloma
avoids floating-point atomics and sums in fixed orders to achieve it.
See [Determinism](concepts/precision.md#determinism).

Device
: The hardware that holds a mesh's data and runs the kernels: a GPU, or the CPU through the CubeCL
CPU runtime.
See [Compute devices](devices.md).

`DeviceGrid`
: The trait through which a downstream crate reads a mesh's device arrays: cell blocks, geometry
blocks, nodes, index maps and topology. Every mesh and submesh implements it.
See [DeviceGrid](tutorials/kernels.md#devicegrid).

DOF transformation
: A matrix that corrects the degrees of freedom on an edge or face of a cell according to the
entity's orientation in that cell.
See [Converting a mesh](elements/basix-dolfinx.md#converting-a-mesh).

Dofmap
: The field `dofmap` of a geometry block: for each cell, one row with the indices of its nodes in
the mesh's node array, in the order of the coordinate element.
See [GeometryBlock](tutorials/kernels.md#geometryblock).

DOLFINx
: The finite element solver of the FEniCS project, built on Basix.
See [Basix and DOLFINx](elements/basix-dolfinx.md).

Dual graph
: The graph whose vertices are the cells of a mesh and whose edges join cells that share a facet.
Partitioners divide it among the ranks.
See [Building on a communicator](tutorials/distributed.md#building-on-a-communicator).

Edge
: An entity of dimension 1. Its vertex list gives it a direction, from its first vertex to its
second.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Elementary tag
: Gmsh's number of the geometric entity that an element belongs to. The reader keeps it, and
`elementary(dim)` returns it.
See [Physical groups](tutorials/tags-submeshes.md#physical-groups).

Entity
: A vertex, edge, face or cell of a mesh. Edges and faces exist only after `create_entities`.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Entity block
: The entities of one dimension and one cell type, for example the triangular faces of a prism
mesh (`EntityBlock`).
See [Cell blocks](concepts/meshes.md#cell-blocks).

Entity map
: The map from a submesh back to its parent (`EntityMap`): the parent index of each child cell,
vertex and node, and the child index of each parent entity.
See [Regions](tutorials/tags-submeshes.md#regions).

Equispaced nodes
: Lagrange nodes on a uniform lattice, at the fractions $i/p$ of each edge. They exist on every
cell type, and Gmsh and VTK files use them.
See [GLL and equispaced nodes](concepts/geometry.md#gll-and-equispaced-nodes).

Face
: An entity of dimension 2. As a sub-entity, a face exists only in three-dimensional cells.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Facet
: An entity of dimension $t - 1$: the faces of a three-dimensional cell, the edges of a
two-dimensional one, the vertices of an interval.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Facet normal
: The outward unit normal of a cell of codimension 0 on one of its facets,
$n = K^{\mathsf T} \hat n / |K^{\mathsf T} \hat n|$ for the reference normal $\hat n$.
See [Facet normals](concepts/geometric-factors.md#facet-normals).

Facet rule
: A quadrature rule on the facets of a cell (`FacetQuadRule`): a Gauss rule on each facet's
reference cell, mapped into the cell's coordinates.
See [Facet rules](concepts/quadrature.md#facet-rules).

Facet-major
: The order of the points of a facet rule: all points of facet 0, then those of facet 1, and so
on.
See [Facet rules](concepts/quadrature.md#facet-rules).

Flat index
: The number of an entity across all blocks of its dimension: the first index of its block plus
its position inside it. Connectivity arrays hold flat indices.
See [Connectivity](concepts/meshes.md#connectivity).

Forward exchange
: The halo exchange from owners to ghosts: every ghost receives its owner's value.
See [Forward](tutorials/halo-exchange.md#forward).

Gauss rule
: tetraloma's quadrature rules: tensor products of Gauss–Legendre rules on intervals,
quadrilaterals and hexahedra, collapsed Gauss–Jacobi rules on triangles, tetrahedra and pyramids,
and a triangle rule times an interval rule on prisms.
See [Quadrature rules](concepts/quadrature.md#quadrature-rules).

Gauss–Newton method
: The iteration of the pull-back on a curve or surface, $t < g$: Newton's method for the
least-squares problem $\min_\xi |X(\xi) - x|^2$, with the pseudo-inverse in place of the inverse.
See [The iteration](concepts/pull-back.md#the-iteration).

Geometric dimension
: The number of coordinates of a point in space, $g$, the type parameter `GD` of a mesh.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Geometric factors
: The quantities that correct for the map between reference and physical cells: the Jacobian,
its determinant, its pseudo-inverse and the normals.
See [Geometric factors](concepts/geometric-factors.md).

Geometry block
: The geometry of one cell block (`GeometryBlock`): its coordinate element and its dofmap.
See [GeometryBlock](tutorials/kernels.md#geometryblock).

Ghost
: A local copy of a cell, vertex, node or entity that another rank owns. Local indices list the
owned objects first and the ghosts after them.
See [Ownership and ghosts](concepts/meshes.md#ownership-and-ghosts).

Ghost mode
: The choice of ghost cells a distributed mesh gets (`GhostMode`): none, the cells that share a
facet with an owned cell, or layers of cells that share a vertex.
See [Ghost modes](tutorials/distributed.md#ghost-modes).

`GhostLevel`
: The attribute of a PVTU header that says whether the pieces hold ghost cells; tetraloma writes 1
if any rank has ghost cells.
See [PVTU output](tutorials/parallel-io.md#pvtu-output).

GLL nodes
: Gauss–Lobatto–Legendre nodes: the end points of the interval and the roots of $P_p'$, mapped to
$[0, 1]$. They keep interpolation well conditioned at high degree, and exist on intervals,
quadrilaterals and hexahedra.
See [GLL and equispaced nodes](concepts/geometry.md#gll-and-equispaced-nodes).

Halo
: The ghost entries of an array with one entry per local entity.
See [Halo exchange](tutorials/halo-exchange.md).

Halo exchange
: The update of the halo between ranks, forward from owners to ghosts or in reverse from ghosts
to owners.
See [Halo exchange](tutorials/halo-exchange.md).

Hilbert ordering
: A cell ordering that sorts the owned cells along a Hilbert curve through their vertex
centroids, so that cells close in space are close in memory.
See [Cell ordering](tutorials/distributed.md#cell-ordering).

Index map
: The description of how a set of objects is shared between ranks (`IndexMap`): the number of
owned objects and their global range, and the global index and owner of each ghost.
See [Index maps](tutorials/halo-exchange.md#index-maps).

Input order
: The default cell ordering: the order of the input, by source rank and then by position.
See [Cell ordering](tutorials/distributed.md#cell-ordering).

Jacobian
: The $g \times t$ matrix $J$ of first derivatives of the map $X$ from the reference cell, with
entries $J_{\gamma d} = \partial X_\gamma / \partial \xi_d$, stored column-major.
See [The Jacobian](concepts/geometric-factors.md#the-jacobian).

Lagrange basis
: The basis functions $\phi_k$ of a coordinate element, with $\phi_k = 1$ at node $k$ and 0 at
every other node.
See [Coordinate elements](concepts/geometry.md#coordinate-elements).

Last step
: The size $|\Delta|_\infty$ of a point's last pull-back step, returned as `last_step`. A point
has converged when it is at most the tolerance.
See [Convergence and failure](concepts/pull-back.md#convergence-and-failure).

Launch shape
: The cube count and cube dimension of a kernel launch. A downstream kernel chooses its own from
the device's properties.
See [The launch shape](tutorials/kernels.md#the-launch-shape).

`local_gpu_index`
: The function that returns a rank's index among the ranks of its node, the GPU that the rank
should open.
See [One GPU per MPI rank](devices.md#one-gpu-per-mpi-rank).

Low-to-high rule
: The rule that fixes the stored orientation of edges and faces by origin ids: an edge runs from
its lower origin id to its higher, and a face starts at its lowest.
See [Permutation bits](concepts/orientation.md#permutation-bits).

Manifold
: Here, a mesh in which every facet belongs to at most two cells. `orient`, `boundary` and
`facet_submesh` require it.
See [Manifold orientation](concepts/orientation.md#manifold-orientation).

Manifold orientation
: A consistent choice of normal over a surface or a curve in the plane, recorded in the cells'
orientation flags by `orient`.
See [Manifold orientation](concepts/orientation.md#manifold-orientation).

Measure
: The length, area or volume of a cell or a mesh: the integral of $\det J$, or of the
pseudo-determinant, over the reference cell. `measure(degree)` sums it over the owned cells of
all ranks and returns an `f64`.
See [Measure](concepts/geometric-factors.md#measure).

MPI-IO
: The parallel file interface of MPI, through which every rank writes its own partition of a
VTKHDF file.
See [VTKHDF output](tutorials/parallel-io.md#vtkhdf-output).

Node
: A point that carries a coordinate, part of the geometry. A straight-sided cell has one node per
vertex; a curved cell has further nodes on its edges, faces or interior.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Node family
: The placement of the Lagrange nodes along each edge (`NodeFamily`): equispaced or
Gauss–Lobatto–Legendre.
See [GLL and equispaced nodes](concepts/geometry.md#gll-and-equispaced-nodes).

Node map
: The permutation between the node order of a file format (Gmsh, VTK) and tetraloma's: entry $k$
is tetraloma's index of the file's $k$-th node.
See [File formats](elements/file-formats.md).

Normal
: The unit normal of a codimension-1 mesh, from the columns of $J$ and the cell's orientation
flag. Curves in 3D and volume meshes have none.
See [Normals](concepts/geometric-factors.md#normals).

NVRTC, PTX
: PTX is NVIDIA's intermediate assembly language, which the driver compiles for the GPU. NVRTC is
the CUDA toolkit's run-time compiler of CUDA C++, which CubeCL's alternative compiler `cuda-cpp`
uses.
See [CUDA](devices.md#cuda).

Orientation flag
: Bit 0 of a cell's word in the cell block's `flags`. Where it is set, the cell's normal is
reversed.
See [Manifold orientation](concepts/orientation.md#manifold-orientation).

Origin
: The centre of the bounding box of a mesh's nodes, kept in `f64` on the host. The device stores
coordinates relative to it, and `origin()` returns it.
See [Tolerances](concepts/precision.md#tolerances).

Origin id
: A 64-bit integer that names a vertex, node or cell independently of the partition: a Gmsh tag,
the id you pass to the builder, or the position in the input. Orientation and the choice of owners
use origin ids.
See [Origin ids](concepts/meshes.md#origin-ids).

Owned, owner
: Every cell and entity belongs to exactly one rank, its owner; on that rank it is owned, on every
other rank that holds it, a ghost.
See [Ownership and ghosts](concepts/meshes.md#ownership-and-ghosts).

Partitioner
: The algorithm that decides which rank owns each cell, a type that implements `Partitioner`:
`Block`, or `Metis`, `Scotch`, `PtScotch` and `ParMetis` behind Cargo features.
See [Partitioners](tutorials/distributed.md#partitioners).

Permutation bits
: The 32-bit word per cell that records how the cell sees each of its edges and faces relative to
their stored orientation: one bit per edge, three per face. `permutation_info(b)` returns the
words of cell block `b`.
See [Permutation bits](concepts/orientation.md#permutation-bits) and
[Orientation bits](elements/orientation-bits.md).

Physical group
: A named set of entities of one dimension in a Gmsh file. It becomes a tag value of the mesh.
See [Physical groups](tutorials/tags-submeshes.md#physical-groups).

Piece
: One rank's part of a PVTU file: a VTU file with the rank's owned and ghost cells.
See [PVTU output](tutorials/parallel-io.md#pvtu-output).

Plane, plane size
: A group of units of a cube that execute in lockstep, a warp on NVIDIA GPUs. tetraloma's kernels
are correct for every plane size, including the size 1 of the CPU runtime.
See [The CPU runtime](devices.md#the-cpu-runtime).

Pseudo-determinant
: $\sqrt{\det(J^{\mathsf T} J)}$, the volume factor of a map whose Jacobian is not square: the
length of the tangent of a curve, the area factor of a surface.
See [The determinant](concepts/geometric-factors.md#the-determinant).

Pseudo-inverse
: $K = (J^{\mathsf T} J)^{-1} J^{\mathsf T}$, a $t \times g$ matrix with $K J = I_t$. It equals
$J^{-1}$ in codimension 0.
See [The pseudo-inverse](concepts/geometric-factors.md#the-pseudo-inverse).

Pull-back
: The inverse of the map: given a physical point $x$ and a cell, the reference point $\xi$ with
$X(\xi) = x$, or the closest one on a curve or surface.
See [The pull-back](concepts/pull-back.md).

PVTU
: VTK's parallel unstructured-grid format: a small header that lists one VTU piece per rank.
See [PVTU output](tutorials/parallel-io.md#pvtu-output).

`QuadField`
: The values of one geometric factor on one cell block at the points of a rule, on the device,
in tiles and chunks.
See [BlockFields](tutorials/kernels.md#blockfields).

Quadrature rule
: Points $\xi_q$ and weights $w_q$ on a reference cell that replace an integral by a weighted sum.
A `QuadRule` holds one point set per cell type.
See [Quadrature rules](concepts/quadrature.md#quadrature-rules).

Rank
: One process of an MPI program. Each rank holds its part of a distributed mesh on its own device.
See [One GPU per MPI rank](devices.md#one-gpu-per-mpi-rank).

Rank-0 partitioner
: A partitioner that gathers the whole dual graph on rank 0 and partitions it there: `Metis` and
`Scotch`. It suits meshes of up to about $10^7$ cells.
See [Partitioners](features.md#partitioners).

Reference cell
: The fixed cell, in the unit box $[0, 1]^t$, of which every cell of its type is the image under a
map.
See [Reference cells](concepts/reference-cells.md#reference-cells).

Reference coordinates
: The coordinates $\xi = (\xi_0, \dots, \xi_{t-1})$ of a point of the reference cell.
See [Reference cells](concepts/reference-cells.md#reference-cells).

Reference normal
: The fixed outward unit normal $\hat n$ of a reference cell on one of its facets, returned by
`reference_normal(q)` of a facet rule.
See [Facet normals](concepts/geometric-factors.md#facet-normals).

Reference-to-physical map
: The map $X$ from the reference cell to a cell of the mesh, $X(\xi) = \sum_k x_k \phi_k(\xi)$,
built from the cell's nodes and the Lagrange basis.
See [The reference-to-physical map](concepts/reference-cells.md#the-reference-to-physical-map).

Region
: A set of cells, the volume part of a physical group, selected by name.
See [Regions](tutorials/tags-submeshes.md#regions).

Reverse Cuthill–McKee
: A cell ordering that numbers the cells breadth-first through the facet graph, from a cell with
the fewest neighbours, and reverses the order, so that neighbouring cells get close indices.
See [Cell ordering](tutorials/distributed.md#cell-ordering).

Reverse exchange, `ReverseOp`
: The halo exchange from ghosts to owners: each owner combines its value with those of its ghost
copies, by the operation `ReverseOp` (`Add`, `Min`, `Max` or `Insert`).
See [Reverse](tutorials/halo-exchange.md#reverse).

Round-robin
: A partition that sends graph vertex $i$ to rank $i \bmod p$. The tutorial's own partitioner
`RoundRobin` uses it to show the `Partitioner` trait.
See [Partitioners](tutorials/distributed.md#partitioners).

Runtime, runtime feature
: A CubeCL backend, such as the CPU runtime, CUDA or wgpu, and the feature of the `cubecl`
dependency that links it into your program.
See [Runtime features of CubeCL](getting-started/installation.md#runtime-features-of-cubecl).

Scatterer
: The object that runs the halo exchanges of one index map on device arrays with a fixed block
size (`Scatterer`).
See [Scatterers](tutorials/halo-exchange.md#scatterers).

Seed
: The number that an external partitioner takes to make its partition reproducible: the same
input, seed and number of ranks give the same partition.
See [Partitioners](tutorials/distributed.md#partitioners).

`Serial`
: The communicator of one process. A serial mesh uses it, and its index maps have no ghosts.
See [Ownership and ghosts](concepts/meshes.md#ownership-and-ghosts).

Serendipity elements
: Incomplete quadratic elements of Gmsh, without face and interior nodes. The reader completes
them to Lagrange cells of degree 2.
See [Gmsh](elements/file-formats.md#gmsh).

Sub-entity
: An edge or face of a reference cell, with a local index and a list of local vertices.
See [Sub-entities](concepts/reference-cells.md#sub-entities).

Submesh
: A mesh made from selected cells, facets or edges of a parent mesh, together with an entity map
back to the parent.
See [Tags and submeshes](tutorials/tags-submeshes.md).

Tabulation
: The table of basis function values $\phi_k(\xi_q)$, or of their derivatives, at the points of a
rule, computed once on the host and kept on the device.
See [Tabulation](concepts/quadrature.md#tabulation).

Tag
: A value attached to an entity on the device, such as the physical group of a Gmsh file
(`MeshTags`); the `TagRegistry` holds a mesh's tags and their names.
See [Physical groups](tutorials/tags-submeshes.md#physical-groups).

Tensor order
: The vertex and node order of quadrilaterals and hexahedra, also called lexicographic order: the
first coordinate changes fastest. The boundary of the reference quadrilateral runs through the
vertices 0, 1, 3, 2.
See [Reference cells](concepts/reference-cells.md#reference-cells).

Tile
: A group of `tile()` consecutive cells whose values a field at quadrature points interleaves, so
that neighbouring cells sit next to each other in memory.
See [BlockFields](tutorials/kernels.md#blockfields).

Topological dimension
: The dimension $t$ of the cells of a mesh, 1, 2 or 3, the type parameter `TD` of a mesh.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Transport
: How a halo exchange moves values between ranks (`Transport`). Every scatterer uses
`HostStaged`: values are packed on the device, copied to the host, sent by MPI and uploaded again.
See [Scatterers](tutorials/halo-exchange.md#scatterers).

Vector size, lane
: The number of values that one vector instruction processes together, set per kernel family and
device by built-in policies. Each lane of a vector is one cell or one element.
See [Performance settings](devices.md#performance-settings).

Vertex
: A corner of a cell, part of the topology. Vertices differ from nodes: a curved cell has nodes
that are not vertices.
See [Cells and entities](concepts/meshes.md#cells-and-entities).

Vertex weights
: The estimated work of each cell, from its type and degree, that the partitioners balance across
the ranks.
See [Partitioners](tutorials/distributed.md#partitioners).

VTKHDF
: VTK's format in an HDF5 file. tetraloma writes one file with one partition per rank, each with
the same cells, points and arrays as the rank's PVTU piece.
See [VTKHDF output](tutorials/parallel-io.md#vtkhdf-output).

`vtkGhostType`
: The cell array that marks ghost cells in PVTU pieces and VTKHDF partitions: 0 for an owned cell,
1 for a ghost.
See [PVTU output](tutorials/parallel-io.md#pvtu-output).

`with_mesh!`
: The macro that runs one expression with the typed mesh inside an `AnyMesh`, once per variant.
See [AnyMesh](concepts/typed-dimensions.md#anymesh) and
[with_mesh!](tutorials/kernels.md#with_mesh).
