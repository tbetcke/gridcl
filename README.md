# tetraloma

tetraloma is a GPU-resident mesh library in Rust, built on [CubeCL](https://github.com/tracel-ai/cubecl)
0.11. It is meant as the foundation for strongly typed finite and boundary element crates.

- Topological and geometric dimensions are part of the type, `Mesh<F, TD, GD, C>`; cell types and
  geometry degree are runtime values stored per block.
- Intervals, triangles, quadrilaterals, tetrahedra, hexahedra, prisms and pyramids, hybrid meshes, and
  geometry of arbitrary polynomial degree.
- Mesh arrays live on the device: CUDA, HIP, Vulkan, Metal and the CubeCL CPU runtime, in f32 and f64.
- MPI-distributed meshes with ownership, ghost layers and parallel partitioning (PT-Scotch by default).
- Gmsh MSH 4.1 import and export; VTK export (VTU, PVTU, VTKHDF) with Lagrange cells.

## Status

Under active development and not yet released: the public API may still change. The crate name
`tetraloma` is reserved on crates.io; the `tetraloma-*` crates are published with the first release.

## Documentation

- [docs/design.md](docs/design.md): the design, the source of truth for the implementation.
- [docs/adr/](docs/adr/): architecture decision records.
- [docs/dev-environment.md](docs/dev-environment.md): toolchain, test targets and how to run the tests.
- [IMPLEMENTATION.md](IMPLEMENTATION.md): how the milestones are planned, implemented and reviewed.

Build and test on the CubeCL CPU runtime with the pinned toolchain (`rust-toolchain.toml`):

```sh
cargo test --workspace
```

## Licence

Licensed under either of the [Apache License, Version 2.0](LICENSE-APACHE) or the
[MIT licence](LICENSE-MIT), at your option.
