//! The integration tests of gridcl, one module per file (Cargo.toml, [[test]]).

#[macro_use]
mod common;
mod any_mesh;
mod build_on_threads;
mod builder;
mod compile;
mod determinism;
mod factor_invariance;
mod gmsh;
mod index_map_threads;
mod io_threads;
mod kernel_invariance;
mod manifold_factors;
mod mixed_surface;
mod orient;
mod orient_threads;
mod parallel_threads;
mod partition_threads;
mod permutation_info;
mod precision;
mod pull_back;
mod quad_field_chunks;
mod scatter_threads;
mod sphere_area;
mod submesh;
mod submesh_threads;
mod tags_threads;
mod topology_counts;
mod unit_volume;
mod vtk;
