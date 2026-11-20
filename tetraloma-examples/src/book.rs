//! The book's Rust blocks as doctests (ADR-0067, rule 5): one module per chapter of
//! docs/user/src, with the chapter as its documentation, so that `cargo test --doc` runs every
//! `rust` block of the book. The blocks open their device as a user does, through
//! `cubecl::Device`, so they run on the CPU runtime. Compiled only for doctests.
//!
//! `cargo test -p tetraloma-examples --doc concepts` runs one part of the book, and
//! `--doc book::tutorials::kernels` one chapter. scripts/check-user-docs.sh (check g) fails if a
//! chapter of SUMMARY.md is missing here.

#[doc = include_str!("../../docs/user/src/introduction.md")]
pub mod introduction {}

#[doc = include_str!("../../docs/user/src/devices.md")]
pub mod devices {}

#[doc = include_str!("../../docs/user/src/features.md")]
pub mod features {}

#[doc = include_str!("../../docs/user/src/troubleshooting.md")]
pub mod troubleshooting {}

#[doc = include_str!("../../docs/user/src/glossary.md")]
pub mod glossary {}

/// The chapters in docs/user/src/getting-started/.
pub mod getting_started {
    #[doc = include_str!("../../docs/user/src/getting-started/installation.md")]
    pub mod installation {}

    #[doc = include_str!("../../docs/user/src/getting-started/first-program.md")]
    pub mod first_program {}
}

/// The chapters in docs/user/src/concepts/.
pub mod concepts {
    #[doc = include_str!("../../docs/user/src/concepts/meshes.md")]
    pub mod meshes {}

    #[doc = include_str!("../../docs/user/src/concepts/reference-cells.md")]
    pub mod reference_cells {}

    #[doc = include_str!("../../docs/user/src/concepts/geometry.md")]
    pub mod geometry {}

    #[doc = include_str!("../../docs/user/src/concepts/geometric-factors.md")]
    pub mod geometric_factors {}

    #[doc = include_str!("../../docs/user/src/concepts/orientation.md")]
    pub mod orientation {}

    #[doc = include_str!("../../docs/user/src/concepts/quadrature.md")]
    pub mod quadrature {}

    #[doc = include_str!("../../docs/user/src/concepts/pull-back.md")]
    pub mod pull_back {}

    #[doc = include_str!("../../docs/user/src/concepts/typed-dimensions.md")]
    pub mod typed_dimensions {}

    #[doc = include_str!("../../docs/user/src/concepts/precision.md")]
    pub mod precision {}
}

/// The chapters in docs/user/src/elements/.
pub mod elements {
    #[doc = include_str!("../../docs/user/src/elements/reference-cells.md")]
    pub mod reference_cells {}

    #[doc = include_str!("../../docs/user/src/elements/lagrange.md")]
    pub mod lagrange {}

    #[doc = include_str!("../../docs/user/src/elements/orientation-bits.md")]
    pub mod orientation_bits {}

    #[doc = include_str!("../../docs/user/src/elements/file-formats.md")]
    pub mod file_formats {}

    #[doc = include_str!("../../docs/user/src/elements/basix-dolfinx.md")]
    pub mod basix_dolfinx {}
}

/// The chapters in docs/user/src/tutorials/.
pub mod tutorials {
    #[doc = include_str!("../../docs/user/src/tutorials/build-mesh.md")]
    pub mod build_mesh {}

    #[doc = include_str!("../../docs/user/src/tutorials/read-write.md")]
    pub mod read_write {}

    #[doc = include_str!("../../docs/user/src/tutorials/iterate.md")]
    pub mod iterate {}

    #[doc = include_str!("../../docs/user/src/tutorials/geometric-data.md")]
    pub mod geometric_data {}

    #[doc = include_str!("../../docs/user/src/tutorials/quadrature.md")]
    pub mod quadrature {}

    #[doc = include_str!("../../docs/user/src/tutorials/pull-back.md")]
    pub mod pull_back {}

    #[doc = include_str!("../../docs/user/src/tutorials/tags-submeshes.md")]
    pub mod tags_submeshes {}

    #[doc = include_str!("../../docs/user/src/tutorials/distributed.md")]
    pub mod distributed {}

    #[doc = include_str!("../../docs/user/src/tutorials/halo-exchange.md")]
    pub mod halo_exchange {}

    #[doc = include_str!("../../docs/user/src/tutorials/parallel-io.md")]
    pub mod parallel_io {}

    #[doc = include_str!("../../docs/user/src/tutorials/kernels.md")]
    pub mod kernels {}
}
