//! Test-only stand-in for gridcl's `Mesh` (docs/design.md, Listings B3 and B4).
//!
//! Names, generic parameters and impl bounds follow the listings; return types are simplified.
//! Nothing here is executed: the cases only type-check calls. The `.stderr` baselines quote the
//! impl headers `impl<F, TD, GD, C> Mesh<F, TD, GD, C>`, the bound lines
//! `TD: TopoDim + Codim1<GD>,` and `TD::Lower: TopoDim + EmbeddedIn<GD>,` and the line
//! `pub fn new(cell: TD::Cell) -> Self {`, and `pub struct D0;` and `pub struct D3;` of
//! src/dim.rs (all without line numbers): editing any of them means regenerating and reviewing
//! the baselines.
#![allow(dead_code, unused)]

use core::marker::PhantomData;

use gridcl_dims::{Codim0, Codim1, D1, D2, D3, Dim, EmbeddedIn, TopoDim};

/// Stand-in for gridcl-device's `Real`.
pub trait Real: Copy + 'static {}
impl Real for f32 {}
impl Real for f64 {}

/// Stand-in for gridcl-parallel's `Comm` and `Serial`.
pub trait Comm {}
#[derive(Clone, Copy, Debug, Default)]
pub struct Serial;
impl Comm for Serial {}

/// Stand-ins for gridcl-geometry's `QuadRule` and gridcl's `EntityMap` and error type.
pub struct QuadRule<F: Real, TD: TopoDim>(PhantomData<(F, TD)>);
pub struct EntityMap;
#[derive(Debug)]
pub struct StubError;

pub struct Mesh<F: Real, TD: TopoDim, GD: Dim, C: Comm = Serial>(PhantomData<(F, TD, GD, C)>);

pub type CurveMesh<F, C = Serial> = Mesh<F, D1, D3, C>;
pub type PlanarMesh<F, C = Serial> = Mesh<F, D2, D2, C>;
pub type SurfaceMesh<F, C = Serial> = Mesh<F, D2, D3, C>;
pub type VolumeMesh<F, C = Serial> = Mesh<F, D3, D3, C>;

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
    C: Comm,
{
    pub fn tdim(&self) -> usize {
        unimplemented!()
    }
    pub fn gdim(&self) -> usize {
        unimplemented!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + Codim0<GD>,
    GD: Dim,
    C: Comm,
{
    pub fn inverse_jacobians(&self, rule: &QuadRule<F, TD>) -> Result<(), StubError> {
        unimplemented!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    TD: TopoDim + Codim1<GD>,
    GD: Dim,
    C: Comm,
{
    pub fn normals(&self, rule: &QuadRule<F, TD>) -> Result<(), StubError> {
        unimplemented!()
    }
}

impl<F, TD, GD, C> Mesh<F, TD, GD, C>
where
    F: Real,
    GD: Dim,
    C: Comm,
    TD: TopoDim + EmbeddedIn<GD>,
    TD::Lower: TopoDim + EmbeddedIn<GD>,
{
    pub fn boundary(&self) -> Result<(Mesh<F, TD::Lower, GD, C>, EntityMap), StubError> {
        unimplemented!()
    }
}

pub struct CellBlock<TD: TopoDim> {
    pub cell: TD::Cell,
}

impl<TD: TopoDim> CellBlock<TD> {
    pub fn new(cell: TD::Cell) -> Self {
        unimplemented!()
    }
}
