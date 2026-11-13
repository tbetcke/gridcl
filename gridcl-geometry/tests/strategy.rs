//! Strategy dispatch (M9h T4; ADR-0042, item 3): every launcher of a degree-dependent family
//! asks the degree policy, `record_strategies` shows what ran, and a forced strategy that does
//! not run yet is an error rather than a silent `direct`. `loops` runs for every family (T6a,
//! T6b); `sumfact` for the factors on tensor rules (T7a) and facet rules (T7b), and on prisms
//! (T7c), and for the pull-back on quadrilaterals, hexahedra and prisms (T8).

use gridcl::prelude::*;
use gridcl::{FacetQuadRule, FieldShape, PullBackOptions};
use gridcl_geometry::testing::{
    Strategy, StrategyFamily, StrategyOverride, record_strategies, with_strategy,
};

/// Every family on `mesh` with the Gauss rule of degree 2: E-vector, the factors, `normals`
/// (the facet normals of codimension 0 or the normals of codimension 1) and the pull-back.
fn every_family<TD, GD>(
    mesh: &Mesh<f32, TD, GD>,
    normals: impl Fn(&Mesh<f32, TD, GD>, &QuadRule<f32, TD>) -> Result<()>,
) -> Result<()>
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let rule = QuadRule::for_grid(mesh, 2)?;
    let x = mesh.physical_points(&rule)?;
    mesh.jacobians(&rule)?;
    mesh.detj(&rule)?;
    normals(mesh, &rule)?;
    mesh.pull_back(&x, &PullBackOptions::default())?;
    Ok(())
}

/// The families that ran, each once, in order of first launch.
fn families(ran: &[(StrategyFamily, Strategy)]) -> Vec<StrategyFamily> {
    let mut out = Vec::new();
    for (f, _) in ran {
        if !out.contains(f) {
            out.push(*f);
        }
    }
    out
}

#[test]
fn record_strategies_shows_direct() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let sphere = gridcl_testing::sphere(0, 2)
        .build::<f32, D2, D3>(&client)
        .unwrap();
    let ((), ran) = record_strategies(|| {
        with_every(StrategyOverride::Policy, || {
            every_family(&square, |m, _| {
                m.facet_normals(&FacetQuadRule::for_grid(m, 2)?).map(drop)
            })
            .unwrap();
            every_family(&sphere, |m, rule| m.normals(rule).map(drop)).unwrap();
        })
    });
    // The policy is `direct` everywhere until T11b; the E-vector is built at the first factor.
    assert!(ran.iter().all(|(_, s)| *s == Strategy::Direct), "{ran:?}");
    assert_eq!(
        families(&ran),
        [Points, EVector, Jacobian, FacetNormals, PullBack, Normals]
    );
    // The E-vector is cached: one build per mesh.
    assert_eq!(ran.iter().filter(|(f, _)| *f == EVector).count(), 2);
}

#[test]
fn forced_direct_matches_the_policy_bitwise() {
    let client = gridcl_testing::test_client();
    let run = |s| {
        let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 2)
            .perturbed(0.05)
            .build::<f32, D3, D3>(&client)
            .unwrap();
        let rule = QuadRule::for_grid(&mesh, 4).unwrap();
        with_every(s, || {
            let j = mesh.jacobians(&rule).unwrap();
            j.blocks[0].to_host_cell_major().unwrap()
        })
    };
    let policy = run(StrategyOverride::Policy);
    let direct = run(StrategyOverride::Fixed(Strategy::Direct));
    let bits = |v: &Vec<[[f32; 3]; 3]>| -> Vec<u32> {
        v.iter().flatten().flatten().map(|x| x.to_bits()).collect()
    };
    assert_eq!(bits(&policy), bits(&direct));
}

#[test]
fn strategies_that_do_not_run_yet_are_errors() {
    let client = gridcl_testing::test_client();
    let triangle = gridcl_testing::square_box(Cell2::Triangle, 2, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let quadrilateral = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    for (mesh, family, s, want) in [
        (
            &quadrilateral,
            StrategyFamily::PullBack,
            Strategy::Shared,
            "shared",
        ),
        (
            &triangle,
            StrategyFamily::Points,
            Strategy::Shared,
            "shared",
        ),
        (
            &triangle,
            StrategyFamily::Points,
            Strategy::Cooperative,
            "cooperative",
        ),
    ] {
        let rule = QuadRule::for_grid(mesh, 2).unwrap();
        let x = mesh.physical_points(&rule).unwrap();
        let (got, ran) = record_strategies(|| {
            with_strategy(family, StrategyOverride::Fixed(s), || match family {
                StrategyFamily::PullBack => {
                    mesh.pull_back(&x, &PullBackOptions::default()).map(drop)
                }
                _ => mesh.physical_points(&rule).map(drop),
            })
        });
        match got {
            Err(GridError::NotYetImplemented(task)) => {
                assert!(task.contains(&format!("strategy {want} ")), "{s:?}: {task}");
            }
            other => panic!("{s:?}: {other:?}"),
        }
        assert!(ran.is_empty(), "{s:?} ran {ran:?}");
    }
}

#[test]
fn forced_loops_runs_loops() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let triangle = gridcl_testing::square_box(Cell2::Triangle, 2, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    // Sum factorisation does not apply to a triangle: its fallback `loops` runs.
    for s in [Strategy::Loops, Strategy::SumFactorised] {
        let mesh = triangle.try_clone().unwrap();
        let rule = QuadRule::for_grid(&mesh, 2).unwrap();
        let ((), ran) = record_strategies(|| {
            with_strategy(EVector, StrategyOverride::Fixed(s), || {
                with_strategy(Points, StrategyOverride::Fixed(s), || {
                    with_strategy(Jacobian, StrategyOverride::Fixed(s), || {
                        with_strategy(PullBack, StrategyOverride::Fixed(s), || {
                            let x = mesh.physical_points(&rule).unwrap();
                            mesh.jacobians(&rule).unwrap();
                            mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
                        })
                    })
                })
            })
        });
        assert_eq!(
            ran,
            [
                (Points, Strategy::Loops),
                (EVector, Strategy::Loops),
                (Jacobian, Strategy::Loops),
                (PullBack, Strategy::Loops)
            ],
            "{s:?}"
        );
    }
}

/// X and J on `mesh` with every factor family and the E-vector forced to `s`, cell-major.
fn points_and_jacobians<F: Real>(
    mesh: &Mesh<F, D3, D3>,
    rule: &QuadRule<F, D3>,
    s: Strategy,
) -> (Vec<f64>, Vec<f64>, Vec<(StrategyFamily, Strategy)>) {
    use StrategyFamily::*;
    let mesh = mesh.try_clone().unwrap();
    let s = StrategyOverride::Fixed(s);
    let ((x, j), ran) = record_strategies(|| {
        with_strategy(EVector, s, || {
            with_strategy(Points, s, || {
                with_strategy(Jacobian, s, || {
                    let x = mesh.physical_points(rule).unwrap();
                    let j = mesh.jacobians(rule).unwrap();
                    (
                        x.blocks[0].to_host_cell_major().unwrap(),
                        j.blocks[0].to_host_cell_major().unwrap(),
                    )
                })
            })
        })
    });
    (
        x.into_iter().flatten().map(F::widen).collect(),
        j.into_iter().flatten().flatten().map(F::widen).collect(),
        ran,
    )
}

#[test]
fn forced_sumfact_runs_on_tensor_rules() {
    let client = gridcl_testing::test_client();
    sumfact_on_tensor_rules::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        sumfact_on_tensor_rules::<f64>(&client, 1e-13);
    }
}

/// Sum-factorised X and J of perturbed hexahedra (both node families) and prisms at p = 3
/// against `loops`, relative to the largest entry, within `tol`.
fn sumfact_on_tensor_rules<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    for (cell, family) in [
        (Cell3::Hexahedron, NodeFamily::Equispaced),
        (Cell3::Hexahedron, NodeFamily::Gll),
        (Cell3::Prism, NodeFamily::Equispaced),
    ] {
        let mesh = gridcl_testing::cube_box(cell, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let rule = QuadRule::for_grid(&mesh, 6).unwrap();
        let (x, j, ran) = points_and_jacobians(&mesh, &rule, Strategy::SumFactorised);
        assert_eq!(
            ran,
            [
                (Points, Strategy::SumFactorised),
                // The E-vector is a gather: sum factorisation falls back to `loops`.
                (EVector, Strategy::Loops),
                (Jacobian, Strategy::SumFactorised)
            ]
        );
        let (x_loops, j_loops, _) = points_and_jacobians(&mesh, &rule, Strategy::Loops);
        for (what, got, want) in [("X", &x, &x_loops), ("J", &j, &j_loops)] {
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            for (g, w) in got.iter().zip(want.iter()) {
                assert!(
                    (g - w).abs() <= tol * scale,
                    "{what} {cell:?} {family:?}: {g} vs {w}"
                );
            }
        }
    }
}

/// The values of every block of `field`, concatenated.
fn blocks_f64<F: Real, S: FieldShape>(field: &BlockFields<F, S>) -> Vec<f64> {
    field
        .blocks
        .iter()
        .flat_map(|b| gridcl_testing::reference::field_f64(b).unwrap())
        .collect()
}

/// Named quantities, each flattened.
type Quantities = Vec<(&'static str, Vec<f64>)>;

/// `f` on a fresh copy of `mesh` with every family forced to `s`, and what ran.
fn forced<F, TD, GD>(
    mesh: &Mesh<F, TD, GD>,
    s: Strategy,
    f: impl FnOnce(&Mesh<F, TD, GD>) -> Quantities,
) -> (Quantities, Vec<(StrategyFamily, Strategy)>)
where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let mesh = mesh.try_clone().unwrap();
    record_strategies(|| with_every(StrategyOverride::Fixed(s), || f(&mesh)))
}

/// The quantities of `f` under `sumfact` against `loops`, relative to each quantity's largest
/// entry, within `tol`; `sumfact` runs for each of `families`, `loops` for everything else.
fn sumfact_against_loops<F, TD, GD>(
    what: &str,
    mesh: &Mesh<F, TD, GD>,
    families: &[StrategyFamily],
    tol: f64,
    f: impl Fn(&Mesh<F, TD, GD>) -> Quantities,
) where
    F: Real,
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let (got, ran) = forced(mesh, Strategy::SumFactorised, &f);
    for family in families {
        assert!(
            ran.contains(&(*family, Strategy::SumFactorised)),
            "{what}: {family:?} not sum-factorised: {ran:?}"
        );
    }
    assert!(
        ran.iter()
            .all(|(fam, s)| *s == Strategy::Loops || families.contains(fam)),
        "{what}: {ran:?}"
    );
    let (want, _) = forced(mesh, Strategy::Loops, &f);
    for ((name, got), (_, want)) in got.iter().zip(&want) {
        assert_eq!(got.len(), want.len());
        let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for (g, w) in got.iter().zip(want) {
            assert!((g - w).abs() <= tol * scale, "{name} of {what}: {g} vs {w}");
        }
    }
}

#[test]
fn forced_sumfact_runs_for_derived_factors_and_facet_rules() {
    let client = gridcl_testing::test_client();
    derived_factors_and_facets::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        derived_factors_and_facets::<f64>(&client, 1e-13);
    }
}

/// det J, K and the facet normals of perturbed quadrilaterals, hexahedra and prisms, and det J,
/// K and the normals of the mixed surface (its quadrilateral block), at p = 3, under `sumfact`
/// against `loops`.
fn derived_factors_and_facets<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    fn codim0<F: Real, TD: TopoDim + Codim0<TD>>(mesh: &Mesh<F, TD, TD>) -> Quantities {
        let rule = QuadRule::for_grid(mesh, 6).unwrap();
        let facets = FacetQuadRule::for_grid(mesh, 6).unwrap();
        vec![
            ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
            (
                "K",
                blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
            ),
            (
                "facet normals",
                blocks_f64(&mesh.facet_normals(&facets).unwrap()),
            ),
        ]
    }
    for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
        let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D2, D2>(client)
            .unwrap();
        let cube = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let factors = [Jacobian, FacetNormals];
        sumfact_against_loops(
            &format!("square {family:?}"),
            &square,
            &factors,
            tol,
            codim0,
        );
        sumfact_against_loops(&format!("cube {family:?}"), &cube, &factors, tol, codim0);
    }
    let prisms = gridcl_testing::cube_box(Cell3::Prism, 2, 3)
        .perturbed(0.05)
        .build::<F, D3, D3>(client)
        .unwrap();
    sumfact_against_loops("prisms", &prisms, &[Jacobian, FacetNormals], tol, codim0);
    let surface = gridcl_testing::mixed_surface(1, 3)
        .build::<F, D2, D3>(client)
        .unwrap();
    sumfact_against_loops(
        "mixed surface",
        &surface,
        &[Jacobian, Normals],
        tol,
        |mesh| {
            let rule = QuadRule::for_grid(mesh, 6).unwrap();
            vec![
                ("det J", blocks_f64(&mesh.detj(&rule).unwrap())),
                (
                    "K",
                    blocks_f64(&mesh.pseudo_inverse_jacobians(&rule).unwrap()),
                ),
                ("normals", blocks_f64(&mesh.normals(&rule).unwrap())),
            ]
        },
    );
}

#[test]
fn forced_sumfact_pull_back_agrees_with_loops() {
    let client = gridcl_testing::test_client();
    sumfact_pull_backs::<f32>(&client, 1e-5);
    if gridcl_testing::supports::<f64>(&client) {
        sumfact_pull_backs::<f64>(&client, 1e-12);
    }
}

/// ξ and the last step of perturbed quadrilaterals, hexahedra (both node families), prisms and
/// the mixed surface (its quadrilateral block, codimension 1) at p = 3, at the Gauss points of
/// degree 6, under `sumfact` against `loops`; every point converges under both.
fn sumfact_pull_backs<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    fn pull_back<F: Real, TD: TopoDim + EmbeddedIn<GD>, GD: Dim>(
        mesh: &Mesh<F, TD, GD>,
    ) -> Quantities {
        let x = mesh
            .physical_points(&QuadRule::for_grid(mesh, 6).unwrap())
            .unwrap();
        let pb = mesh.pull_back(&x, &PullBackOptions::default()).unwrap();
        let steps = blocks_f64(&pb.last_step);
        let converged = if size_of::<F>() == 8 { 1e-13 } else { 1e-5 };
        assert!(steps.iter().all(|&s| s <= converged), "{steps:?}");
        vec![("ξ", blocks_f64(&pb.reference))]
    }
    for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
        let square = gridcl_testing::square_box(Cell2::Quadrilateral, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D2, D2>(client)
            .unwrap();
        let cube = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
            .perturbed(0.05)
            .with_node_family(family)
            .build::<F, D3, D3>(client)
            .unwrap();
        let what = format!("square {family:?}");
        sumfact_against_loops(&what, &square, &[Points, PullBack], tol, pull_back);
        let what = format!("cube {family:?}");
        sumfact_against_loops(&what, &cube, &[Points, PullBack], tol, pull_back);
    }
    let prisms = gridcl_testing::cube_box(Cell3::Prism, 2, 3)
        .perturbed(0.05)
        .build::<F, D3, D3>(client)
        .unwrap();
    sumfact_against_loops("prisms", &prisms, &[Points, PullBack], tol, pull_back);
    let surface = gridcl_testing::mixed_surface(1, 3)
        .build::<F, D2, D3>(client)
        .unwrap();
    sumfact_against_loops(
        "mixed surface",
        &surface,
        &[Points, PullBack],
        tol,
        pull_back,
    );
}

#[test]
fn sumfact_falls_back_to_loops_without_tensor_points() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 1, 2)
        .with_node_family(NodeFamily::Gll)
        .build::<f32, D3, D3>(&client)
        .unwrap();
    let sumfact = StrategyOverride::Fixed(Strategy::SumFactorised);
    let ((), ran) = record_strategies(|| {
        with_strategy(EVector, sumfact, || {
            with_strategy(Points, sumfact, || {
                // The equispaced lattice records no factors.
                gridcl_geometry::equispaced_nodes(&mesh).unwrap();
            })
        })
    });
    assert_eq!(ran, [(Points, Strategy::Loops), (EVector, Strategy::Loops)]);
}

#[test]
fn overrides_nest_per_family() {
    use StrategyFamily::*;
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 2)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 2).unwrap();
    let cooperative = StrategyOverride::Fixed(Strategy::Cooperative);
    let direct = StrategyOverride::Fixed(Strategy::Direct);
    // Another family's override does not reach the points.
    with_strategy(Jacobian, cooperative, || {
        mesh.physical_points(&rule).unwrap();
        assert!(mesh.jacobians(&rule).is_err());
        // The innermost override wins, and the outer one is back afterwards.
        with_strategy(Jacobian, direct, || mesh.jacobians(&rule).unwrap());
        assert!(mesh.jacobians(&rule).is_err());
    });
    mesh.jacobians(&rule).unwrap();
}

#[test]
fn records_nest_and_survive_panics() {
    let client = gridcl_testing::test_client();
    let mesh = gridcl_testing::square_box(Cell2::Triangle, 1, 1)
        .build::<f32, D2, D2>(&client)
        .unwrap();
    let rule = QuadRule::for_grid(&mesh, 2).unwrap();
    let ((inner, panicked), outer) = record_strategies(|| {
        let ((), inner) = record_strategies(|| {
            mesh.physical_points(&rule).unwrap();
        });
        let panicked = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            with_strategy(
                StrategyFamily::Jacobian,
                StrategyOverride::Fixed(Strategy::Direct),
                || {
                    let _ = record_strategies(|| {
                        mesh.jacobians(&rule).unwrap();
                        panic!("inside");
                    });
                },
            )
        }))
        .is_err();
        (inner, panicked)
    });
    assert!(panicked);
    assert_eq!(
        families(&inner),
        [StrategyFamily::Points, StrategyFamily::EVector]
    );
    // The outer record sees the inner launches, also those of the scope that panicked.
    assert_eq!(
        families(&outer),
        [
            StrategyFamily::Points,
            StrategyFamily::EVector,
            StrategyFamily::Jacobian
        ]
    );
    // The hook of the scope that panicked is gone.
    with_strategy(
        StrategyFamily::Jacobian,
        StrategyOverride::Fixed(Strategy::Cooperative),
        || assert!(mesh.jacobians(&rule).is_err()),
    );
    mesh.jacobians(&rule).unwrap();
}

/// Runs `f` with every family overridden by `s`.
fn with_every<R>(s: StrategyOverride, f: impl FnOnce() -> R) -> R {
    StrategyFamily::ALL
        .into_iter()
        .rev()
        .fold(Box::new(f) as Box<dyn FnOnce() -> R>, |f, family| {
            Box::new(move || with_strategy(family, s, f))
        })()
}
