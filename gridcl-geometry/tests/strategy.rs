//! Strategy dispatch (M9h T4; ADR-0042, item 3): every launcher of a degree-dependent family
//! asks the degree policy, `record_strategies` shows what ran, and a forced strategy that does
//! not run yet is an error rather than a silent `direct`. `loops` runs for every family (T6a,
//! T6b); `sumfact` for the factors on tensor rules (T7a).

use gridcl::prelude::*;
use gridcl::{FacetQuadRule, PullBackOptions};
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
            Strategy::SumFactorised,
            "sumfact",
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

/// Sum-factorised X and J of perturbed hexahedra at p = 3 against `loops`, relative to the
/// largest entry, within `tol`.
fn sumfact_on_tensor_rules<F: Real>(client: &cubecl::prelude::Client, tol: f64) {
    use StrategyFamily::*;
    for family in [NodeFamily::Equispaced, NodeFamily::Gll] {
        let mesh = gridcl_testing::cube_box(Cell3::Hexahedron, 2, 3)
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
                    "{what} {family:?}: {g} vs {w}"
                );
            }
        }
    }
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
                with_strategy(FacetNormals, sumfact, || {
                    // The equispaced lattice and (until T7b) the facet rule record no factors.
                    gridcl_geometry::equispaced_nodes(&mesh).unwrap();
                    mesh.facet_normals(&FacetQuadRule::for_grid(&mesh, 2).unwrap())
                        .unwrap();
                })
            })
        })
    });
    assert_eq!(
        ran,
        [
            (Points, Strategy::Loops),
            (EVector, Strategy::Loops),
            (FacetNormals, Strategy::Loops)
        ]
    );
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
