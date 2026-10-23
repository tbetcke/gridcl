//! The erased layer (Listing B6, M3 T16): `TryFrom<AnyMesh>` succeeds only for the matching
//! variant and gives `DimensionMismatch` otherwise; `with_mesh!` runs over every variant.

use cubecl::prelude::Client;
use gridcl::prelude::*;

fn client() -> Client {
    gridcl_testing::test_client()
}

/// One P1 cell of type `cell`, its vertices the first `tdim + 1` unit vectors (and the origin)
/// of R^gdim, or the unit square or cube for quadrilaterals and hexahedra.
fn one_cell<TD, GD>(client: &Client, cell: TD::Cell) -> Mesh<f32, TD, GD>
where
    TD: TopoDim + EmbeddedIn<GD>,
    GD: Dim,
{
    let td = TD::N;
    let gd = GD::N;
    let num_vertices = match td {
        1 => 2,
        2 if Into::<CellType>::into(cell) == CellType::Quadrilateral => 4,
        2 => 3,
        _ => 4,
    };
    // Vertex v has the bits of v as its first tdim coordinates (reference order of A.1 for
    // intervals, quadrilaterals; the unit simplex for triangles and tetrahedra).
    let mut coords = vec![0.0; num_vertices * gd];
    for v in 0..num_vertices {
        for d in 0..td {
            coords[v * gd + d] = if num_vertices == td + 1 {
                f64::from(u8::from(v == d + 1))
            } else {
                f64::from(u8::from((v >> d) & 1 == 1))
            };
        }
    }
    let nodes: Vec<u32> = (0..num_vertices as u32).collect();
    let element = CoordinateElement::<TD> {
        cell,
        degree: 1,
        family: NodeFamily::Equispaced,
    };
    MeshBuilder::new(client)
        .nodes(&coords, None)
        .cell_block(element, &nodes, None)
        .build()
        .expect("mesh")
}

/// One mesh of each of the six variants, in Listing B6 order.
fn all_variants(client: &Client) -> Vec<AnyMesh<f32>> {
    vec![
        AnyMesh::Curve1(one_cell(client, Cell1::Interval)),
        AnyMesh::Curve2(one_cell(client, Cell1::Interval)),
        AnyMesh::Curve3(one_cell(client, Cell1::Interval)),
        AnyMesh::Planar(one_cell(client, Cell2::Quadrilateral)),
        AnyMesh::Surface(one_cell(client, Cell2::Triangle)),
        AnyMesh::Volume(one_cell(client, Cell3::Tetrahedron)),
    ]
}

const DIMS: [(usize, usize); 6] = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)];

/// `T::try_from(any)` for the variant with dimensions `expected`: Ok exactly when `any` has
/// them, `DimensionMismatch { expected, found }` otherwise.
fn check_try_from<T>(any: AnyMesh<f32>, expected: (usize, usize))
where
    T: TryFrom<AnyMesh<f32>, Error = GridError>,
{
    let found = (any.tdim(), any.gdim());
    match T::try_from(any) {
        Ok(_) => assert_eq!(found, expected),
        Err(GridError::DimensionMismatch {
            expected: e,
            found: f,
        }) => {
            assert_ne!(found, expected);
            assert_eq!((e, f), (expected, found));
        }
        Err(e) => panic!("unexpected error {e:?}"),
    }
}

#[test]
fn try_from_gives_dimension_mismatch() {
    let c = client();
    for target in DIMS {
        for any in all_variants(&c) {
            match target {
                (1, 1) => check_try_from::<Mesh<f32, D1, D1>>(any, target),
                (1, 2) => check_try_from::<Mesh<f32, D1, D2>>(any, target),
                (1, 3) => check_try_from::<CurveMesh<f32>>(any, target),
                (2, 2) => check_try_from::<PlanarMesh<f32>>(any, target),
                (2, 3) => check_try_from::<SurfaceMesh<f32>>(any, target),
                _ => check_try_from::<VolumeMesh<f32>>(any, target),
            }
        }
    }
}

#[test]
fn with_mesh_runs_over_every_variant() {
    let c = client();
    let anys = all_variants(&c);
    let seen: Vec<(usize, usize, usize, usize)> = anys
        .iter()
        .map(|any| {
            with_mesh!(any.as_ref(), m => (
                m.tdim(),
                m.gdim(),
                m.cell_blocks()[0].index_map.num_local(),
                m.node_index_map().num_local(),
            ))
        })
        .collect();
    let expected_nodes = [2, 2, 2, 4, 3, 4];
    for (i, any) in anys.iter().enumerate() {
        let (tdim, gdim, cells, nodes) = seen[i];
        assert_eq!((tdim, gdim), DIMS[i]);
        assert_eq!((any.tdim(), any.gdim()), DIMS[i]);
        assert_eq!(cells, 1);
        assert_eq!(nodes, expected_nodes[i]);
    }
}

#[test]
fn with_mesh_computes_on_the_typed_mesh() {
    // The body runs generic code of each variant's types: the measure of the one cell.
    let c = client();
    let expected = [1.0, 1.0, 1.0, 1.0, 0.5, 1.0 / 6.0];
    for (any, want) in all_variants(&c).iter().zip(expected) {
        let got = with_mesh!(any.as_ref(), m => m.measure(2)).expect("measure");
        assert!((got - want).abs() <= 1e-5 * want, "{got} vs {want}");
    }
}
