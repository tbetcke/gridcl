//! M2 acceptance: "on the CPU runtime, `LaunchPlan` uses one unit per hardware thread and blocks
//! of whole 128-byte lines; with injected device properties, no plan exceeds
//! `max_units_per_cube`, `max_cube_dim` or `max_cube_count` (property test)".

mod common;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;
use gridcl_device::testing::{with_cpu_block_bytes, with_properties, with_vector_size};
use gridcl_device::{
    GridError, KernelKind, LaunchPlan, VECTOR_PAD, VectorPolicy, VectorSizeOverride,
};
use gridcl_testing::test_client;
use proptest::prelude::*;

const KINDS: [KernelKind; 13] = [
    KernelKind::Fill,
    KernelKind::Copy,
    KernelKind::Reduce,
    KernelKind::ScanTile,
    KernelKind::SortDigits,
    KernelKind::Gather,
    KernelKind::Pack,
    KernelKind::TopologyKeys,
    KernelKind::PermutationInfo,
    KernelKind::EVector,
    KernelKind::GeomFactors,
    KernelKind::PullBack,
    KernelKind::Scalar,
];

/// Cube count of a plan as (x, y, z).
fn cube_count(plan: &LaunchPlan) -> (u32, u32, u32) {
    match plan.cube_count {
        CubeCount::Static(x, y, z) => (x, y, z),
        CubeCount::Dynamic(_) => panic!("a plan has a dynamic cube count"),
    }
}

/// Units launched, times items_per_unit: the vectors the launch can cover.
fn covered(plan: &LaunchPlan) -> u128 {
    let (x, y, z) = cube_count(plan);
    [x, y, z, plan.cube_dim.num_elems(), plan.items_per_unit]
        .into_iter()
        .map(u128::from)
        .product()
}

/// The checks every plan must pass under the properties `props`.
fn check_plan(plan: &LaunchPlan, props: &DeviceProperties, items: usize, max_vector: usize) {
    let hw = &props.hardware;
    let (x, y, z) = cube_count(plan);
    let dim = plan.cube_dim;
    assert!(
        x >= 1 && y >= 1 && z >= 1,
        "empty cube count {:?}",
        (x, y, z)
    );
    assert!(
        x <= hw.max_cube_count.0 && y <= hw.max_cube_count.1 && z <= hw.max_cube_count.2,
        "cube count {:?} exceeds max_cube_count {:?}",
        (x, y, z),
        hw.max_cube_count
    );
    assert!(
        dim.x <= hw.max_cube_dim.0 && dim.y <= hw.max_cube_dim.1 && dim.z <= hw.max_cube_dim.2,
        "cube dim {dim:?} exceeds max_cube_dim {:?}",
        hw.max_cube_dim
    );
    assert!(
        dim.num_elems() >= 1 && dim.num_elems() <= hw.max_units_per_cube,
        "cube dim {dim:?} exceeds max_units_per_cube {}",
        hw.max_units_per_cube
    );
    let n = plan.vector_size;
    assert!(
        n.is_power_of_two() && VECTOR_PAD.is_multiple_of(n) && n <= max_vector,
        "vector size {n} with max_vector {max_vector}"
    );
    assert!(plan.items_per_unit >= 1, "items_per_unit is 0");
    let scalars = u128::from(plan.num_vectors) * n as u128;
    let padded = items.max(1).next_multiple_of(VECTOR_PAD) as u128;
    assert!(
        scalars >= items as u128 && scalars <= padded,
        "{} vectors of {n} for {items} items",
        plan.num_vectors
    );
    let covered = covered(plan);
    assert!(
        covered >= u128::from(plan.num_vectors),
        "the launch covers {covered} vectors, fewer than {}",
        plan.num_vectors
    );
    assert!(
        covered <= u128::from(u32::MAX),
        "cube count x cube_dim x items_per_unit = {covered} exceeds u32"
    );
}

#[test]
fn cpu_plan_one_unit_per_thread_whole_lines() {
    let client = test_client();
    let mut cases: Vec<(&str, DeviceProperties)> = Vec::new();
    let props = client.properties().clone();
    if props.hardware.num_cpu_cores.is_some() {
        println!(
            "CPU runtime: max_units_per_cube {}, available_parallelism {:?}",
            props.hardware.max_units_per_cube,
            std::thread::available_parallelism()
        );
        cases.push(("the CPU runtime", props.clone()));
    } else {
        println!(
            "{} is not the CPU runtime: injected CPU properties only",
            client.name()
        );
    }
    // Injected CPU properties, so the rule is also checked on GPU test devices.
    let mut cpu = props;
    cpu.hardware.num_cpu_cores = Some(12);
    cpu.hardware.max_units_per_cube = 12;
    cpu.hardware.max_cube_dim = (12, 12, 12);
    cpu.hardware.max_cube_count = (u32::MAX, u32::MAX, u32::MAX);
    cpu.hardware.load_width = 128;
    cpu.hardware.vector_register_count = Some(32);
    cases.push(("injected CPU properties", cpu));

    for (what, props) in cases {
        let units = props.hardware.max_units_per_cube;
        with_properties(props.clone(), || {
            for kind in KINDS {
                for elem_bytes in [4usize, 8] {
                    for n in [1usize, 2, 4, 8, 16] {
                        for items in [1usize, 1000, 1_000_000, 10_000_003] {
                            let plan = with_vector_size(VectorSizeOverride::Fixed(n), || {
                                LaunchPlan::elementwise(&client, kind, items, elem_bytes, 16, 1)
                            })
                            .unwrap_or_else(|e| panic!("{what}: {kind:?} {items}: {e}"));
                            check_plan(&plan, &props, items, 16);
                            assert_eq!(
                                plan.cube_dim.num_elems(),
                                units,
                                "{what}: {kind:?}: one unit per hardware thread"
                            );
                            let target = VectorPolicy::for_client(&client)
                                .cpu_block_bytes(kind, elem_bytes)
                                .expect("a CPU class has CPU blocks");
                            let block =
                                plan.items_per_unit as usize * plan.vector_size * elem_bytes;
                            assert!(
                                block.is_multiple_of(128)
                                    && (target / 2 < block && block <= target),
                                "{what}: {kind:?}, {elem_bytes}-byte items at vector size {}: \
                                 a block of {block} bytes is not about the policy's {target} \
                                 bytes of whole 128-byte lines",
                                plan.vector_size
                            );
                        }
                    }
                }
            }
        });
    }
}

/// Properties with arbitrary launch limits.
fn injected(
    base: &DeviceProperties,
    cpu: bool,
    units: u32,
    dim: (u32, u32, u32),
    count: (u32, u32, u32),
    load_width: u32,
) -> DeviceProperties {
    let mut props = base.clone();
    let hw = &mut props.hardware;
    hw.num_cpu_cores = cpu.then_some(units);
    hw.max_units_per_cube = units;
    hw.max_cube_dim = dim;
    hw.max_cube_count = count;
    hw.load_width = load_width;
    hw.vector_register_count = cpu.then_some(32);
    props
}

fn limit() -> impl Strategy<Value = u32> {
    prop_oneof![
        1u32..=8,
        1u32..=65_535,
        Just(65_535u32),
        Just(i32::MAX as u32)
    ]
}

fn override_value() -> impl Strategy<Value = Option<VectorSizeOverride>> {
    prop_oneof![
        Just(None),
        Just(Some(VectorSizeOverride::Scalar)),
        Just(Some(VectorSizeOverride::Policy)),
        Just(Some(VectorSizeOverride::Widest)),
        prop::sample::select(vec![1usize, 2, 4, 8, 16])
            .prop_map(|n| Some(VectorSizeOverride::Fixed(n))),
    ]
}

proptest! {
    #![proptest_config(common::prop_config())]

    #[test]
    fn prop_plan_respects_injected_limits(
        cpu in any::<bool>(),
        units in prop_oneof![1u32..=16, 1u32..=1024],
        dim in (1u32..=1024, 1u32..=1024, 1u32..=64),
        count in (limit(), limit(), limit()),
        load_width in prop::sample::select(vec![128u32, 256, 512]),
        items in prop_oneof![0usize..=10_000, 0usize..=100_000_000, 0usize..=(1usize << 34)],
        elem_bytes in prop::sample::select(vec![4usize, 8]),
        max_vector in prop::sample::select(vec![1usize, 2, 4, 8, 16]),
        live in 1usize..=8,
        kind in prop::sample::select(KINDS.to_vec()),
        vector in override_value(),
        block in prop::option::of(prop::sample::select(vec![4usize << 10, 16 << 10, 64 << 10])),
    ) {
        let client = test_client();
        let props = injected(client.properties(), cpu, units, dim, count, load_width);
        let plan = with_properties(props.clone(), || {
            let make = || LaunchPlan::elementwise(&client, kind, items, elem_bytes, max_vector, live);
            let make = || match vector {
                Some(v) => with_vector_size(v, make),
                None => make(),
            };
            match block {
                Some(b) => with_cpu_block_bytes(b, make),
                None => make(),
            }
        });
        if let (Ok(plan), Some(b), true) = (&plan, block, cpu) {
            let bytes = plan.items_per_unit as usize * plan.vector_size * elem_bytes;
            prop_assert!(bytes.is_multiple_of(128) && bytes <= b.max(128), "block {bytes} for {b}");
        }
        match plan {
            Ok(plan) => check_plan(&plan, &props, items, max_vector),
            Err(GridError::TooLarge) => {
                // A plan must exist when even one vector of one scalar per cube fits.
                let padded = items.max(1).next_multiple_of(VECTOR_PAD) as u128;
                let cubes = u128::from(count.0) * u128::from(count.1) * u128::from(count.2);
                prop_assert!(
                    padded > 1u128 << 30 || padded > cubes,
                    "TooLarge for {items} items with max_cube_count {count:?}"
                );
            }
            Err(other) => prop_assert!(false, "unexpected error {other}"),
        }
    }
}
