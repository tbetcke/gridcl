//! M10 bullet 5 (ADR-0058): `DeviceClass` has exactly the variants `Cpu` and `Gpu`, and
//! `vector-policy.toml` holds exactly four sets, `[Cpu]` and `[Gpu]` for 4- and 8-byte scalars.

use tetraloma_device::{DeviceClass, KernelKind, VectorPolicy};

/// Every kind, by its key in `vector-policy.toml`.
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

#[test]
fn device_class_has_exactly_cpu_and_gpu() {
    let client = tetraloma_testing::test_client();
    let class = DeviceClass::for_client(&client);
    // Exhaustive: a third variant does not compile.
    let cpu = match class {
        DeviceClass::Cpu => true,
        DeviceClass::Gpu => false,
    };
    assert_eq!(
        cpu,
        client.properties().hardware.num_cpu_cores.is_some(),
        "Cpu exactly for devices with CPU cores"
    );
    // The CPU runtime is Cpu; Metal, CUDA, HIP and the wgpu backends are Gpu.
    let device = std::env::var(tetraloma_testing::TEST_DEVICE_VAR).unwrap_or_default();
    let want_cpu = matches!(device.as_str(), "" | "cpu");
    assert_eq!(cpu, want_cpu, "GRIDCL_TEST_DEVICE={device:?}");
}

/// The table `key` of `table`.
fn sub<'a>(table: &'a toml::Table, key: &str, what: &str) -> &'a toml::Table {
    table
        .get(key)
        .and_then(|v| v.as_table())
        .unwrap_or_else(|| panic!("{what}: no table {key}"))
}

/// The keys of `table`, sorted.
fn keys(table: &toml::Table) -> Vec<&str> {
    let mut keys: Vec<&str> = table.keys().map(String::as_str).collect();
    keys.sort_unstable();
    keys
}

#[test]
fn vector_policy_has_four_sets() {
    let text = include_str!("../vector-policy.toml");
    let table: toml::Table = text.parse().expect("vector-policy.toml parses");
    assert_eq!(keys(&table), ["Cpu", "Gpu"], "vector-policy.toml: classes");
    for class in ["Cpu", "Gpu"] {
        let section = sub(&table, class, "vector-policy.toml");
        let want: &[&str] = if class == "Cpu" {
            &["cpu_block", "cpu_units", "quad_tile", "vector_size"]
        } else {
            &["quad_tile", "vector_size"]
        };
        assert_eq!(keys(section), want, "[{class}]: tables");
        assert_eq!(
            keys(sub(section, "quad_tile", class)),
            ["4", "8"],
            "[{class}] quad_tile: scalar sizes"
        );
        for &table in want.iter().filter(|&&t| t != "quad_tile") {
            let rows = sub(section, table, class);
            let names: Vec<String> = KINDS.iter().map(|k| format!("{k:?}")).collect();
            let mut want_rows: Vec<&str> = names.iter().map(String::as_str).collect();
            want_rows.sort_unstable();
            assert_eq!(keys(rows), want_rows, "[{class}.{table}]: kinds");
            for kind in &want_rows {
                let row = sub(rows, kind, &format!("[{class}.{table}]"));
                assert_eq!(
                    keys(row),
                    ["4", "8"],
                    "[{class}.{table}] {kind}: scalar sizes"
                );
            }
        }
    }
    // The policy of the test device parses (one of the two sets).
    let client = tetraloma_testing::test_client();
    let policy = VectorPolicy::for_client(&client);
    for kind in KINDS {
        for bytes in [4, 8] {
            assert!(policy.vector_size(kind, bytes).is_power_of_two());
        }
    }
    let _ = DeviceClass::for_client(&client);
}
