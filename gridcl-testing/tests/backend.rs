//! M0 follow-up (finding 9; docs/plans/M2.md, T10): the test client runs on the backend
//! `GRIDCL_TEST_DEVICE` names, not on a fallback.
//!
//! `Client::name()` is the runtime name CubeCL 0.11 passes to `ServerUtilities::init`:
//!
//! | `GRIDCL_TEST_DEVICE` | `client.name()` | Source (cubecl-*-0.11.0) |
//! | --- | --- | --- |
//! | unset, empty or `cpu` | `cpu` | cubecl-cpu, `src/runtime.rs` |
//! | `cuda` | `cuda` | cubecl-cuda, `src/runtime.rs` |
//! | `hip` | `hip` | cubecl-hip, `src/runtime.rs` |
//! | `vulkan` | `wgpu<spirv>` | cubecl-wgpu, `runtime_name` in `src/runtime.rs` |
//! | `metal` | `wgpu<msl>` | cubecl-wgpu, `runtime_name` in `src/runtime.rs` |
//! | `metal-native` | `metal` | cubecl-metal, `src/runtime.rs` |
//! | `wgpu` | `wgpu<wgsl>`, `wgpu<msl>` or `wgpu<spirv>` | cubecl-wgpu, `runtime_name` |
//!
//! cubecl-wgpu falls back to `wgpu<wgsl>` when the adapter lacks what the SPIR-V or MSL
//! compiler needs; for `vulkan` and `metal` that fallback is a failure here. For `wgpu` the
//! compiler depends on the adapter and the enabled features, so the test only checks the
//! prefix and prints the name.

use gridcl_testing::{TEST_DEVICE_VAR, test_client};

/// The names `client.name()` may return for a `GRIDCL_TEST_DEVICE` value.
fn expected_names(device: &str) -> &'static [&'static str] {
    match device {
        "" | "cpu" => &["cpu"],
        "cuda" => &["cuda"],
        "hip" => &["hip"],
        "vulkan" => &["wgpu<spirv>"],
        "metal" => &["wgpu<msl>"],
        "metal-native" => &["metal"],
        "wgpu" => &["wgpu<wgsl>", "wgpu<msl>", "wgpu<spirv>"],
        other => panic!("{TEST_DEVICE_VAR}={other}: no expected runtime name"),
    }
}

#[test]
fn test_client_matches_requested_backend() {
    let device = std::env::var(TEST_DEVICE_VAR).unwrap_or_default();
    let client = test_client();
    let name = client.name();
    println!("{TEST_DEVICE_VAR}={device:?}: client.name() = {name:?}");
    let expected = expected_names(&device);
    assert!(
        expected.contains(&name),
        "{TEST_DEVICE_VAR}={device:?} runs on {name:?}, expected one of {expected:?}"
    );
}
