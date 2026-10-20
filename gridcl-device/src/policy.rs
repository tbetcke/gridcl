//! `KernelKind`, `DeviceClass`, `VectorSizeOverride` and `VectorPolicy` (Listing S2).

use std::collections::HashMap;
use std::sync::OnceLock;

use cubecl::ir::DeviceProperties;
use cubecl::prelude::*;

use crate::{VECTOR_PAD, testing};

/// Kernel families with an entry in the vector policy.
///
/// # Example
///
/// ```
/// use gridcl_device::{KernelKind, VectorPolicy};
///
/// let client = gridcl_testing::test_client();
/// let n = VectorPolicy::for_client(&client).vector_size(KernelKind::Fill, 4);
/// assert!(n.is_power_of_two());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum KernelKind {
    Fill,
    Copy,
    Reduce,
    ScanTile,
    SortDigits,
    Gather,
    Pack,
    TopologyKeys,
    PermutationInfo,
    EVector,
    GeomFactors,
    PullBack,
}

impl KernelKind {
    const ALL: [KernelKind; 12] = [
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
    ];

    /// The key of the family in vector-policy.toml.
    fn name(self) -> &'static str {
        match self {
            KernelKind::Fill => "Fill",
            KernelKind::Copy => "Copy",
            KernelKind::Reduce => "Reduce",
            KernelKind::ScanTile => "ScanTile",
            KernelKind::SortDigits => "SortDigits",
            KernelKind::Gather => "Gather",
            KernelKind::Pack => "Pack",
            KernelKind::TopologyKeys => "TopologyKeys",
            KernelKind::PermutationInfo => "PermutationInfo",
            KernelKind::EVector => "EVector",
            KernelKind::GeomFactors => "GeomFactors",
            KernelKind::PullBack => "PullBack",
        }
    }
}

/// Device classes the policy distinguishes, derived from `Client::properties()`.
///
/// # Example
///
/// ```
/// use gridcl_device::DeviceClass;
///
/// let client = gridcl_testing::test_client();
/// let class = DeviceClass::for_client(&client);
/// assert_eq!(class == DeviceClass::Gpu, client.properties().hardware.num_cpu_cores.is_none());
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DeviceClass {
    CpuSimd128,
    CpuSimd256,
    CpuSimd512,
    Gpu,
}

impl DeviceClass {
    const ALL: [DeviceClass; 4] = [
        DeviceClass::CpuSimd128,
        DeviceClass::CpuSimd256,
        DeviceClass::CpuSimd512,
        DeviceClass::Gpu,
    ];

    /// A device with `hardware.num_cpu_cores` is a CPU, classed by `hardware.load_width` (512
    /// bits: CpuSimd512, 256: CpuSimd256, else CpuSimd128); every other device is Gpu.
    /// `load_width` alone does not tell a CPU from wgpu, which also reports 128 bits.
    pub fn for_client(client: &Client) -> DeviceClass {
        DeviceClass::from_properties(&testing::properties(client))
    }

    pub(crate) fn from_properties(props: &DeviceProperties) -> DeviceClass {
        match (props.hardware.num_cpu_cores, props.hardware.load_width) {
            (None, _) => DeviceClass::Gpu,
            (Some(_), 512) => DeviceClass::CpuSimd512,
            (Some(_), 256) => DeviceClass::CpuSimd256,
            (Some(_), _) => DeviceClass::CpuSimd128,
        }
    }

    pub(crate) fn is_cpu(self) -> bool {
        self != DeviceClass::Gpu
    }

    fn name(self) -> &'static str {
        match self {
            DeviceClass::CpuSimd128 => "CpuSimd128",
            DeviceClass::CpuSimd256 => "CpuSimd256",
            DeviceClass::CpuSimd512 => "CpuSimd512",
            DeviceClass::Gpu => "Gpu",
        }
    }

    fn index(self) -> usize {
        DeviceClass::ALL
            .iter()
            .position(|&c| c == self)
            .expect("every class is in ALL")
    }
}

/// The environment variable that overrides the vector size of every plan.
pub(crate) const VECTOR_SIZE_VAR: &str = "GRIDCL_VECTOR_SIZE";
/// The environment variable that overrides the QuadField tile width.
pub(crate) const QUAD_TILE_VAR: &str = "GRIDCL_QUAD_TILE";

/// Value of `GRIDCL_VECTOR_SIZE` (`scalar | policy | widest | <n>`) and of
/// `testing::with_vector_size`. `Widest` is the first of `io_optimized_vector_sizes(elem_bytes)`.
/// `Fixed(n)` and `Widest` skip the register budget but stay clamped by `max_vector` (and by the
/// device's `max_vector_size`); `Policy` applies the full rule of
/// [`LaunchPlan::elementwise`](crate::LaunchPlan::elementwise).
///
/// # Example
///
/// ```
/// use gridcl_device::{KernelKind, LaunchPlan, VectorSizeOverride};
/// use gridcl_device::testing::with_vector_size;
///
/// let client = gridcl_testing::test_client();
/// let plan = with_vector_size(VectorSizeOverride::Scalar, || {
///     LaunchPlan::elementwise(&client, KernelKind::Fill, 1000, 4, 16, 1)
/// })
/// .unwrap();
/// assert_eq!(plan.vector_size, 1);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorSizeOverride {
    Scalar,
    Policy,
    Widest,
    Fixed(usize),
}

impl VectorSizeOverride {
    /// Parses a value of `GRIDCL_VECTOR_SIZE`: `scalar`, `policy`, `widest`, or a power of two
    /// from 1 to `VECTOR_PAD`.
    pub(crate) fn parse(value: &str) -> Result<VectorSizeOverride, String> {
        match value {
            "scalar" => Ok(VectorSizeOverride::Scalar),
            "policy" => Ok(VectorSizeOverride::Policy),
            "widest" => Ok(VectorSizeOverride::Widest),
            n => match n.parse::<usize>() {
                Ok(n) if is_vector_size(n) => Ok(VectorSizeOverride::Fixed(n)),
                _ => Err(format!(
                    "{VECTOR_SIZE_VAR}={value}: expected scalar, policy, widest or a power of two \
                     from 1 to {VECTOR_PAD}"
                )),
            },
        }
    }

    /// The scoped override, else `GRIDCL_VECTOR_SIZE` (read once per process), else `Policy`.
    pub(crate) fn current() -> VectorSizeOverride {
        static ENV: OnceLock<Option<VectorSizeOverride>> = OnceLock::new();
        testing::vector_override()
            .or(*ENV.get_or_init(|| {
                read_env(VECTOR_SIZE_VAR)
                    .map(|v| VectorSizeOverride::parse(&v).unwrap_or_else(|e| panic!("{e}")))
            }))
            .unwrap_or(VectorSizeOverride::Policy)
    }
}

/// Parses a value of `GRIDCL_QUAD_TILE`: 1, 2, 4, 8 or 16.
pub(crate) fn parse_quad_tile(value: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(n) if is_vector_size(n) => Ok(n),
        _ => Err(format!(
            "{QUAD_TILE_VAR}={value}: expected 1, 2, 4, 8 or 16"
        )),
    }
}

/// A power of two that divides `VECTOR_PAD`.
pub(crate) fn is_vector_size(n: usize) -> bool {
    n.is_power_of_two() && VECTOR_PAD.is_multiple_of(n)
}

/// The value of `var`, or `None` when it is unset or empty.
fn read_env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

/// Vector sizes and tile widths chosen from benchmarks (docs/perf/vectorization.md), compiled in
/// from gridcl-device/vector-policy.toml. `GRIDCL_VECTOR_SIZE` and `GRIDCL_QUAD_TILE` override
/// it for tests and benchmarks.
///
/// # Example
///
/// ```
/// use gridcl_device::VectorPolicy;
///
/// let client = gridcl_testing::test_client();
/// assert!([1, 2, 4, 8, 16].contains(&VectorPolicy::for_client(&client).quad_tile()));
/// ```
#[derive(Debug)]
pub struct VectorPolicy {
    vector_sizes: HashMap<(KernelKind, usize), usize>,
    quad_tile: usize,
}

/// The policy file, compiled in.
const POLICY_TOML: &str = include_str!("../vector-policy.toml");

impl VectorPolicy {
    /// The policy of the client's device class, parsed once per class.
    pub fn for_client(client: &Client) -> &'static VectorPolicy {
        VectorPolicy::for_class(DeviceClass::for_client(client))
    }

    pub(crate) fn for_class(class: DeviceClass) -> &'static VectorPolicy {
        static POLICIES: [OnceLock<VectorPolicy>; 4] = [
            OnceLock::new(),
            OnceLock::new(),
            OnceLock::new(),
            OnceLock::new(),
        ];
        POLICIES[class.index()].get_or_init(|| {
            VectorPolicy::parse(POLICY_TOML, class)
                .unwrap_or_else(|e| panic!("gridcl-device/vector-policy.toml: {e}"))
        })
    }

    /// Preferred vector size for one kernel family and scalar size; 1 until benchmarked, and 1
    /// for scalar sizes the policy does not list.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{KernelKind, VECTOR_PAD, VectorPolicy};
    ///
    /// let client = gridcl_testing::test_client();
    /// let n = VectorPolicy::for_client(&client).vector_size(KernelKind::Fill, 4);
    /// assert!(n.is_power_of_two() && VECTOR_PAD % n == 0);
    /// ```
    pub fn vector_size(&self, kind: KernelKind, elem_bytes: usize) -> usize {
        self.vector_sizes
            .get(&(kind, elem_bytes))
            .copied()
            .unwrap_or(1)
    }

    /// Tile width of QuadField and E-vector chunks: 1, 2, 4, 8 or 16. The scoped override
    /// (`testing::with_quad_tile`), then `GRIDCL_QUAD_TILE`, then the policy.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::VectorPolicy;
    ///
    /// let client = gridcl_testing::test_client();
    /// let tile = VectorPolicy::for_client(&client).quad_tile();
    /// assert!([1, 2, 4, 8, 16].contains(&tile));
    /// ```
    pub fn quad_tile(&self) -> usize {
        static ENV: OnceLock<Option<usize>> = OnceLock::new();
        testing::quad_tile_override()
            .or(*ENV.get_or_init(|| {
                read_env(QUAD_TILE_VAR)
                    .map(|v| parse_quad_tile(&v).unwrap_or_else(|e| panic!("{e}")))
            }))
            .unwrap_or(self.quad_tile)
    }

    /// The policy of `class` in the TOML text `text`.
    fn parse(text: &str, class: DeviceClass) -> Result<VectorPolicy, String> {
        let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
        let section = table
            .get(class.name())
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("no table [{}]", class.name()))?;
        let quad_tile = section
            .get("quad_tile")
            .and_then(|v| v.as_integer())
            .ok_or_else(|| format!("[{}] has no integer quad_tile", class.name()))?;
        let quad_tile = usize::try_from(quad_tile)
            .ok()
            .filter(|&t| is_vector_size(t))
            .ok_or_else(|| {
                format!(
                    "[{}] quad_tile {quad_tile} is not 1, 2, 4, 8 or 16",
                    class.name()
                )
            })?;
        let sizes = section
            .get("vector_size")
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("no table [{}.vector_size]", class.name()))?;
        let mut vector_sizes = HashMap::new();
        for kind in KernelKind::ALL {
            let entry = sizes
                .get(kind.name())
                .and_then(|v| v.as_table())
                .ok_or_else(|| format!("[{}.vector_size] has no {}", class.name(), kind.name()))?;
            for (bytes, n) in entry {
                let what = || format!("[{}.vector_size] {} {bytes}", class.name(), kind.name());
                let bytes: usize = bytes
                    .parse()
                    .map_err(|_| format!("{}: not a scalar size", what()))?;
                let n = n
                    .as_integer()
                    .and_then(|n| usize::try_from(n).ok())
                    .filter(|&n| is_vector_size(n))
                    .ok_or_else(|| {
                        format!("{}: not a power of two dividing {VECTOR_PAD}", what())
                    })?;
                vector_sizes.insert((kind, bytes), n);
            }
        }
        Ok(VectorPolicy {
            vector_sizes,
            quad_tile,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_policy_file_parses_with_one_and_sixteen_everywhere() {
        for class in DeviceClass::ALL {
            let policy = VectorPolicy::parse(POLICY_TOML, class).unwrap();
            assert_eq!(policy.quad_tile, 16, "{class:?}");
            for kind in KernelKind::ALL {
                for bytes in [4, 8] {
                    assert_eq!(
                        policy.vector_size(kind, bytes),
                        1,
                        "{class:?} {kind:?} {bytes}"
                    );
                }
            }
            assert_eq!(policy.vector_sizes.len(), 2 * KernelKind::ALL.len());
        }
    }

    #[test]
    fn invalid_policy_entries_are_rejected() {
        let bad_size = POLICY_TOML.replacen("Fill = { 4 = 1", "Fill = { 4 = 3", 1);
        assert!(VectorPolicy::parse(&bad_size, DeviceClass::CpuSimd128).is_err());
        let bad_tile = POLICY_TOML.replacen("quad_tile = 16", "quad_tile = 32", 1);
        assert!(VectorPolicy::parse(&bad_tile, DeviceClass::CpuSimd128).is_err());
        let missing = POLICY_TOML.replacen("PullBack = { 4 = 1, 8 = 1 }\n", "", 1);
        assert!(VectorPolicy::parse(&missing, DeviceClass::CpuSimd128).is_err());
    }

    #[test]
    fn override_parsing() {
        assert_eq!(
            VectorSizeOverride::parse("scalar"),
            Ok(VectorSizeOverride::Scalar)
        );
        assert_eq!(
            VectorSizeOverride::parse("policy"),
            Ok(VectorSizeOverride::Policy)
        );
        assert_eq!(
            VectorSizeOverride::parse("widest"),
            Ok(VectorSizeOverride::Widest)
        );
        for n in [1, 2, 4, 8, 16] {
            assert_eq!(
                VectorSizeOverride::parse(&n.to_string()),
                Ok(VectorSizeOverride::Fixed(n))
            );
        }
        for bad in ["", "0", "3", "32", "-1", "Widest", " 4", "four"] {
            let e = VectorSizeOverride::parse(bad).unwrap_err();
            assert!(e.starts_with("GRIDCL_VECTOR_SIZE="), "{e}");
        }
        assert_eq!(parse_quad_tile("8"), Ok(8));
        for bad in ["0", "12", "32", "x"] {
            assert!(
                parse_quad_tile(bad)
                    .unwrap_err()
                    .starts_with("GRIDCL_QUAD_TILE=")
            );
        }
    }

    #[test]
    fn device_class_from_properties() {
        let mut props = gridcl_testing::test_client().properties().clone();
        props.hardware.num_cpu_cores = None;
        for width in [128, 256, 512] {
            props.hardware.load_width = width;
            assert_eq!(DeviceClass::from_properties(&props), DeviceClass::Gpu);
        }
        props.hardware.num_cpu_cores = Some(8);
        for (width, class) in [
            (64, DeviceClass::CpuSimd128),
            (128, DeviceClass::CpuSimd128),
            (256, DeviceClass::CpuSimd256),
            (512, DeviceClass::CpuSimd512),
        ] {
            props.hardware.load_width = width;
            assert_eq!(DeviceClass::from_properties(&props), class);
        }
    }
}
