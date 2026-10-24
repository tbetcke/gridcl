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
    /// Kernels not on the candidate list; always vector size 1 (ADR-0015).
    Scalar,
}

impl KernelKind {
    const ALL: [KernelKind; 13] = [
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
            KernelKind::Scalar => "Scalar",
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

    /// The scoped override of `kind`, else the scoped override, else `GRIDCL_VECTOR_SIZE` (read
    /// once per process), else `Policy`.
    pub(crate) fn current(kind: KernelKind) -> VectorSizeOverride {
        static ENV: OnceLock<Option<VectorSizeOverride>> = OnceLock::new();
        testing::kind_vector_override(kind)
            .or_else(testing::vector_override)
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

/// The bytes of one CPU block unit in vector-policy.toml.
const CPU_BLOCK_UNIT: usize = 4 << 10;
/// The CPU block of kinds and scalar sizes the policy does not list: 16 KiB.
const DEFAULT_CPU_BLOCK_BYTES: usize = 16 << 10;

/// A CPU block size: a power of two from 4 KiB to 64 KiB.
pub(crate) fn is_cpu_block_bytes(bytes: usize) -> bool {
    bytes.is_power_of_two() && (CPU_BLOCK_UNIT..=16 * CPU_BLOCK_UNIT).contains(&bytes)
}

/// A power of two that divides `VECTOR_PAD`.
pub(crate) fn is_vector_size(n: usize) -> bool {
    n.is_power_of_two() && VECTOR_PAD.is_multiple_of(n)
}

/// The value of `var`, or `None` when it is unset or empty.
fn read_env(var: &str) -> Option<String> {
    std::env::var(var).ok().filter(|v| !v.is_empty())
}

/// Vector sizes, QuadField tiles and CPU block sizes chosen from benchmarks
/// (docs/perf/vectorization.md), per (`KernelKind`, `DeviceClass`, element size), compiled in
/// from gridcl-device/vector-policy.toml (ADR-0015). `GRIDCL_VECTOR_SIZE` and
/// `GRIDCL_QUAD_TILE` override it for tests and benchmarks.
///
/// # Example
///
/// ```
/// use gridcl_device::VectorPolicy;
///
/// let client = gridcl_testing::test_client();
/// assert!([1, 2, 4, 8, 16].contains(&VectorPolicy::for_client(&client).quad_tile(4)));
/// ```
#[derive(Debug)]
pub struct VectorPolicy {
    vector_sizes: HashMap<(KernelKind, usize), usize>,
    /// Tile width per scalar size.
    quad_tiles: HashMap<usize, usize>,
    /// CPU block in bytes per kind and scalar size; `None` on `Gpu`.
    cpu_blocks: Option<HashMap<(KernelKind, usize), usize>>,
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

    /// Tile width of QuadField and E-vector chunks for one scalar size: 1, 2, 4, 8 or 16. The
    /// scoped override (`testing::with_quad_tile`), then `GRIDCL_QUAD_TILE`, then the policy;
    /// 1 for scalar sizes the policy does not list. The overrides apply to every scalar size.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::VectorPolicy;
    ///
    /// let client = gridcl_testing::test_client();
    /// let tile = VectorPolicy::for_client(&client).quad_tile(size_of::<f32>());
    /// assert!([1, 2, 4, 8, 16].contains(&tile));
    /// ```
    pub fn quad_tile(&self, elem_bytes: usize) -> usize {
        static ENV: OnceLock<Option<usize>> = OnceLock::new();
        testing::quad_tile_override()
            .or(*ENV.get_or_init(|| {
                read_env(QUAD_TILE_VAR)
                    .map(|v| parse_quad_tile(&v).unwrap_or_else(|e| panic!("{e}")))
            }))
            .unwrap_or_else(|| self.quad_tiles.get(&elem_bytes).copied().unwrap_or(1))
    }

    /// CPU block of one kind and element size in bytes (vector-policy.toml stores units of
    /// 4 KiB); `None` on `Gpu`. The scoped override (`testing::with_cpu_block_bytes`), then the
    /// policy; 16 KiB for kinds and scalar sizes the policy does not list.
    ///
    /// # Example
    ///
    /// ```
    /// use gridcl_device::{DeviceClass, KernelKind, VectorPolicy};
    ///
    /// let client = gridcl_testing::test_client();
    /// let block = VectorPolicy::for_client(&client).cpu_block_bytes(KernelKind::Fill, 4);
    /// match DeviceClass::for_client(&client) {
    ///     DeviceClass::Gpu => assert_eq!(block, None),
    ///     _ => assert!(block.is_some_and(|b| b.is_power_of_two() && (4096..=65536).contains(&b))),
    /// }
    /// ```
    pub fn cpu_block_bytes(&self, kind: KernelKind, elem_bytes: usize) -> Option<usize> {
        let blocks = self.cpu_blocks.as_ref()?;
        Some(testing::cpu_block_override().unwrap_or_else(|| {
            blocks
                .get(&(kind, elem_bytes))
                .copied()
                .unwrap_or(DEFAULT_CPU_BLOCK_BYTES)
        }))
    }

    /// The policy of `class` in the TOML text `text`.
    fn parse(text: &str, class: DeviceClass) -> Result<VectorPolicy, String> {
        let table: toml::Table = text.parse().map_err(|e| format!("{e}"))?;
        let c = class.name();
        let section = table
            .get(c)
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("no table [{c}]"))?;
        let quad_tile = section
            .get("quad_tile")
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("[{c}] has no table quad_tile"))?;
        let quad_tiles = parse_sizes(quad_tile, &format!("[{c}] quad_tile"), |t| {
            is_vector_size(t).then_some(t).ok_or("not 1, 2, 4, 8 or 16")
        })?;
        let sizes = section
            .get("vector_size")
            .and_then(|v| v.as_table())
            .ok_or_else(|| format!("no table [{c}.vector_size]"))?;
        let mut vector_sizes = HashMap::new();
        for kind in KernelKind::ALL {
            let what = format!("[{c}.vector_size] {}", kind.name());
            let row = parse_kind_row(sizes, kind, &what, |n| {
                if !is_vector_size(n) {
                    Err("not a power of two dividing VECTOR_PAD")
                } else if kind == KernelKind::Scalar && n != 1 {
                    Err("Scalar kernels run at vector size 1")
                } else {
                    Ok(n)
                }
            })?;
            vector_sizes.extend(row.into_iter().map(|(bytes, n)| ((kind, bytes), n)));
        }
        let cpu_blocks = match (class.is_cpu(), section.get("cpu_block")) {
            (false, None) => None,
            (false, Some(_)) => return Err(format!("[{c}.cpu_block] on a GPU class")),
            (true, None) => return Err(format!("no table [{c}.cpu_block]")),
            (true, Some(blocks)) => {
                let blocks = blocks
                    .as_table()
                    .ok_or_else(|| format!("[{c}.cpu_block] is not a table"))?;
                let mut bytes_of = HashMap::new();
                for kind in KernelKind::ALL {
                    let what = format!("[{c}.cpu_block] {}", kind.name());
                    let row = parse_kind_row(blocks, kind, &what, |units| {
                        units
                            .checked_mul(CPU_BLOCK_UNIT)
                            .filter(|&b| is_cpu_block_bytes(b))
                            .ok_or("not a power of two from 1 to 16 (units of 4 KiB)")
                    })?;
                    bytes_of.extend(row.into_iter().map(|(bytes, b)| ((kind, bytes), b)));
                }
                Some(bytes_of)
            }
        };
        Ok(VectorPolicy {
            vector_sizes,
            quad_tiles,
            cpu_blocks,
        })
    }
}

/// The row of `kind` in the per-kind table `table`, by scalar size, through `check`.
fn parse_kind_row(
    table: &toml::Table,
    kind: KernelKind,
    what: &str,
    check: impl Fn(usize) -> Result<usize, &'static str>,
) -> Result<HashMap<usize, usize>, String> {
    let row = table
        .get(kind.name())
        .and_then(|v| v.as_table())
        .ok_or_else(|| format!("{what}: missing"))?;
    parse_sizes(row, what, check)
}

/// A table `{ <scalar size> = <value> }`, each value through `check`.
fn parse_sizes(
    table: &toml::Table,
    what: &str,
    check: impl Fn(usize) -> Result<usize, &'static str>,
) -> Result<HashMap<usize, usize>, String> {
    let mut out = HashMap::new();
    for (bytes, v) in table {
        let what = || format!("{what} {bytes}");
        let bytes: usize = bytes
            .parse()
            .map_err(|_| format!("{}: not a scalar size", what()))?;
        let v = v
            .as_integer()
            .and_then(|v| usize::try_from(v).ok())
            .ok_or_else(|| format!("{}: not a non-negative integer", what()))?;
        let v = check(v).map_err(|e| format!("{}: {v} is {e}", what()))?;
        out.insert(bytes, v);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// M3s acceptance (bullet 4): every kind and class has `vector_size` and per-size
    /// `quad_tile` entries, and every CPU class `cpu_block` entries; every value is a power of
    /// two from 1 to 16, and `Scalar` runs at vector size 1.
    #[test]
    fn policy_covers_every_kind_and_class() {
        let table: toml::Table = POLICY_TOML.parse().unwrap();
        // The value of `key` in `table`: a power of two from 1 to 16.
        let entry = |table: &toml::Table, key: &str, what: &str| -> usize {
            let v = table
                .get(key)
                .and_then(|v| v.as_integer())
                .unwrap_or_else(|| panic!("{what}: no integer entry {key}"));
            assert!(
                (1..=16).contains(&v) && (v as usize).is_power_of_two(),
                "{what} {key} = {v} is not a power of two from 1 to 16"
            );
            v as usize
        };
        let sub = |table: &toml::Table, key: &str, what: &str| -> toml::Table {
            table
                .get(key)
                .and_then(|v| v.as_table())
                .unwrap_or_else(|| panic!("{what}: no table {key}"))
                .clone()
        };
        for class in DeviceClass::ALL {
            let section = sub(&table, class.name(), "vector-policy.toml");
            let quad_tile = sub(&section, "quad_tile", class.name());
            let sizes = sub(&section, "vector_size", class.name());
            let blocks = section.get("cpu_block").and_then(|v| v.as_table());
            assert_eq!(
                blocks.is_some(),
                class.is_cpu(),
                "[{}.cpu_block] exists exactly on CPU classes",
                class.name()
            );
            let policy = VectorPolicy::parse(POLICY_TOML, class).unwrap();
            for bytes in ["4", "8"] {
                entry(&quad_tile, bytes, &format!("[{}] quad_tile", class.name()));
            }
            for kind in KernelKind::ALL {
                let what = format!("[{}.vector_size] {}", class.name(), kind.name());
                let row = sub(&sizes, kind.name(), &what);
                for bytes in [4usize, 8] {
                    let n = entry(&row, &bytes.to_string(), &what);
                    assert_eq!(policy.vector_size(kind, bytes), n, "{what} {bytes}");
                    if kind == KernelKind::Scalar {
                        assert_eq!(n, 1, "{what} {bytes}");
                    }
                }
                if let Some(blocks) = blocks {
                    let what = format!("[{}.cpu_block] {}", class.name(), kind.name());
                    let row = sub(blocks, kind.name(), &what);
                    for bytes in [4usize, 8] {
                        let units = entry(&row, &bytes.to_string(), &what);
                        assert_eq!(
                            policy.cpu_block_bytes(kind, bytes),
                            Some(units << 12),
                            "{what} {bytes}"
                        );
                    }
                } else {
                    assert_eq!(policy.cpu_block_bytes(kind, 4), None, "{class:?}");
                }
            }
        }
    }

    #[test]
    fn invalid_policy_entries_are_rejected() {
        let bad_size = POLICY_TOML.replacen("Fill = { 4 = 1", "Fill = { 4 = 3", 1);
        assert!(VectorPolicy::parse(&bad_size, DeviceClass::CpuSimd128).is_err());
        let bad_tile = POLICY_TOML.replacen("quad_tile = { 4 = 16", "quad_tile = { 4 = 32", 1);
        assert!(VectorPolicy::parse(&bad_tile, DeviceClass::CpuSimd128).is_err());
        let scalar_tile = POLICY_TOML.replacen("quad_tile = { 4 = 16", "quad_tile = 16", 1);
        assert!(VectorPolicy::parse(&scalar_tile, DeviceClass::CpuSimd128).is_err());
        for units in ["0", "2", "3", "32"] {
            let bad_block =
                POLICY_TOML.replacen("Fill = { 4 = 4", &format!("Fill = {{ 4 = {units}"), 1);
            let accepted = VectorPolicy::parse(&bad_block, DeviceClass::CpuSimd128).is_ok();
            assert_eq!(accepted, units == "2", "cpu_block {units}");
        }
        let no_blocks = POLICY_TOML.replacen("[CpuSimd128.cpu_block]", "[CpuSimd128.other]", 1);
        assert!(VectorPolicy::parse(&no_blocks, DeviceClass::CpuSimd128).is_err());
        let gpu_blocks = format!("{POLICY_TOML}[Gpu.cpu_block]\nFill = {{ 4 = 4, 8 = 4 }}\n");
        assert!(VectorPolicy::parse(&gpu_blocks, DeviceClass::Gpu).is_err());
        let missing = POLICY_TOML.replacen("PullBack = { 4 = 1, 8 = 1 }\n", "", 1);
        assert!(VectorPolicy::parse(&missing, DeviceClass::CpuSimd128).is_err());
        let vector_scalar = POLICY_TOML.replacen("Scalar = { 4 = 1", "Scalar = { 4 = 2", 1);
        assert!(VectorPolicy::parse(&vector_scalar, DeviceClass::CpuSimd128).is_err());
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
