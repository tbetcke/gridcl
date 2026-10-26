//! The vector-size sweep: every family, scalar type, size and configuration of "Benchmarks",
//! measured with the protocol of [`crate::protocol`].
//!
//! - [`device`]: the device-primitive families (Fill, Copy, Reduce, ScanTile, SortDigits,
//!   Gather) and the scalar helpers measured for `KernelKind::Scalar`'s CPU blocks;
//! - [`mesh`]: the mesh families (TopologyKeys, PermutationInfo, EVector, GeomFactors), with
//!   cells as items.

pub mod device;
pub mod mesh;

use std::collections::BTreeMap;

use cubecl::prelude::Client;
use gridcl_device::{DeviceClass, KernelKind, check_precision};

use crate::protocol::{
    self, Config, Measured, Workload, bandwidth_ratio, is_cpu, measure, with_config,
};
use crate::record::{Cores, CubeDimMode, Scalar, SweepRecord};

/// The protocol's problem sizes.
pub const SIZES: [usize; 3] = [100_000, 1_000_000, 10_000_000];
/// CPU block targets of the protocol, in bytes.
pub const BLOCKS: [usize; 3] = [4 << 10, 16 << 10, 64 << 10];
/// QuadField tile widths of the protocol (GeomFactors).
pub const TILES: [usize; 3] = [1, 4, 16];

/// The families the sweep measures, with their scalar types.
pub fn families() -> Vec<(KernelKind, &'static [Scalar])> {
    use Scalar::{F32, F64, U32};
    vec![
        (KernelKind::Fill, &[F32, F64, U32]),
        (KernelKind::Copy, &[U32]),
        (KernelKind::Reduce, &[F32, F64, U32]),
        (KernelKind::ScanTile, &[U32]),
        (KernelKind::SortDigits, &[U32]),
        (KernelKind::Gather, &[F32, F64, U32]),
        (KernelKind::TopologyKeys, &[U32]),
        (KernelKind::PermutationInfo, &[U32]),
        (KernelKind::EVector, &[F32, F64]),
        (KernelKind::GeomFactors, &[F32, F64]),
        // Histogram and segmented reduce, for the CPU blocks of Scalar (ADR-0015, item 7).
        (KernelKind::Scalar, &[U32, F64]),
    ]
}

/// The workload of `kind` in `scalar` at size `n` (about `n` cells for the mesh families).
///
/// # Errors
///
/// If the sweep does not measure `kind` in `scalar`, or the inputs cannot be created.
pub fn workload(
    client: &Client,
    kind: KernelKind,
    scalar: Scalar,
    n: usize,
) -> Result<Box<dyn Workload>, String> {
    device::workload(client, kind, scalar, n)
        .or_else(|| mesh::workload(client, kind, scalar, n))
        .unwrap_or_else(|| Err(format!("{kind:?} is not measured in {}", scalar.name())))
}

/// What the sweep runs.
#[derive(Clone, Debug)]
pub struct SweepOptions {
    /// Machine: `m3max`, or a target of scripts/hpc/remote-test.sh.
    pub target: String,
    pub families: Vec<KernelKind>,
    /// Scalar types to keep; `None` measures every type of each family.
    pub scalars: Option<Vec<Scalar>>,
    pub sizes: Vec<usize>,
    /// Vector sizes to keep; `None` measures every size the device offers.
    pub vector_sizes: Option<Vec<usize>>,
    /// CPU block targets in bytes (CPU devices only).
    pub blocks: Vec<usize>,
    pub cube_dims: Vec<CubeDimMode>,
    pub runs: usize,
    pub fallback_reason: Option<String>,
}

/// What every record of a run shares.
#[derive(Clone, Debug)]
pub struct Machine {
    pub backend: String,
    pub device: String,
    pub class: DeviceClass,
    pub commit: String,
    pub cores: Cores,
    pub kernel_cache: String,
}

impl Machine {
    pub fn new(client: &Client) -> Machine {
        Machine {
            backend: protocol::backend(),
            device: client.properties().identity.name.clone(),
            class: DeviceClass::for_client(client),
            commit: protocol::commit(),
            cores: protocol::cores(),
            kernel_cache: protocol::kernel_cache().to_owned(),
        }
    }
}

/// Runs the sweep and passes each record to `on_record` as soon as it is measured.
///
/// Each (family, type, size) gets one set of inputs. The bandwidth reference is a u32 copy of
/// the same length under the policy, measured once per size. GeomFactors also runs every tile
/// width of [`TILES`], at the vector sizes the tile holds (a factor plan's widest size is its
/// tile; ADR-0012); the other families keep the policy's tile.
///
/// # Errors
///
/// The first failed measurement, naming its configuration.
pub fn sweep(
    client: &Client,
    machine: &Machine,
    options: &SweepOptions,
    on_record: &mut dyn FnMut(&SweepRecord),
) -> Result<Vec<SweepRecord>, String> {
    let class = machine.class;
    let has_f64 = check_precision::<f64>(client).is_ok();
    let vector_sizes: Vec<usize> = protocol::vector_sizes()
        .into_iter()
        .filter(|v| {
            options
                .vector_sizes
                .as_ref()
                .is_none_or(|keep| keep.contains(v))
        })
        .collect();
    let blocks: Vec<Option<usize>> = if is_cpu(class) {
        options.blocks.iter().copied().map(Some).collect()
    } else {
        vec![None]
    };
    let cube_dims: Vec<CubeDimMode> = options
        .cube_dims
        .iter()
        .copied()
        .filter(|&mode| {
            let keep =
                mode == CubeDimMode::Plan || (is_cpu(class) && machine.cores.performance.is_some());
            if !keep {
                eprintln!("vector_sweep: no performance-core cube_dim on this device; skipped");
            }
            keep
        })
        .collect();

    let mut copy_reference: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    let mut records = Vec::new();
    for (kind, scalars) in families() {
        if !options.families.contains(&kind) {
            continue;
        }
        // Scalar kernels plan at size 1; only their CPU blocks are chosen (ADR-0015, item 7).
        let scalar_kind = kind == KernelKind::Scalar;
        if scalar_kind && !is_cpu(class) {
            continue;
        }
        let kind_sizes: Vec<usize> = vector_sizes
            .iter()
            .copied()
            .filter(|&v| !scalar_kind || v == 1)
            .collect();
        let tiles: Vec<Option<usize>> = if kind == KernelKind::GeomFactors {
            TILES.iter().copied().map(Some).collect()
        } else {
            vec![None]
        };
        for &scalar in scalars {
            if options
                .scalars
                .as_ref()
                .is_some_and(|keep| !keep.contains(&scalar))
            {
                continue;
            }
            if scalar == Scalar::F64 && !has_f64 {
                eprintln!("vector_sweep: {kind:?} f64 skipped: the device has no f64");
                continue;
            }
            for &n in &options.sizes {
                let mut work = workload(client, kind, scalar, n)?;
                let reference = match copy_reference.get(&n) {
                    Some(&r) => r,
                    None => {
                        let r = measure_copy_reference(client, n, options.runs)?;
                        copy_reference.insert(n, r);
                        r
                    }
                };
                for &cube_dim in &cube_dims {
                    for (&block_bytes, &tile) in blocks
                        .iter()
                        .flat_map(|b| tiles.iter().map(move |t| (b, t)))
                    {
                        for &vector_size in &kind_sizes {
                            if tile.is_some_and(|t| vector_size > t) {
                                continue;
                            }
                            let config = Config {
                                vector_size,
                                block_bytes,
                                tile,
                                cube_dim,
                            };
                            let what = format!("{kind:?} {} n = {n} {config:?}", scalar.name());
                            let measured =
                                with_config(kind, &config, machine.cores.performance, || {
                                    measure(client, kind, options.runs, work.as_mut())
                                })
                                .and_then(|m| m)
                                .map_err(|e| format!("{what}: {e}"))?;
                            let record = sweep_record(
                                machine,
                                options,
                                (kind, scalar, n),
                                work.as_ref(),
                                &config,
                                &measured,
                                reference,
                            );
                            on_record(&record);
                            records.push(record);
                        }
                    }
                }
            }
        }
    }
    Ok(records)
}

/// Bytes and median of a u32 copy of `n` items under the policy.
fn measure_copy_reference(client: &Client, n: usize, runs: usize) -> Result<(usize, f64), String> {
    let mut copy = workload(client, KernelKind::Copy, Scalar::U32, n)?;
    let measured = measure(client, KernelKind::Copy, runs, copy.as_mut())
        .map_err(|e| format!("copy reference at n = {n}: {e}"))?;
    let bytes = copy.bytes().expect("copy has a byte count");
    Ok((bytes, measured.median_s()))
}

fn sweep_record(
    machine: &Machine,
    options: &SweepOptions,
    (family, scalar, nominal_n): (KernelKind, Scalar, usize),
    work: &dyn Workload,
    config: &Config,
    measured: &Measured,
    (copy_bytes, copy_median_s): (usize, f64),
) -> SweepRecord {
    let median_s = measured.median_s();
    SweepRecord {
        backend: machine.backend.clone(),
        device: machine.device.clone(),
        device_class: machine.class,
        target: options.target.clone(),
        family,
        scalar,
        nominal_n,
        n: work.items(),
        vector_size: config.vector_size,
        vector_size_ran: measured.vector_size_ran,
        tile: config.tile,
        block_bytes: config.block_bytes,
        cube_dim: config.cube_dim,
        units: measured.units,
        times_s: measured.times_s.clone(),
        median_s,
        calls_per_run: measured.calls_per_run,
        calibration_s: measured.calibration_s,
        compile_s: measured.compile_s(),
        kernel_cache: machine.kernel_cache.clone(),
        bandwidth_ratio: work
            .bytes()
            .map(|bytes| bandwidth_ratio(bytes, median_s, copy_bytes, copy_median_s)),
        commit: machine.commit.clone(),
        profile: protocol::profile().to_owned(),
        cores: machine.cores.clone(),
        fallback_reason: options.fallback_reason.clone(),
        node_load: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The mesh families, which [`smoke_sweep_at_ten_thousand`] leaves to
    /// [`mesh_smoke_sweep_at_a_thousand_cells`].
    const MESH: [KernelKind; 4] = [
        KernelKind::TopologyKeys,
        KernelKind::PermutationInfo,
        KernelKind::EVector,
        KernelKind::GeomFactors,
    ];

    /// The sweep at 10^4 items, two runs, every device family: every configuration gives a
    /// record whose size ran is at most the size asked for; Scalar runs on CPUs only, at size 1.
    #[test]
    fn smoke_sweep_at_ten_thousand() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let options = SweepOptions {
            target: "test".to_owned(),
            families: families()
                .into_iter()
                .map(|(k, _)| k)
                .filter(|k| !MESH.contains(k))
                .collect(),
            scalars: None,
            sizes: vec![10_000],
            vector_sizes: None,
            blocks: vec![4 << 10, 64 << 10],
            cube_dims: vec![CubeDimMode::Plan],
            runs: 2,
            fallback_reason: None,
        };
        let mut seen = 0;
        let records = sweep(&client, &machine, &options, &mut |_| seen += 1).unwrap();
        assert_eq!(seen, records.len());
        let sizes = protocol::vector_sizes().len();
        let blocks = if is_cpu(machine.class) { 2 } else { 1 };
        let has_f64 = check_precision::<f64>(&client).is_ok();
        let types = |kind: KernelKind| {
            let (_, scalars) = families().into_iter().find(|(k, _)| *k == kind).unwrap();
            scalars
                .iter()
                .filter(|&&s| has_f64 || s != Scalar::F64)
                .count()
        };
        for (kind, _) in families() {
            let of = records.iter().filter(|r| r.family == kind).count();
            let want = match kind {
                _ if MESH.contains(&kind) => 0,
                KernelKind::Scalar if is_cpu(machine.class) => types(kind) * blocks,
                KernelKind::Scalar => 0,
                _ => types(kind) * blocks * sizes,
            };
            assert_eq!(of, want, "{kind:?}");
        }
        for r in &records {
            assert_eq!((r.n, r.nominal_n, r.times_s.len()), (10_000, 10_000, 2));
            assert!(r.vector_size_ran <= r.vector_size, "{r:?}");
            assert!(r.median_s > 0.0 && r.units > 0, "{r:?}");
            let has_bytes = r.family != KernelKind::SortDigits;
            assert_eq!(r.bandwidth_ratio.is_some(), has_bytes, "{r:?}");
            assert!(r.bandwidth_ratio.is_none_or(|b| b > 0.0), "{r:?}");
            assert_eq!(r.block_bytes.is_some(), is_cpu(machine.class));
        }
    }

    /// The mesh families at about 10^3 cells, two runs, one block: the records count the cells
    /// of the boxes, GeomFactors runs every tile at the sizes it holds, and every warm-up passed
    /// its check.
    #[test]
    fn mesh_smoke_sweep_at_a_thousand_cells() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let options = SweepOptions {
            target: "test".to_owned(),
            families: MESH.to_vec(),
            scalars: None,
            sizes: vec![1000],
            vector_sizes: None,
            blocks: vec![16 << 10],
            cube_dims: vec![CubeDimMode::Plan],
            runs: 2,
            fallback_reason: None,
        };
        let records = sweep(&client, &machine, &options, &mut |_| ()).unwrap();
        let sizes = protocol::vector_sizes();
        let has_f64 = check_precision::<f64>(&client).is_ok();
        let types = if has_f64 { 2 } else { 1 };
        let tet_cells = 6 * 5 * 5 * 5;
        for kind in MESH {
            let of: Vec<&SweepRecord> = records.iter().filter(|r| r.family == kind).collect();
            let (cells, want) = match kind {
                KernelKind::TopologyKeys | KernelKind::PermutationInfo => (tet_cells, sizes.len()),
                KernelKind::EVector => (tet_cells, types * sizes.len()),
                _ => {
                    let per_tile: usize = TILES
                        .iter()
                        .map(|&t| sizes.iter().filter(|&&v| v <= t).count())
                        .sum();
                    (mesh::factor_cells(1000), types * per_tile)
                }
            };
            assert_eq!(of.len(), want, "{kind:?}");
            for r in of {
                assert_eq!(
                    (r.n, r.nominal_n, r.times_s.len()),
                    (cells, 1000, 2),
                    "{r:?}"
                );
                assert!(r.vector_size_ran <= r.vector_size, "{r:?}");
                assert!(r.median_s > 0.0 && r.units > 0, "{r:?}");
                assert!(r.bandwidth_ratio.is_some_and(|b| b > 0.0), "{r:?}");
                match kind {
                    KernelKind::GeomFactors => {
                        let tile = r.tile.expect("GeomFactors records a tile");
                        assert!(r.vector_size <= tile, "{r:?}");
                    }
                    _ => assert_eq!(r.tile, None, "{r:?}"),
                }
            }
        }
    }
}
