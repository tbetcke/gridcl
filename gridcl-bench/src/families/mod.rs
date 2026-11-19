//! The vector-size sweep: every family, scalar type, size and configuration of "Benchmarks",
//! measured with the protocol of [`crate::protocol`].
//!
//! - [`device`]: the device-primitive families (Fill, Copy, Reduce, ScanTile, SortDigits,
//!   Gather) and the scalar helpers measured for `KernelKind::Scalar`'s CPU blocks;
//! - [`mesh`]: the mesh families (TopologyKeys, PermutationInfo, EVector, GeomFactors), with
//!   cells as items;
//! - [`degree`]: the M9h degree sweep of the degree-dependent families at p = 1–8, with
//!   quadrature points as items.

pub mod degree;
pub mod device;
pub mod mesh;

use std::collections::BTreeMap;

use cubecl::prelude::Client;
use gridcl_device::{CpuUnits, DeviceClass, KernelKind, check_precision};

use crate::candidates::{VectorConfig, candidate_kinds, cpu_grid, grid_omitted, vector_candidates};
use crate::protocol::{
    self, Config, Measured, Workload, bandwidth_ratio, is_cpu, measure, timed_run, with_config,
};
use crate::record::{Cores, CubeDimMode, LegacyClass, Scalar, SweepRecord};

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
    /// Kernel families to measure.
    pub families: Vec<KernelKind>,
    /// Scalar types to keep; `None` measures every type of each family.
    pub scalars: Option<Vec<Scalar>>,
    /// Problem sizes.
    pub sizes: Vec<usize>,
    /// Vector sizes to keep; `None` measures every size the device offers.
    pub vector_sizes: Option<Vec<usize>>,
    /// CPU block targets in bytes (CPU devices only).
    pub blocks: Vec<usize>,
    /// The ways of choosing the CPU `cube_dim` to measure.
    pub cube_dims: Vec<CubeDimMode>,
    /// Timed runs per configuration.
    pub runs: usize,
    /// Why a fallback target ran in place of the usual one (ADR-0014, ADR-0016); every record
    /// carries it.
    pub fallback_reason: Option<String>,
}

/// What every record of a run shares.
#[derive(Clone, Debug)]
pub struct Machine {
    /// CubeCL runtime as `GRIDCL_TEST_DEVICE` names it.
    pub backend: String,
    /// Device name as the client reports it.
    pub device: String,
    /// Device class of the client.
    pub class: DeviceClass,
    /// The commit the benchmark ran from.
    pub commit: String,
    /// Core counts of the machine.
    pub cores: Cores,
    /// State of CubeCL's persistent kernel cache: `on` or `off`.
    pub kernel_cache: String,
    /// `hardware.load_width` of the client in bits (ADR-0058 item 1, ADR-0060 item 2).
    pub load_width: u32,
}

impl Machine {
    /// The shared fields of a run on `client`: its backend, device and class, the commit, the
    /// core counts and the kernel-cache state.
    pub fn new(client: &Client) -> Machine {
        Machine {
            backend: protocol::backend(),
            device: client.properties().identity.name.clone(),
            class: DeviceClass::for_client(client),
            commit: protocol::commit(),
            cores: protocol::cores(),
            kernel_cache: protocol::kernel_cache().to_owned(),
            load_width: client.properties().hardware.load_width,
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
                                (&options.target, &options.fallback_reason),
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

/// What the candidate sweep runs (ADR-0057 item 3, ADR-0060 item 2).
#[derive(Clone, Debug)]
pub struct CandidateOptions {
    /// Machine: `macmini`, or a target of scripts/hpc/remote-test.sh.
    pub target: String,
    /// Kinds to measure, of [`candidate_kinds`].
    pub families: Vec<KernelKind>,
    /// Scalar types to keep; `None` measures every type of each kind.
    pub scalars: Option<Vec<Scalar>>,
    /// Problem sizes; the protocol's are [`protocol::CANDIDATE_SIZES`].
    pub sizes: Vec<usize>,
    /// Timed runs per configuration.
    pub runs: usize,
    /// Why a fallback target ran in place of the usual one (ADR-0014, ADR-0016); every record
    /// carries it.
    pub fallback_reason: Option<String>,
}

/// The device's widest vector size for `kind` in `scalar`: its widest IO size for the element
/// size the kind plans with (u32 words for Pack), at most `VECTOR_PAD`.
pub fn widest_vector_size(client: &Client, kind: KernelKind, scalar: Scalar) -> usize {
    let bytes = if kind == KernelKind::Pack {
        size_of::<u32>()
    } else {
        scalar.bytes()
    };
    client
        .properties()
        .io_optimized_vector_sizes(bytes)
        .next()
        .unwrap_or(1)
        .min(gridcl_device::VECTOR_PAD)
}

/// The protocol configurations of `candidates` on `machine`: a performance-core candidate on a
/// device without a performance-core count runs on all threads, which give its time
/// (ADR-0017); each configuration once, in the order of `candidates`.
pub fn candidate_configs(machine: &Machine, candidates: &[VectorConfig]) -> Vec<Config> {
    let mut out: Vec<Config> = Vec::new();
    for c in candidates {
        let performance =
            c.units == Some(CpuUnits::Performance) && machine.cores.performance.is_some();
        let config = Config {
            vector_size: c.vector_size,
            block_bytes: c.block_bytes,
            tile: c.tile,
            cube_dim: if performance {
                CubeDimMode::PerformanceCores
            } else {
                CubeDimMode::Plan
            },
        };
        if !out.contains(&config) {
            out.push(config);
        }
    }
    out
}

/// Runs the candidate protocol (ADR-0057 item 3): for every kind, scalar type and size of
/// `options`, the candidates of `candidates::vector_candidates` for the machine's class, and
/// passes each record to `on_record` as soon as it is measured. Inputs, the bandwidth
/// reference and the check are those of [`sweep`]; Scalar runs on CPUs only.
///
/// # Errors
///
/// The first failed measurement, naming its configuration.
pub fn candidate_sweep(
    client: &Client,
    machine: &Machine,
    options: &CandidateOptions,
    on_record: &mut dyn FnMut(&SweepRecord),
) -> Result<Vec<SweepRecord>, String> {
    let class = machine.class;
    let has_f64 = check_precision::<f64>(client).is_ok();
    let mut copy_reference: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    let mut records = Vec::new();
    for (kind, scalars) in candidate_kinds() {
        if !options.families.contains(&kind) || (kind == KernelKind::Scalar && !is_cpu(class)) {
            continue;
        }
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
            let widest = widest_vector_size(client, kind, scalar);
            let configs =
                candidate_configs(machine, &vector_candidates(class, kind, scalar, widest));
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
                for config in &configs {
                    let what = format!("{kind:?} {} n = {n} {config:?}", scalar.name());
                    let measured = with_config(kind, config, machine.cores.performance, || {
                        measure(client, kind, options.runs, work.as_mut())
                    })
                    .and_then(|m| m)
                    .map_err(|e| format!("{what}: {e}"))?;
                    let record = sweep_record(
                        machine,
                        (&options.target, &options.fallback_reason),
                        (kind, scalar, n),
                        work.as_ref(),
                        config,
                        &measured,
                        reference,
                    );
                    on_record(&record);
                    records.push(record);
                }
            }
        }
    }
    Ok(records)
}

/// What the M10c grid sweep runs (ADR-0065 items 1, 2 and 4).
#[derive(Clone, Debug)]
pub struct GridOptions {
    /// Machine: `macmini`, `locust`, `kathleen` (or `myriad-cpu` as Kathleen's fallback).
    pub target: String,
    /// Kinds to measure, of [`candidate_kinds`].
    pub families: Vec<KernelKind>,
    /// Scalar types to keep; `None` measures every type of each kind.
    pub scalars: Option<Vec<Scalar>>,
    /// Problem sizes; the protocol's are [`protocol::CANDIDATE_SIZES`].
    pub sizes: Vec<usize>,
    /// Interleaved rounds, one timed run of every slot each.
    pub rounds: usize,
    /// The process's seed, from which each entry's round orders are shuffled.
    pub seed: u64,
    /// Why a fallback target ran in place of the usual one (ADR-0016); every record carries it.
    pub fallback_reason: Option<String>,
}

/// The `[Cpu]` vector entries of the grid, in the order of `vector-policy.toml`: each candidate
/// kind with an element size and its member scalars of that size (13 four-byte and 8 eight-byte
/// entries). The index of an entry in this list seeds its shuffle.
pub fn grid_entries() -> Vec<(KernelKind, usize, Vec<Scalar>)> {
    let mut out = Vec::new();
    for (kind, scalars) in candidate_kinds() {
        for bytes in [4, 8] {
            let members: Vec<Scalar> = scalars
                .iter()
                .copied()
                .filter(|s| s.bytes() == bytes)
                .collect();
            if !members.is_empty() {
                out.push((kind, bytes, members));
            }
        }
    }
    out
}

/// splitmix64 (Steele, Lea and Flood 2014): the generator of the grid's shuffles.
#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    /// The generator of entry `entry` in the process of `seed`.
    pub fn for_entry(seed: u64, entry: usize) -> SplitMix64 {
        SplitMix64(seed.wrapping_mul(1_000_003).wrapping_add(entry as u64))
    }

    /// The next 64 random bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// Shuffles `v` in place (Fisher-Yates).
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        for i in (1..v.len()).rev() {
            let j = (self.next_u64() % (i as u64 + 1)) as usize;
            v.swap(i, j);
        }
    }
}

/// The orders of `rounds` rounds over `slots` slots of entry `entry` in the process of `seed`:
/// round k times the slots in the order `orders[k]`, a permutation of `0..slots`.
pub fn round_orders(seed: u64, entry: usize, slots: usize, rounds: usize) -> Vec<Vec<usize>> {
    let mut rng = SplitMix64::for_entry(seed, entry);
    (0..rounds)
        .map(|_| {
            let mut order: Vec<usize> = (0..slots).collect();
            rng.shuffle(&mut order);
            order
        })
        .collect()
}

/// One slot of an entry: a (member scalar, size, configuration) with its calibration.
struct Slot {
    scalar: Scalar,
    size: usize,
    work: usize,
    config: Config,
    measured: Measured,
    positions: Vec<usize>,
}

/// Runs the M10c grid (ADR-0065 items 1, 2 and 4) on a CPU device and passes each record to
/// `on_record` once its entry is measured.
///
/// Per `[Cpu]` vector entry of [`grid_entries`] (kind and element size): every member scalar,
/// size and configuration of [`cpu_grid`] (a performance-core configuration on a device without
/// a performance-core count runs on all threads and counts for both, ADR-0017; the cases of
/// [`crate::candidates::GRID_OMITTED`] for the target left out) is one slot. Each slot is
/// warmed up, checked and calibrated (the protocol's [`measure`] with zero runs); then come
/// `rounds` rounds, each one batched run of every slot in an order shuffled from the seed and
/// the entry's index ([`round_orders`]). A record's `positions` give its slot's place in each
/// round. The inputs of every member and size of an entry stay alive until its rounds end. The
/// bandwidth reference is a u32 copy of each size, measured once per process as in [`sweep`].
///
/// # Errors
///
/// A GPU device, or the first failed measurement, naming its configuration.
pub fn grid_sweep(
    client: &Client,
    machine: &Machine,
    options: &GridOptions,
    on_record: &mut dyn FnMut(&SweepRecord),
) -> Result<Vec<SweepRecord>, String> {
    if !is_cpu(machine.class) {
        return Err(format!(
            "the grid is for `[Cpu]` entries; {} is a {:?} device",
            machine.device, machine.class
        ));
    }
    let has_f64 = check_precision::<f64>(client).is_ok();
    let mut copy_reference: BTreeMap<usize, (usize, f64)> = BTreeMap::new();
    let mut records = Vec::new();
    for (entry, (kind, bytes, members)) in grid_entries().into_iter().enumerate() {
        if !options.families.contains(&kind) {
            continue;
        }
        let members: Vec<Scalar> = members
            .into_iter()
            .filter(|s| options.scalars.as_ref().is_none_or(|keep| keep.contains(s)))
            .filter(|&s| {
                let keep = s != Scalar::F64 || has_f64;
                if !keep {
                    eprintln!("vector_sweep: {kind:?} f64 skipped: the device has no f64");
                }
                keep
            })
            .collect();
        if members.is_empty() {
            continue;
        }
        let mut works: Vec<Box<dyn Workload>> = Vec::new();
        let mut slots: Vec<Slot> = Vec::new();
        for &scalar in &members {
            let configs = candidate_configs(machine, &cpu_grid(kind, scalar));
            for &size in &options.sizes {
                if grid_omitted(&options.target, kind, scalar, size) {
                    eprintln!(
                        "vector_sweep: {kind:?} {} at {size} left out on {} (GRID_OMITTED)",
                        scalar.name(),
                        options.target
                    );
                    continue;
                }
                if let std::collections::btree_map::Entry::Vacant(e) = copy_reference.entry(size) {
                    e.insert(measure_copy_reference(
                        client,
                        size,
                        protocol::CANDIDATE_RUNS,
                    )?);
                }
                let mut work = workload(client, kind, scalar, size)?;
                for config in &configs {
                    let what = format!("{kind:?} {} n = {size} {config:?}", scalar.name());
                    let measured = with_config(kind, config, machine.cores.performance, || {
                        measure(client, kind, 0, work.as_mut())
                    })
                    .and_then(|m| m)
                    .map_err(|e| format!("{what}: {e}"))?;
                    slots.push(Slot {
                        scalar,
                        size,
                        work: works.len(),
                        config: *config,
                        measured,
                        positions: Vec::with_capacity(options.rounds),
                    });
                }
                works.push(work);
            }
        }
        for order in round_orders(options.seed, entry, slots.len(), options.rounds) {
            for (position, &i) in order.iter().enumerate() {
                let slot = &mut slots[i];
                let work = works[slot.work].as_mut();
                let calls = slot.measured.calls_per_run;
                let time = with_config(kind, &slot.config, machine.cores.performance, || {
                    timed_run(client, work, calls)
                })
                .and_then(|t| t)
                .map_err(|e| {
                    format!(
                        "{kind:?} {} n = {} {:?}: {e}",
                        slot.scalar.name(),
                        slot.size,
                        slot.config
                    )
                })?;
                slot.measured.times_s.push(time);
                slot.positions.push(position);
            }
        }
        eprintln!(
            "vector_sweep: {kind:?} {bytes}-byte: {} slots, {} rounds",
            slots.len(),
            options.rounds
        );
        for slot in slots {
            let mut record = sweep_record(
                machine,
                (&options.target, &options.fallback_reason),
                (kind, slot.scalar, slot.size),
                works[slot.work].as_ref(),
                &slot.config,
                &slot.measured,
                copy_reference[&slot.size],
            );
            record.seed = Some(options.seed);
            record.positions = Some(slot.positions);
            on_record(&record);
            records.push(record);
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
    (target, fallback_reason): (&str, &Option<String>),
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
        legacy_class: LegacyClass::of_new(machine.class),
        target: target.to_owned(),
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
        load_width: Some(machine.load_width),
        fallback_reason: fallback_reason.clone(),
        node_load: None,
        seed: None,
        positions: None,
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

    /// The candidate protocol at 2000 items, two runs, every kind: one record per kind, type and
    /// protocol configuration of its candidates, each measuring its configuration; Pack and
    /// PullBack pass their checks; Scalar runs on CPUs only.
    #[test]
    fn candidate_smoke_sweep_covers_every_kind() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        assert_eq!(machine.load_width, client.properties().hardware.load_width);
        let options = CandidateOptions {
            target: "test".to_owned(),
            families: candidate_kinds().into_iter().map(|(k, _)| k).collect(),
            scalars: None,
            sizes: vec![2000],
            runs: 2,
            fallback_reason: None,
        };
        let records = candidate_sweep(&client, &machine, &options, &mut |_| ()).unwrap();
        let has_f64 = check_precision::<f64>(&client).is_ok();
        for (kind, scalars) in candidate_kinds() {
            for &scalar in scalars {
                let of: Vec<&SweepRecord> = records
                    .iter()
                    .filter(|r| r.family == kind && r.scalar == scalar)
                    .collect();
                if (scalar == Scalar::F64 && !has_f64)
                    || (kind == KernelKind::Scalar && !is_cpu(machine.class))
                {
                    assert!(of.is_empty(), "{kind:?} {}", scalar.name());
                    continue;
                }
                let widest = widest_vector_size(&client, kind, scalar);
                let configs = candidate_configs(
                    &machine,
                    &vector_candidates(machine.class, kind, scalar, widest),
                );
                assert_eq!(of.len(), configs.len(), "{kind:?} {}", scalar.name());
                for (r, c) in of.iter().zip(&configs) {
                    assert_eq!(
                        (r.vector_size, r.tile, r.block_bytes, r.cube_dim),
                        (c.vector_size, c.tile, c.block_bytes, c.cube_dim),
                        "{kind:?} {}",
                        scalar.name()
                    );
                    assert_eq!((r.nominal_n, r.times_s.len()), (2000, 2));
                    assert_eq!(r.load_width, Some(machine.load_width));
                    assert_eq!(r.device_class, machine.class);
                    assert!(r.vector_size_ran <= r.vector_size, "{r:?}");
                }
            }
        }
        let pull_back = records
            .iter()
            .find(|r| r.family == KernelKind::PullBack)
            .unwrap();
        assert!(pull_back.n >= 1000, "points as items: {}", pull_back.n);
    }

    /// The grid's shuffles are deterministic per seed and entry, and each round is one
    /// permutation of the slots.
    #[test]
    fn round_orders_are_seeded_permutations() {
        let a = round_orders(1, 3, 48, 5);
        assert_eq!(a, round_orders(1, 3, 48, 5));
        assert_ne!(a, round_orders(2, 3, 48, 5));
        assert_ne!(a, round_orders(1, 4, 48, 5));
        assert_eq!(a.len(), 5);
        for order in &a {
            let mut sorted = order.clone();
            sorted.sort_unstable();
            assert_eq!(sorted, (0..48).collect::<Vec<_>>());
        }
        assert_ne!(a[0], a[1], "rounds differ");
        assert!(round_orders(1, 0, 0, 2).iter().all(Vec::is_empty));
    }

    /// 13 four-byte and 8 eight-byte entries, each with its member scalars of that size.
    #[test]
    fn grid_entries_are_the_typed_cpu_entries() {
        let e = grid_entries();
        assert_eq!(e.iter().filter(|(_, b, _)| *b == 4).count(), 13);
        assert_eq!(e.iter().filter(|(_, b, _)| *b == 8).count(), 8);
        for (kind, bytes, members) in &e {
            assert!(!members.is_empty());
            assert!(members.iter().all(|s| s.bytes() == *bytes), "{kind:?}");
        }
        assert!(e.contains(&(KernelKind::Fill, 4, vec![Scalar::F32, Scalar::U32])));
        assert!(e.contains(&(KernelKind::Scalar, 8, vec![Scalar::F64])));
    }

    /// The grid at 10^3 items, two rounds, three kinds (one with two members, GeomFactors and
    /// Scalar): every slot gives a record with the process's seed and its place in each round,
    /// each round is a permutation of the entry's slots, and both unit settings run where the
    /// device reports performance cores.
    #[test]
    fn grid_smoke_sweep_interleaves() {
        let client = gridcl_testing::test_client();
        let machine = Machine::new(&client);
        let options = GridOptions {
            target: "test".to_owned(),
            families: vec![
                KernelKind::Fill,
                KernelKind::GeomFactors,
                KernelKind::Scalar,
            ],
            scalars: Some(vec![Scalar::F32, Scalar::U32]),
            sizes: vec![1000],
            rounds: 2,
            seed: 7,
            fallback_reason: None,
        };
        if !is_cpu(machine.class) {
            assert!(grid_sweep(&client, &machine, &options, &mut |_| ()).is_err());
            return;
        }
        let mut seen = 0;
        let records = grid_sweep(&client, &machine, &options, &mut |_| seen += 1).unwrap();
        assert_eq!(seen, records.len());
        let performance = machine.cores.performance.is_some();
        let per = |n: usize| if performance { n } else { n / 2 };
        for (kind, members, configs) in [
            (KernelKind::Fill, 2, 24),
            (KernelKind::GeomFactors, 1, 24),
            (KernelKind::Scalar, 1, 6),
        ] {
            let of: Vec<&SweepRecord> = records.iter().filter(|r| r.family == kind).collect();
            assert_eq!(of.len(), members * per(configs), "{kind:?}");
            for round in 0..2 {
                let mut at: Vec<usize> = of
                    .iter()
                    .map(|r| r.positions.as_ref().unwrap()[round])
                    .collect();
                at.sort_unstable();
                assert_eq!(
                    at,
                    (0..of.len()).collect::<Vec<_>>(),
                    "{kind:?} round {round}"
                );
            }
            let modes: std::collections::BTreeSet<_> =
                of.iter().map(|r| format!("{:?}", r.cube_dim)).collect();
            assert_eq!(
                modes.len(),
                if performance { 2 } else { 1 },
                "{kind:?}: {modes:?}"
            );
        }
        for r in &records {
            assert_eq!(r.seed, Some(7));
            assert_eq!(r.times_s.len(), 2, "{r:?}");
            assert_eq!(r.positions.as_ref().map(Vec::len), Some(2), "{r:?}");
            assert_eq!(r.nominal_n, 1000);
            assert!(r.vector_size_ran <= r.vector_size, "{r:?}");
            assert!(
                r.median_s > 0.0 && r.units > 0 && r.calls_per_run > 0,
                "{r:?}"
            );
            assert!(r.block_bytes.is_some(), "{r:?}");
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
