//! The host-staged halo exchange, part by part (M10 T6; ADR-0059): pack, device-to-host copy,
//! host neighbourhood all-to-all-v, host-to-device copy and unpack, and the whole `forward` and
//! `reverse` (`Add`), for halos of 10^3, 10^4 and 10^5 nodes with block 3 and 10^6 owned nodes
//! per rank; the median of 50 timed runs after 5 warm-ups. Writes `m10-halo-*.json` records.

fn main() {
    todo!("M10 T6")
}
