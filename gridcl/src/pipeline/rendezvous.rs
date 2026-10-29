//! Variable-length i64 records between ranks, and the rendezvous rank of a key (ADR-0026,
//! item 9).

use gridcl_device::Result;
use gridcl_parallel::{CommDyn, CommExt, owner_hash};

/// The rendezvous rank of `key` on `size` ranks: `owner_hash(key) mod size`.
pub(crate) fn rendezvous_rank(key: &[i64], size: usize) -> usize {
    (owner_hash(key) % size as u64) as usize
}

/// Records to send, one list per destination rank. A record is a length word followed by
/// its words.
pub(crate) struct Outbox {
    bufs: Vec<Vec<i64>>,
}

impl Outbox {
    /// An empty outbox for `size` ranks.
    pub(crate) fn new(size: usize) -> Outbox {
        Outbox {
            bufs: vec![Vec::new(); size],
        }
    }

    /// Appends the record `parts` (concatenated) for rank `dest`.
    pub(crate) fn push(&mut self, dest: usize, parts: &[&[i64]]) {
        let buf = &mut self.bufs[dest];
        buf.push(parts.iter().map(|p| p.len()).sum::<usize>() as i64);
        for p in parts {
            buf.extend_from_slice(p);
        }
    }

    /// Sends every record to its rank and returns the records received. Collective.
    ///
    /// # Errors
    ///
    /// Errors of `CommExt::all_to_all_v`.
    pub(crate) fn exchange(self, comm: &dyn CommDyn) -> Result<Inbox> {
        let counts: Vec<usize> = self.bufs.iter().map(Vec::len).collect();
        let (data, counts) = comm.all_to_all_v(&self.bufs.concat(), &counts)?;
        Ok(Inbox { data, counts })
    }
}

/// The records received by [`Outbox::exchange`].
pub(crate) struct Inbox {
    data: Vec<i64>,
    counts: Vec<usize>,
}

impl Inbox {
    /// Every record with its source rank, by source rank and then in sending order.
    pub(crate) fn records(&self) -> impl Iterator<Item = (usize, &[i64])> {
        let mut source = 0;
        let mut end = self.counts.first().copied().unwrap_or(0);
        let mut at = 0;
        std::iter::from_fn(move || {
            while at == end {
                source += 1;
                end += *self.counts.get(source)?;
            }
            let len = self.data[at] as usize;
            let record = &self.data[at + 1..at + 1 + len];
            at += 1 + len;
            Some((source, record))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gridcl_testing::ThreadComm;

    #[test]
    fn records_arrive_by_source_in_order() {
        let got = ThreadComm::run(3, |comm| {
            let r = comm.rank() as i64;
            let mut out = Outbox::new(3);
            // Rank r sends r + 1 records to rank (r + 1) mod 3, one empty, and one to itself.
            for k in 0..=r {
                out.push(((r + 1) % 3) as usize, &[&[r, k], &[]]);
            }
            out.push(((r + 1) % 3) as usize, &[]);
            out.push(r as usize, &[&[100 + r]]);
            let inbox = out.exchange(&comm).unwrap();
            inbox
                .records()
                .map(|(s, rec)| (s, rec.to_vec()))
                .collect::<Vec<_>>()
        });
        assert_eq!(
            got[0],
            [
                (0, vec![100]),
                (2, vec![2, 0]),
                (2, vec![2, 1]),
                (2, vec![2, 2]),
                (2, vec![])
            ]
        );
        assert_eq!(got[1], [(0, vec![0, 0]), (0, vec![]), (1, vec![101])]);
        assert_eq!(
            got[2],
            [
                (1, vec![1, 0]),
                (1, vec![1, 1]),
                (1, vec![]),
                (2, vec![102])
            ]
        );
    }

    #[test]
    fn rendezvous_ranks_follow_owner_hash() {
        assert_eq!(rendezvous_rank(&[42], 4), 3);
        assert_eq!(rendezvous_rank(&[1, 2, 3], 1), 0);
    }
}
