//! MPI counts and displacements: element counts as `i32`, checked against `i32::MAX`.

use crate::{GridError, Result};

/// The `i32` counts and displacements (exclusive prefix sums) of `counts`.
///
/// # Errors
///
/// `MessageTooLarge { count }` with the first count or displacement above `i32::MAX`.
#[cfg_attr(not(feature = "mpi"), allow(dead_code))]
pub(crate) fn checked_counts(counts: &[usize]) -> Result<(Vec<i32>, Vec<i32>)> {
    let fit = |count: usize| i32::try_from(count).map_err(|_| GridError::MessageTooLarge { count });
    let mut out = Vec::with_capacity(counts.len());
    let mut displs = Vec::with_capacity(counts.len());
    let mut at = 0_usize;
    for &c in counts {
        displs.push(fit(at)?);
        out.push(fit(c)?);
        // Above i32::MAX it fails as the next displacement, so it cannot overflow usize first
        // unless `c` itself is above i32::MAX, which `fit(c)` has rejected.
        at += c;
    }
    Ok((out, displs))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_displacements() {
        assert_eq!(
            checked_counts(&[3, 0, 2]).unwrap(),
            (vec![3, 0, 2], vec![0, 3, 3])
        );
        assert_eq!(checked_counts(&[]).unwrap(), (vec![], vec![]));
    }

    #[test]
    fn above_i32_max_is_message_too_large() {
        let max = i32::MAX as usize;
        assert!(checked_counts(&[max]).is_ok());
        assert!(matches!(
            checked_counts(&[max + 1]),
            Err(GridError::MessageTooLarge { count }) if count == max + 1
        ));
        // Each count fits; the displacement of the third does not.
        assert!(matches!(
            checked_counts(&[max, 1, 1]),
            Err(GridError::MessageTooLarge { count }) if count == max + 1
        ));
        assert!(checked_counts(&[max, 0]).is_ok());
    }
}
