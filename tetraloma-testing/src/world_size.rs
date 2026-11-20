//! The rank-count guard of the MPI runs (docs/design.md, "Testing and validation"; ADR-0068).
//!
//! `scripts/mpi-test.sh`, `scripts/run-examples.sh` and `scripts/bench/halo-exchange.sh` export
//! the rank count they request as [`EXPECT_RANKS_VAR`] to every rank. `tetraloma_mpi_tests::run`,
//! the MPI examples and the halo-exchange benchmark compare it with their world size before
//! anything else, so that a launcher that starts each process as its own world of one rank
//! (Ubuntu 24.04's MPICH, Launchpad bug 2072338) fails at once instead of passing on one rank.

/// The environment variable that holds the rank count the launching script requested.
pub const EXPECT_RANKS_VAR: &str = "TETRALOMA_EXPECT_RANKS";

/// Checks the world size `size` against [`EXPECT_RANKS_VAR`]; an unset variable means no check
/// (direct runs and users' runs).
///
/// # Errors
///
/// The message of [`check_world_size_against`].
///
/// # Example
///
/// ```
/// // Without TETRALOMA_EXPECT_RANKS any world size passes.
/// # if std::env::var_os(tetraloma_testing::EXPECT_RANKS_VAR).is_none() {
/// assert!(tetraloma_testing::check_world_size(3).is_ok());
/// # }
/// ```
pub fn check_world_size(size: usize) -> Result<(), String> {
    let _ = size;
    todo!()
}

/// Checks the world size `size` against `expected`, the value of [`EXPECT_RANKS_VAR`]
/// (`None`: unset, no check).
///
/// # Errors
///
/// A message that names the world size, the expected count and [`EXPECT_RANKS_VAR`], if
/// `expected` is not a positive integer or differs from `size`.
///
/// # Example
///
/// ```
/// use tetraloma_testing::check_world_size_against;
///
/// assert!(check_world_size_against(2, Some("2")).is_ok());
/// assert!(check_world_size_against(1, None).is_ok());
/// let message = check_world_size_against(1, Some("2")).unwrap_err();
/// assert!(message.contains("world size 1") && message.contains("TETRALOMA_EXPECT_RANKS=2"));
/// ```
pub fn check_world_size_against(size: usize, expected: Option<&str>) -> Result<(), String> {
    let _ = (size, expected);
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The message names the world size, the expected value and the variable.
    fn assert_names(message: &str, size: usize, expected: &str) {
        for part in [
            format!("world size {size}"),
            format!("{EXPECT_RANKS_VAR}={expected}"),
        ] {
            assert!(message.contains(&part), "{message:?} lacks {part:?}");
        }
    }

    #[test]
    fn unset_is_no_check() {
        for size in [1, 2, 7] {
            assert_eq!(check_world_size_against(size, None), Ok(()));
        }
    }

    #[test]
    fn equal_passes() {
        for size in [1, 2, 3, 4, 16] {
            assert_eq!(
                check_world_size_against(size, Some(&size.to_string())),
                Ok(())
            );
        }
    }

    #[test]
    fn singleton_fails() {
        for expected in ["2", "3", "4"] {
            let message = check_world_size_against(1, Some(expected)).unwrap_err();
            assert_names(&message, 1, expected);
        }
    }

    #[test]
    fn other_mismatch_fails() {
        for (size, expected) in [(3, "4"), (4, "2"), (2, "16")] {
            let message = check_world_size_against(size, Some(expected)).unwrap_err();
            assert_names(&message, size, expected);
        }
    }

    #[test]
    fn invalid_value_fails() {
        for expected in ["0", "-1", "two", "", " 2"] {
            let message = check_world_size_against(2, Some(expected)).unwrap_err();
            assert_names(&message, 2, expected);
            assert!(message.contains("not a positive integer"), "{message:?}");
        }
    }
}
