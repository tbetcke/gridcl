//! VTKHDF output (docs/design.md, "VTK", VTKHDF; ADR-0035, rule 8): one `UnstructuredGrid` file
//! with one piece per rank.

use std::path::Path;

use gridcl_device::Result;
use gridcl_parallel::CommDyn;

use super::VtuFile;

/// Writes this rank's `piece` into the VTKHDF file at `path`, collectively over `comm`: every
/// rank calls it, pieces in rank order.
///
/// # Errors
///
/// `Io` (agreed by every rank); `NotYetImplemented("compressed VTKHDF")` is for the facade.
pub fn write_vtkhdf(path: &Path, piece: &VtuFile, comm: &dyn CommDyn) -> Result<()> {
    let _ = (path, piece, comm);
    todo!("write_vtkhdf (M8 T9)")
}
