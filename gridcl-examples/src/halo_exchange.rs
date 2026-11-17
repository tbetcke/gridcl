//! `forward` and `reverse` on a mesh's node index map (Listing B13).

use cubecl::prelude::Client;
use gridcl::Comm;

/// Runs the halo exchange of a distributed mesh on `comm`: one value per node and component
/// (block 3), owner to ghosts and back with `Add`.
pub fn run<C: Comm>(_client: &Client, _comm: C) -> gridcl::Result<()> {
    todo!("M10 T11")
}
