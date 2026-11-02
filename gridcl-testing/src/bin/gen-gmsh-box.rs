//! `gen-gmsh-box N PATH`: writes `cube_box(Tetrahedron, N, 1)` (6N³ tetrahedra) as the binary
//! Gmsh file of the M8 memory test, with one volume and six side physical groups (ADR-0035,
//! rule 10; `gridcl_testing::io_parallel::write_box_msh`). Creates PATH's directory.

use std::path::PathBuf;
use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    let (Some(n), Some(path)) = (
        args.get(1).and_then(|n| n.parse::<usize>().ok()),
        args.get(2),
    ) else {
        eprintln!("usage: gen-gmsh-box N PATH");
        return ExitCode::from(2);
    };
    if n == 0 || args.len() != 3 {
        eprintln!("usage: gen-gmsh-box N PATH (N >= 1)");
        return ExitCode::from(2);
    }
    let path = PathBuf::from(path);
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
        && let Err(e) = std::fs::create_dir_all(dir)
    {
        eprintln!("gen-gmsh-box: {}: {e}", dir.display());
        return ExitCode::FAILURE;
    }
    match gridcl_testing::io_parallel::write_box_msh(n, &path) {
        Ok(()) => {
            println!(
                "gen-gmsh-box: {}: {} tetrahedra",
                path.display(),
                6 * n * n * n
            );
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gen-gmsh-box: {}: {e}", path.display());
            ExitCode::FAILURE
        }
    }
}
