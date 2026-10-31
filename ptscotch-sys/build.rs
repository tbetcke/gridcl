//! Bindings to PT-Scotch under the feature `mpi` (ADR-0029, items 4 and 5); nothing without it.
//!
//! PT-Scotch is found through `SCOTCH_DIR` (a prefix with `include/` and `lib/`), else in the
//! system paths (Debian and Ubuntu: `/usr/include/scotch`; Homebrew: `/opt/homebrew`,
//! `/usr/local`). The MPI include paths come from `build-probe-mpi`, as for mpi-sys. bindgen
//! binds only the `SCOTCH_` items; the MPI types are `mpi::ffi`'s (`src/lib.rs`).

fn main() {
    #[cfg(feature = "mpi")]
    bind::run();
}

#[cfg(feature = "mpi")]
mod bind {
    use std::env;
    use std::path::{Path, PathBuf};

    use bindgen::callbacks::{IntKind, ParseCallbacks};

    /// System locations of `ptscotch.h`, each with the library directory to search (`None`:
    /// the linker's default paths).
    const SYSTEM_PATHS: [(&str, Option<&str>); 5] = [
        ("/usr/include/scotch", None),
        ("/usr/include", None),
        ("/usr/local/include/scotch", Some("/usr/local/lib")),
        ("/usr/local/include", Some("/usr/local/lib")),
        ("/opt/homebrew/include", Some("/opt/homebrew/lib")),
    ];

    /// Types the strategy flags `SCOTCH_STRAT*` as `SCOTCH_Num`, the type of the `flagval`
    /// arguments of the `SCOTCH_strat*Build` functions; other integer macros are C `int`.
    #[derive(Debug)]
    struct StratFlags;

    impl ParseCallbacks for StratFlags {
        fn int_macro(&self, name: &str, _value: i64) -> Option<IntKind> {
            name.starts_with("SCOTCH_STRAT").then_some(IntKind::Custom {
                name: "SCOTCH_Num",
                is_signed: true,
            })
        }
    }

    /// The include directory of `ptscotch.h` and the library directory, if not a default one.
    fn locate() -> (PathBuf, Option<PathBuf>) {
        if let Some(dir) = env::var_os("SCOTCH_DIR") {
            let dir = PathBuf::from(dir);
            let include = dir.join("include");
            assert!(
                include.join("ptscotch.h").is_file(),
                "ptscotch-sys: SCOTCH_DIR={} has no include/ptscotch.h",
                dir.display()
            );
            let lib = ["lib", "lib64"]
                .iter()
                .map(|name| dir.join(name))
                .find(|lib| lib.is_dir())
                .unwrap_or_else(|| {
                    panic!(
                        "ptscotch-sys: SCOTCH_DIR={} has no lib/ or lib64/",
                        dir.display()
                    )
                });
            return (include, Some(lib));
        }
        SYSTEM_PATHS
            .iter()
            .find(|(include, _)| Path::new(include).join("ptscotch.h").is_file())
            .map(|(include, lib)| (PathBuf::from(include), lib.map(PathBuf::from)))
            .unwrap_or_else(|| {
                panic!(
                    "ptscotch-sys: ptscotch.h not found; set SCOTCH_DIR to the PT-Scotch prefix \
                     (with include/ and lib/), or install it (Ubuntu: libptscotch-dev; \
                     Homebrew: scotch)"
                )
            })
    }

    pub fn run() {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=SCOTCH_DIR");

        let (include, lib) = locate();
        let mpi = match build_probe_mpi::probe() {
            Ok(mpi) => mpi,
            Err(errors) => {
                for (i, error) in errors.iter().enumerate() {
                    println!("cargo:warning=MPI probe {}: {error}", i + 1);
                }
                panic!("ptscotch-sys: no MPI library found (build-probe-mpi)");
            }
        };

        // The probed MPI's headers first, so its mpi.h wins over one beside ptscotch.h.
        let mut builder = bindgen::builder();
        for dir in &mpi.include_paths {
            builder = builder.clang_arg(format!("-I{}", dir.display()));
        }
        builder = builder.clang_arg(format!("-I{}", include.display()));

        // The PT-Scotch manual's include order: stdio.h and mpi.h before ptscotch.h, which
        // includes scotch.h itself.
        let bindings = builder
            .header_contents(
                "ptscotch_wrapper.h",
                "#include <stdio.h>\n#include <mpi.h>\n#include <ptscotch.h>\n",
            )
            // Only the SCOTCH_ items, not what they reach: MPI_Comm, MPI_Datatype and FILE
            // come from src/lib.rs (mpi::ffi's types; FILE opaque).
            .allowlist_recursively(false)
            .allowlist_function("SCOTCH_(dgraph|strat|random|context|version).*")
            .allowlist_type("SCOTCH_.*")
            .allowlist_var("SCOTCH_.*")
            .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
            .parse_callbacks(Box::new(StratFlags))
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("ptscotch-sys: bindgen failed on ptscotch.h");
        let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
        bindings
            .write_to_file(out.join("ptscotch.rs"))
            .expect("ptscotch-sys: cannot write the bindings");

        // mpi-sys links MPI itself. Shared libraries resolve in any order; for static ones
        // ptscotch needs scotch, and both need ptscotcherr's error handler.
        if let Some(lib) = lib {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
        for name in ["ptscotch", "scotch", "ptscotcherr"] {
            println!("cargo:rustc-link-lib={name}");
        }
    }
}
