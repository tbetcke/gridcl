//! Bindings to ParMETIS under the feature `mpi` (ADR-0029, items 4 and 5); nothing without it.
//!
//! ParMETIS is found through `PARMETIS_DIR` (a prefix with `include/` and `lib/`), else in the
//! system paths (Debian and Ubuntu: `/usr/include`; `/usr/local`, `/opt/homebrew`). `metis.h`,
//! which `parmetis.h` includes, is found through `METIS_DIR`, else beside `parmetis.h` or in the
//! system paths. The MPI include paths come from `build-probe-mpi`, as for mpi-sys. bindgen binds
//! only the `ParMETIS_` functions, `idx_t`, `real_t` and the constants of `src/lib.rs`; the MPI
//! types are `mpi::ffi`'s.

fn main() {
    #[cfg(feature = "mpi")]
    bind::run();
}

#[cfg(feature = "mpi")]
mod bind {
    use std::env;
    use std::path::{Path, PathBuf};

    /// System include directories of `parmetis.h` and `metis.h`, each with the library
    /// directory to search (`None`: the linker's default paths).
    const SYSTEM_PATHS: [(&str, Option<&str>); 3] = [
        ("/usr/include", None),
        ("/usr/local/include", Some("/usr/local/lib")),
        ("/opt/homebrew/include", Some("/opt/homebrew/lib")),
    ];

    /// The include directory of `<header>` and the library directory, if not a default one,
    /// from the prefix in `<var>`, else from the system paths (`None` if neither has it).
    fn locate(var: &str, header: &str) -> Option<(PathBuf, Option<PathBuf>)> {
        if let Some(dir) = env::var_os(var) {
            let dir = PathBuf::from(dir);
            let include = dir.join("include");
            assert!(
                include.join(header).is_file(),
                "parmetis-sys: {var}={} has no include/{header}",
                dir.display()
            );
            let lib = ["lib", "lib64"]
                .iter()
                .map(|name| dir.join(name))
                .find(|lib| lib.is_dir())
                .unwrap_or_else(|| {
                    panic!(
                        "parmetis-sys: {var}={} has no lib/ or lib64/",
                        dir.display()
                    )
                });
            return Some((include, Some(lib)));
        }
        SYSTEM_PATHS
            .iter()
            .find(|(include, _)| Path::new(include).join(header).is_file())
            .map(|(include, lib)| (PathBuf::from(include), lib.map(PathBuf::from)))
    }

    pub fn run() {
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-env-changed=PARMETIS_DIR");
        println!("cargo:rerun-if-env-changed=METIS_DIR");

        let (include, lib) = locate("PARMETIS_DIR", "parmetis.h").unwrap_or_else(|| {
            panic!(
                "parmetis-sys: parmetis.h not found; set PARMETIS_DIR to the ParMETIS prefix \
                 (with include/ and lib/; scripts/hpc/build-partition-libs.sh builds one), or \
                 install it (Ubuntu: libparmetis-dev)"
            )
        });
        // metis.h: METIS_DIR, else beside parmetis.h (the build script's prefix), else the
        // system paths. Only its include directory matters (see the linking below).
        let metis = if env::var_os("METIS_DIR").is_none() && include.join("metis.h").is_file() {
            None
        } else {
            let (dir, _) = locate("METIS_DIR", "metis.h").unwrap_or_else(|| {
                panic!(
                    "parmetis-sys: metis.h not found; set METIS_DIR to the METIS prefix (with \
                     include/ and lib/), or install it (Ubuntu: libmetis-dev)"
                )
            });
            Some(dir)
        };
        let mpi = match build_probe_mpi::probe() {
            Ok(mpi) => mpi,
            Err(errors) => {
                for (i, error) in errors.iter().enumerate() {
                    println!("cargo:warning=MPI probe {}: {error}", i + 1);
                }
                panic!("parmetis-sys: no MPI library found (build-probe-mpi)");
            }
        };

        // The probed MPI's headers first, so its mpi.h wins over one beside parmetis.h.
        let mut builder = bindgen::builder();
        for dir in &mpi.include_paths {
            builder = builder.clang_arg(format!("-I{}", dir.display()));
        }
        builder = builder.clang_arg(format!("-I{}", include.display()));
        if let Some(dir) = &metis {
            builder = builder.clang_arg(format!("-I{}", dir.display()));
        }

        let bindings = builder
            .header_contents("parmetis_wrapper.h", "#include <parmetis.h>\n")
            // Only these items, not what they reach: MPI_Comm comes from src/lib.rs
            // (mpi::ffi's type).
            .allowlist_recursively(false)
            .allowlist_function("ParMETIS_.*")
            .allowlist_type("idx_t|real_t|rstatus_et")
            .allowlist_var(
                "PARMETIS_(MAJOR|MINOR|SUBMINOR)_VERSION|IDXTYPEWIDTH|REALTYPEWIDTH|METIS_OK|\
                 METIS_ERROR.*",
            )
            .prepend_enum_name(false)
            .default_macro_constant_type(bindgen::MacroTypeVariation::Signed)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("parmetis-sys: bindgen failed on parmetis.h");
        let out = PathBuf::from(env::var("OUT_DIR").expect("cargo sets OUT_DIR"));
        bindings
            .write_to_file(out.join("parmetis.rs"))
            .expect("parmetis-sys: cannot write the bindings");

        // mpi-sys links MPI itself. The shared libparmetis resolves METIS on its own: the build
        // script's contains it, Debian's depends on libmetis.
        if let Some(lib) = lib {
            println!("cargo:rustc-link-search=native={}", lib.display());
        }
        println!("cargo:rustc-link-lib=parmetis");
    }
}
