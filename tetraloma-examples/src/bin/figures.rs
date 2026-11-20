//! Writes the book's figures (`tetraloma_examples::figures::svg_files`) to
//! `docs/user/src/figures/`, replacing every SVG file there.
//!
//! ```sh
//! cargo run -p tetraloma-examples --bin figures
//! ```

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let dir = PathBuf::from(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../docs/user/src/figures"
    ));
    std::fs::create_dir_all(&dir)?;
    for entry in std::fs::read_dir(&dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "svg") {
            std::fs::remove_file(&path)?;
        }
    }
    let files = tetraloma_examples::figures::svg_files();
    for (name, svg) in &files {
        std::fs::write(dir.join(name), svg)?;
    }
    println!("wrote {} figures to {}", files.len(), dir.display());
    Ok(())
}
