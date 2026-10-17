use std::process::ExitCode;

use env_check::{Options, USAGE};

fn main() -> ExitCode {
    let options = match Options::parse(std::env::args().skip(1)) {
        Ok(options) => options,
        Err(message) => {
            eprintln!("env-check: {message}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    let _ = env_check::run(&options);
    todo!()
}
