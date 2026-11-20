# Documents

`docs/` holds two kinds of documents (ADR-0067, rule 1).

## For users

| Path | Contents |
| --- | --- |
| [`user/`](user/) | The user book (mdBook): installation, mathematical concepts, element definitions, tutorials, devices, features. Published with the API documentation of the nine library crates at https://tetracl.github.io/tetraloma/. Build it with `mdbook build docs/user`; `scripts/check-user-docs.sh` builds and checks the whole site ([`dev-environment.md`](dev-environment.md), "User documentation (D1)"). |
| [`user/doc-audit-allow.toml`](user/doc-audit-allow.toml) | The functions exempt from the running-example rule of the API docs, each with a reason. |

## For developers

These are not published and are not written for users.

| Path | Contents |
| --- | --- |
| [`design.md`](design.md) | The design of tetraloma, the source of truth for the implementation. |
| [`adr/`](adr/) | Architecture decision records: every change to the design ([`adr/README.md`](adr/README.md)). |
| [`plans/`](plans/) | One plan per milestone, made from [`plans/TEMPLATE.md`](plans/TEMPLATE.md): tasks, deviations and the review. |
| [`perf/`](perf/) | Benchmark reports and their data (`perf/data/`), on which the vector and degree policies rest. |
| [`dev-environment.md`](dev-environment.md) | Development and test targets, one-time setup, test commands and local flags. |
| [`cubecl-upstream.md`](cubecl-upstream.md) | CubeCL and CubeK bugs, surprises and documentation gaps found during the work, with their workarounds. |

The working agreement for Claude Code is [`CLAUDE.md`](../CLAUDE.md), and the procedure for humans [`IMPLEMENTATION.md`](../IMPLEMENTATION.md), both at the repository root.
