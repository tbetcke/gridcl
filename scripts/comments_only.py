#!/usr/bin/env python3
"""Check (h) of scripts/check-user-docs.sh (ADR-0067, rule 7): Rust sources that differ from a base
commit only in comments.

Usage:
    scripts/comments_only.py --base <commit> <dir>...   compare <dir>/src/**/*.rs with <commit>
    scripts/comments_only.py --self-test                run the planted cases

A file passes when its token sequence, with every comment removed, equals the base commit's. The
tokenizer removes `//` line comments (`///` and `//!` included) and block comments (nested, `/**`
and `/*!` included). It keeps string literals (with escapes), raw strings (`r#"…"#`), byte and C
strings, and character and byte literals whole, and tells lifetimes (`'a`) from character literals
(`'a'`), so a `//` or `/*` inside a literal is never taken for a comment. Whitespace only separates
tokens, so re-wrapping code around a removed comment is no difference. An added or removed file is
a difference.

Every other file of the directories (manifests, build scripts, policy files, data that `src/`
includes) must equal the base commit's byte for byte, except `Cargo.toml`, which may gain comment
lines and the dev-dependency on tetraloma-testing that the API examples use (ADR-0067, rule 6.3).
Standard library only.

Exit status: 0 if every file passes, 1 if one differs, 2 on a usage or git error.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

IDENT_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


class TokenizeError(Exception):
    """A literal or block comment that does not end."""


def _skip_block_comment(src: str, i: int) -> int:
    """The index after the (nested) block comment that starts at `src[i:i+2] == '/*'`."""
    depth = 0
    n = len(src)
    while i < n:
        if src.startswith("/*", i):
            depth += 1
            i += 2
        elif src.startswith("*/", i):
            depth -= 1
            i += 2
            if depth == 0:
                return i
        else:
            i += 1
    raise TokenizeError("unterminated block comment")


def _quoted_end(src: str, i: int, quote: str) -> int:
    """The index after the quoted literal whose opening `quote` is at `src[i]` (escapes honoured)."""
    n = len(src)
    i += 1
    while i < n:
        c = src[i]
        if c == "\\":
            i += 2
        elif c == quote:
            return i + 1
        else:
            i += 1
    raise TokenizeError("unterminated literal")


def _raw_end(src: str, i: int) -> int | None:
    """The index after the raw string whose `r` is at `src[i]`, or None if `src[i]` starts none."""
    j = i + 1
    hashes = 0
    while j < len(src) and src[j] == "#":
        hashes += 1
        j += 1
    if j >= len(src) or src[j] != '"':
        return None
    closing = '"' + "#" * hashes
    end = src.find(closing, j + 1)
    if end < 0:
        raise TokenizeError("unterminated raw string")
    return end + len(closing)


def _char_end(src: str, i: int) -> int | None:
    """The index after the character literal at `src[i] == "'"`, or None for a lifetime or label."""
    n = len(src)
    if i + 1 >= n:
        return None
    if src[i + 1] == "\\":
        return _quoted_end(src, i, "'")
    # One character (any, including a quote of the other kind) and the closing quote.
    if i + 2 < n and src[i + 2] == "'" and src[i + 1] != "'":
        return i + 3
    return None


def tokens(src: str) -> list[str]:
    """The token sequence of Rust source `src` without comments."""
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        c = src[i]
        if c.isspace():
            i += 1
        elif src.startswith("//", i):
            end = src.find("\n", i)
            i = n if end < 0 else end
        elif src.startswith("/*", i):
            i = _skip_block_comment(src, i)
        elif c == '"':
            end = _quoted_end(src, i, '"')
            out.append(src[i:end])
            i = end
        elif c == "'":
            end = _char_end(src, i)
            if end is None:
                # A lifetime or label: the quote and the identifier after it.
                j = i + 1
                while j < n and src[j] in IDENT_CHARS:
                    j += 1
                out.append(src[i:j])
                i = j
            else:
                out.append(src[i:end])
                i = end
        elif c in IDENT_CHARS:
            j = i
            while j < n and src[j] in IDENT_CHARS:
                j += 1
            word = src[i:j]
            # Prefixed literals: r"…", r#"…"#, b"…", br"…", b'…', c"…", cr"…".
            if j < n and word in ("r", "br", "cr") and src[j] in '#"':
                end = _raw_end(src, j - 1)
                if end is not None:
                    out.append(src[i:end])
                    i = end
                    continue
            if j < n and word in ("b", "c") and src[j] == '"':
                end = _quoted_end(src, j, '"')
                out.append(src[i:end])
                i = end
                continue
            if j < n and word == "b" and src[j] == "'":
                end = _quoted_end(src, j, "'")
                out.append(src[i:end])
                i = end
                continue
            out.append(word)
            i = j
        else:
            out.append(c)
            i += 1
    return out


def git_show(commit: str, path: str) -> str | None:
    """The contents of `path` at `commit`, or None if it does not exist there."""
    res = subprocess.run(
        ["git", "show", f"{commit}:{path}"], capture_output=True, text=True, check=False
    )
    return res.stdout if res.returncode == 0 else None


def git_files(commit: str, dirs: list[str]) -> set[str]:
    """The `src/**/*.rs` files of `dirs` at `commit`."""
    res = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit, "--", *[f"{d}/src" for d in dirs]],
        capture_output=True,
        text=True,
        check=True,
    )
    return {p for p in res.stdout.splitlines() if p.endswith(".rs")}


def work_files(dirs: list[str]) -> set[str]:
    """The `src/**/*.rs` files of `dirs` in the working tree."""
    return {str(p) for d in dirs for p in Path(d, "src").rglob("*.rs")}


def first_difference(a: list[str], b: list[str]) -> str:
    """A short description of where token lists `a` (base) and `b` (now) first differ."""
    for k, (x, y) in enumerate(zip(a, b)):
        if x != y:
            ctx_a = " ".join(a[max(0, k - 4) : k + 4])
            ctx_b = " ".join(b[max(0, k - 4) : k + 4])
            return f"token {k}: base `{ctx_a}`, now `{ctx_b}`"
    return f"token {min(len(a), len(b))}: base has {len(a)} tokens, now {len(b)}"


def manifest_lines(text: str) -> list[str]:
    """The lines of a Cargo.toml that count: no blank or comment lines, and no tetraloma-testing
    entry in `[dev-dependencies]`."""
    out, section = [], ""
    for line in text.splitlines():
        t = line.strip()
        if not t or t.startswith("#"):
            continue
        if t.startswith("["):
            section = t
        elif section == "[dev-dependencies]" and t.split("=", 1)[0].strip() == "tetraloma-testing":
            continue
        out.append(t)
    return out


def tracked_files(commit: str | None, dirs: list[str]) -> set[str]:
    """The files of `dirs` other than `src/**/*.rs`: at `commit`, or in the working tree (tracked
    and untracked, git-ignored files excluded) if `commit` is None."""
    cmd = ["git", "ls-tree", "-r", "--name-only", commit] if commit else [
        "git", "ls-files", "--cached", "--others", "--exclude-standard"]
    res = subprocess.run([*cmd, "--", *dirs], capture_output=True, text=True, check=True)
    paths = {p for p in res.stdout.splitlines() if commit or Path(p).exists()}
    return {p for p in paths if not (p.split("/", 2)[1:2] == ["src"] and p.endswith(".rs"))}


def compare_others(base: str, dirs: list[str]) -> list[str]:
    """The non-source files of `dirs` that differ from `base` beyond what `manifest_lines` allows."""
    before = tracked_files(base, dirs)
    now = tracked_files(None, dirs)
    problems = [f"{p}: added" for p in sorted(now - before)]
    problems += [f"{p}: removed" for p in sorted(before - now)]
    for path in sorted(now & before):
        res = subprocess.run(["git", "show", f"{base}:{path}"], capture_output=True, check=True)
        old, new = res.stdout, Path(path).read_bytes()
        if old == new:
            continue
        if Path(path).name == "Cargo.toml":
            if manifest_lines(old.decode()) == manifest_lines(new.decode()):
                continue
        problems.append(f"{path}: differs from the base commit (only src/**/*.rs comments may change)")
    return problems


def compare(base: str, dirs: list[str]) -> list[str]:
    """The differences of `dirs` from `base`, one line each; empty if only comments changed (and
    the allowed manifest lines)."""
    before = git_files(base, dirs)
    now = work_files(dirs)
    problems = compare_others(base, dirs)
    problems += [f"{p}: added" for p in sorted(now - before)]
    problems += [f"{p}: removed" for p in sorted(before - now)]
    for path in sorted(now & before):
        old = git_show(base, path)
        new = Path(path).read_text(encoding="utf-8")
        if old == new:
            continue
        try:
            a, b = tokens(old or ""), tokens(new)
        except TokenizeError as e:
            problems.append(f"{path}: {e}")
            continue
        if a != b:
            problems.append(f"{path}: code differs at {first_difference(a, b)}")
    return problems


# Planted cases: (name, base, now, comments only?). The four cases named by docs/plans/D1.md, T0
# change code next to a comment-like construct and must be found; the others change comments only.
PLANTED = [
    (
        "changed literal",
        'const TOL: f64 = 1e-12; // tolerance\n',
        'const TOL: f64 = 1e-11; // tolerance\n',
        False,
    ),
    (
        "operator inside a doc comment",
        "fn f(x: u8) -> u8 {\n    x /** x * 2 / 1 */ * 2\n}\n",
        "fn f(x: u8) -> u8 {\n    x /** x * 2 / 1 */ + 2\n}\n",
        False,
    ),
    (
        "`//` inside a string",
        'const URL: &str = "https://example.org/a"; // the site\n',
        'const URL: &str = "https://example.org/b"; // the site\n',
        False,
    ),
    (
        "lifetime next to a character literal",
        "fn f<'a>(s: &'a str) -> char { let _ = ('\"', s); '/' }\n",
        "fn f<'a>(s: &'a str) -> char { let _ = ('\"', s); '*' }\n",
        False,
    ),
    (
        "raw string holding a comment opener",
        'const S: &str = r#"/* "# ; const T: u8 = 1;\n',
        'const S: &str = r#"/* "# ; const T: u8 = 2;\n',
        False,
    ),
    (
        "doc comments rewritten",
        "/// See docs/design.md, \"Geometry\".\n//! inner\nfn f() {}\n",
        "// Design: \"Geometry\".\n/// The geometry; see [the book](https://example.org).\n//! inner, rewritten\nfn f() {}\n",
        True,
    ),
    (
        "nested block comment and re-wrapped code",
        "fn f(a: u8, b: u8) -> u8 { a /* one /* two */ */ + b }\n",
        "fn f(a: u8, b: u8) -> u8 {\n    a + b\n}\n",
        True,
    ),
    (
        "byte string and byte character",
        "const B: &[u8] = b\"//x\"; const C: u8 = b'/'; // c\n",
        "const B: &[u8] = b\"//x\"; const C: u8 = b'/';\n",
        True,
    ),
]


# Planted manifest cases: (name, base, now, allowed?).
MANIFEST_BASE = '[package]\nname = "x"\n\n[dependencies]\nmpi = { workspace = true }\n\n[dev-dependencies]\nproptest.workspace = true\n'
MANIFEST_PLANTED = [
    (
        "manifest: tetraloma-testing dev-dependency with a comment",
        MANIFEST_BASE,
        MANIFEST_BASE + '# the API examples\ntetraloma-testing = { path = "../tetraloma-testing" }\n',
        True,
    ),
    (
        "manifest: tetraloma-testing as a normal dependency",
        MANIFEST_BASE,
        MANIFEST_BASE.replace("[dependencies]\n", '[dependencies]\ntetraloma-testing = { path = "../t" }\n'),
        False,
    ),
    (
        "manifest: a changed dependency",
        MANIFEST_BASE,
        MANIFEST_BASE.replace("workspace = true }", "workspace = true, optional = true }"),
        False,
    ),
]


def self_test() -> int:
    """Runs the planted cases; returns the number that the tokenizer or the manifest comparison
    judges wrongly."""
    wrong = 0
    for name, base, now, allowed in MANIFEST_PLANTED:
        same = manifest_lines(base) == manifest_lines(now)
        ok = same == allowed
        print(f"{'ok   ' if ok else 'WRONG'} {name}: {'passes' if same else 'fails'}")
        wrong += not ok
    for name, base, now, comments_only in PLANTED:
        same = tokens(base) == tokens(now)
        ok = same == comments_only
        verdict = "passes" if same else "fails"
        print(f"{'ok   ' if ok else 'WRONG'} {name}: {verdict}")
        wrong += not ok
    return wrong


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--base", help="the base commit")
    parser.add_argument("--self-test", action="store_true", help="run the planted cases")
    parser.add_argument("dirs", nargs="*", help="crate directories (their src/**/*.rs)")
    args = parser.parse_args()
    if args.self_test:
        return 1 if self_test() else 0
    if not args.base or not args.dirs:
        parser.error("--base and at least one directory are needed")
    missing = [d for d in args.dirs if not Path(d, "src").is_dir()]
    if missing:
        parser.error(f"no src/ directory in {', '.join(missing)}")
    try:
        problems = compare(args.base, args.dirs)
    except subprocess.CalledProcessError as e:
        print(f"comments_only.py: git failed: {e.stderr.strip()}", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
