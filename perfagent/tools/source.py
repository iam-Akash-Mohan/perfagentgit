"""Reading the repository: read_file, search_code, list_dir.

Every error message here is written for the model to act on. A good error
costs one turn; a bad one costs four, and the difference is almost always
whether the message contains the correction.
"""

from __future__ import annotations

import subprocess
from difflib import get_close_matches

from typing import Any, Callable

from ..errors import ToolError
from . import SourceRoot, ToolSpec

MAX_LINES = 400
MAX_RESULTS = 40
DEFAULT_PER_FILE_CAP = 3
SEARCH_TIMEOUT_S = 10
MAX_DIR_ENTRIES = 200


# ─────────────────────────────  read_file  ───────────────────────────────


def make_read_file(root: SourceRoot) -> Callable[..., str]:
    def read_file(path: str, start_line: int = 1) -> str:
        target = root.resolve(path)

        if target.is_dir():
            # Don't just fail — tell the model which tool it wanted instead.
            return (f"'{path}' is a directory, not a file. "
                    f"Use list_dir to see what is in it.")

        if not target.exists():
            # The model usually made a typo, so hand it the correct spelling.
            siblings = ([p.name for p in target.parent.iterdir()]
                        if target.parent.is_dir() else [])
            close = get_close_matches(target.name, siblings, n=3)
            hint = f" Did you mean: {', '.join(close)}?" if close else ""
            return f"File not found: '{path}'.{hint}"

        # errors="replace" so a stray byte in a file doesn't crash the run.
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        total = len(lines)

        # start_line is 1-based for the model (matching what editors show);
        # slicing is 0-based. This off-by-one is the classic bug here.
        begin = max(int(start_line) - 1, 0)
        window = lines[begin:begin + MAX_LINES]

        # THE line numbers. Without these the model cannot cite anything — it
        # will invent line numbers instead, confidently.
        numbered = "\n".join(f"{begin + i + 1:5d}| {line}"
                             for i, line in enumerate(window))

        out = (f"{path} (lines {begin + 1}-{begin + len(window)} of {total})\n"
               f"{numbered}")

        # Silent truncation is the worst outcome: the model treats a partial
        # file as complete. Say it was cut, and say how to get the rest.
        if begin + len(window) < total:
            nxt = begin + len(window) + 1
            out += (f"\n\n... truncated at line {begin + len(window)} of "
                    f"{total}. Call read_file(path='{path}', start_line={nxt}) "
                    f"to continue.")
        return out

    return read_file


READ_FILE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {
            "type": "string",
            "description": "Path relative to the repository root, e.g. "
                           "src/main/java/com/ex/web/CustomerController.java",
        },
        "start_line": {
            "type": "integer",
            "description": "Line to start from (1-based). Default 1. Use this "
                           "to continue reading a truncated file.",
        },
    },
    "required": ["path"],
}


# ────────────────────────────  search_code  ──────────────────────────────


def make_search_code(root: SourceRoot) -> Callable[..., str]:
    def search_code(pattern: str, glob: str | None = None,
                    max_per_file: int = DEFAULT_PER_FILE_CAP) -> str:
        try:
            cap = max(1, int(max_per_file))
        except (TypeError, ValueError):
            cap = DEFAULT_PER_FILE_CAP

        # argv list, never shell=True. The agent supplies `pattern` — if that
        # string reached a shell it could run anything. This is the difference
        # between "our code calls a binary" and "the agent has a shell".
        #
        # --sort path forces ripgrep single-threaded and therefore
        # DETERMINISTIC. Without it the same search returns the same lines in
        # a different order run to run, which makes an agent run
        # unreproducible and turns every cassette into a coin flip.
        cmd = ["rg", "--line-number", "--with-filename", "--no-heading",
               "--sort", "path", "--max-count", str(cap), pattern]
        if glob:
            cmd += ["--glob", glob]
        cmd.append(str(root.path))

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=SEARCH_TIMEOUT_S)
        except FileNotFoundError as exc:
            raise ToolError(
                "ripgrep (rg) is not installed, so search_code cannot run.",
                suggestion="Use read_file and list_dir instead.",
            ) from exc
        except subprocess.TimeoutExpired:
            return (f"Search for '{pattern}' timed out after "
                    f"{SEARCH_TIMEOUT_S}s. Use a more specific pattern.")

        # rg exit codes: 0 = matches, 1 = none, 2 = error (bad regex etc.)
        if proc.returncode == 2:
            return f"Invalid search pattern '{pattern}': {proc.stderr.strip()}"

        hits = proc.stdout.splitlines()

        # Zero matches is NOT an error — it's a fact the model needs. Echo the
        # pattern back so it can see what it actually searched for.
        if not hits:
            return f"No matches for '{pattern}'."

        # Strip the absolute prefix: the model works in repo-relative paths,
        # and the absolute prefix on every line is pure token waste.
        prefix = str(root.path) + "/"
        shown = [h.replace(prefix, "") for h in hits[:MAX_RESULTS]]
        out = "\n".join(shown)

        # State the per-file cap explicitly. Without this line the model reads
        # "3 matches in Foo.java" as "there are 3 matches in Foo.java", which
        # is a different claim and often false.
        out += (f"\n\n(at most {cap} match(es) shown PER FILE — a file may "
                f"contain more. Raise max_per_file if you need them.)")

        # The TRUE total, not just "truncated". "40 of 217" tells the model
        # its pattern was too broad; "truncated" tells it nothing.
        if len(hits) > MAX_RESULTS:
            out += (f"\n{MAX_RESULTS} of {len(hits)} matching lines shown. "
                    f"Narrow with a more specific pattern or the glob argument.")
        return out

    return search_code


SEARCH_CODE_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "pattern": {
            "type": "string",
            "description": "Regex, e.g. '@PostMapping' or 'class CustomerService'",
        },
        "glob": {
            "type": "string",
            "description": "Optional file filter, e.g. '**/*.java' or '**/repo/*.java'",
        },
        "max_per_file": {
            "type": "integer",
            "description": f"Matches shown per file. Default "
                           f"{DEFAULT_PER_FILE_CAP}. Raise it when you need "
                           f"every occurrence within one file.",
        },
    },
    "required": ["pattern"],
}


# ──────────────────────────────  list_dir  ───────────────────────────────


def make_list_dir(root: SourceRoot) -> Callable[..., str]:
    def list_dir(path: str = ".") -> str:
        target = root.resolve(path)
        if not target.exists():
            return f"Directory not found: '{path}'."
        if not target.is_dir():
            return f"'{path}' is a file, not a directory. Use read_file."

        entries = sorted(target.iterdir(), key=lambda p: (p.is_file(), p.name))
        visible = [p for p in entries if not p.name.startswith(".")]
        lines = [f"{p.name}/" if p.is_dir() else f"{p.name}  ({p.stat().st_size} bytes)"
                 for p in visible[:MAX_DIR_ENTRIES]]
        out = f"{path or '.'} ({len(visible)} entries)\n" + "\n".join(lines)
        if len(visible) > MAX_DIR_ENTRIES:
            out += f"\n... {MAX_DIR_ENTRIES} of {len(visible)} entries shown."
        return out

    return list_dir


LIST_DIR_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "path": {"type": "string",
                 "description": "Directory relative to the repository root. "
                                "Default '.' (the root itself)."},
    },
    "required": [],
}


def source_tools(root: SourceRoot) -> list[ToolSpec]:
    return [
        ToolSpec(
            name="read_file",
            description=("Read a source file from the repository. Returns the "
                         "contents with line numbers, up to 400 lines at a "
                         "time. Use the line numbers when citing code."),
            parameters=READ_FILE_PARAMETERS,
            handler=make_read_file(root),
        ),
        ToolSpec(
            name="search_code",
            description=("Search the repository for a regex pattern. Returns "
                         "matching lines as path:line:content, capped per file. "
                         "Use it to locate classes, methods, annotations or SQL "
                         "before reading files."),
            parameters=SEARCH_CODE_PARAMETERS,
            handler=make_search_code(root),
        ),
        ToolSpec(
            name="list_dir",
            description=("List a directory in the repository. Use it when you "
                         "do not know how the source tree is laid out."),
            parameters=LIST_DIR_PARAMETERS,
            handler=make_list_dir(root),
        ),
    ]
