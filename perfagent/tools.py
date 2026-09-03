from pathlib import Path
from difflib import get_close_matches
import subprocess

MAX_RESULTS = 40
REPO_ROOT = Path("/Users/akashm/Documents/mini-spring").resolve()
MAX_LINES = 400


def _resolve(rel: str) -> Path:
    """Turn an agent-supplied path into a real path, or refuse."""
    # resolve() collapses .. and follows symlinks — do this BEFORE checking,
    # because "src/../../../etc/passwd" looks innocent until resolved.
    target = (REPO_ROOT / rel).resolve()

    # is_relative_to is the actual containment check. String matching on ".."
    # is not enough: a symlink inside the repo can point anywhere.
    if not target.is_relative_to(REPO_ROOT):
        raise ValueError(f"path '{rel}' is outside the repository")

    return target


def read_file(path: str, start_line: int = 1) -> str:
    target = _resolve(path)

    if target.is_dir():
        # Don't just fail — tell the model which tool it wanted instead.
        return f"'{path}' is a directory, not a file. Use search_code to find files."

    if not target.exists():
        # A good error costs one turn; a bad one costs four. The model
        # usually made a typo, so hand it the correct spelling.
        siblings = [p.name for p in target.parent.iterdir()] if target.parent.is_dir() else []
        close = get_close_matches(target.name, siblings, n=3)
        hint = f" Did you mean: {', '.join(close)}?" if close else ""
        return f"File not found: '{path}'.{hint}"

    # errors="replace" so a stray byte in a file doesn't crash the whole run.
    lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
    total = len(lines)

    # start_line is 1-based for the model (matches what editors show),
    # but list slicing is 0-based. This off-by-one is the classic bug here.
    begin = max(start_line - 1, 0)
    window = lines[begin:begin + MAX_LINES]

    # THE line numbers. Without these the model cannot cite anything —
    # it will invent line numbers instead, confidently.
    numbered = "\n".join(
        f"{begin + i + 1:5d}| {line}" for i, line in enumerate(window)
    )

    out = f"{path} (lines {begin + 1}-{begin + len(window)} of {total})\n{numbered}"

    # Silent truncation is the worst outcome: the model treats a partial file
    # as complete. Say it was cut, and say exactly how to get the rest.
    if begin + len(window) < total:
        nxt = begin + len(window) + 1
        out += (f"\n\n... truncated at line {begin + len(window)} of {total}. "
                f"Call read_file(path='{path}', start_line={nxt}) to continue.")

    return out


READ_FILE_TOOL = {
    "type": "function",
    "function": {
        "name": "read_file",
        # The description IS a prompt. This is where you tell the model when
        # to reach for this tool and what it gets back.
        "description": (
            "Read a source file from the repository. Returns the contents with "
            "line numbers, up to 400 lines at a time. Use the line numbers when "
            "citing code."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path relative to the repository root, "
                                   "e.g. src/main/java/com/ex/web/CustomerController.java",
                },
                "start_line": {
                    "type": "integer",
                    "description": "Line to start from (1-based). Default 1. "
                                   "Use this to continue reading a truncated file.",
                },
            },
            "required": ["path"],
        },
    },
}


def search_code(pattern: str, glob: str | None = None) -> str:
    # argv list, never shell=True. The agent supplies `pattern` — if that
    # string reached a shell, it could run anything. This is the difference
    # between "our code calls a binary" and "the agent has a shell".
    cmd = ["rg", "--line-number", "--with-filename", "--no-heading",
           "--max-count", "5", pattern]
    if glob:
        cmd += ["--glob", glob]
    cmd.append(str(REPO_ROOT))

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except FileNotFoundError:
        return "ripgrep (rg) is not installed."
    except subprocess.TimeoutExpired:
        return f"Search for '{pattern}' timed out. Use a more specific pattern."

    # rg exit codes: 0 = matches, 1 = no matches, 2 = error (bad regex etc.)
    if proc.returncode == 2:
        return f"Invalid search pattern '{pattern}': {proc.stderr.strip()}"

    hits = proc.stdout.splitlines()

    # Zero matches is NOT an error — it's a fact the model needs. Echo the
    # pattern back so it can see what it actually searched for.
    if not hits:
        return f"No matches for '{pattern}'."

    # Strip the absolute prefix. The model works in repo-relative paths, and
    # /Users/akashm/... on every line is pure token waste.
    root = str(REPO_ROOT) + "/"
    shown = [h.replace(root, "") for h in hits[:MAX_RESULTS]]

    out = "\n".join(shown)

    # Tell it the TRUE total, not just that it was cut. "40 of 217" tells the
    # model its pattern was too broad; "truncated" tells it nothing.
    if len(hits) > MAX_RESULTS:
        out += (f"\n\n... {MAX_RESULTS} of {len(hits)} matches shown. "
                f"Narrow with a more specific pattern or the glob argument.")

    return out


SEARCH_CODE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_code",
        "description": (
            "Search the repository for a regex pattern. Returns matching lines "
            "as path:line:content. Use this to locate classes, methods, "
            "annotations, or SQL before reading files."
        ),
        "parameters": {
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
            },
            "required": ["pattern"],
        },
    },
}

def emit_endpoint_model(**kwargs) -> str:
    return "accepted"

EMIT_TOOL = {
    "type": "function",
    "function": {
        "name": "emit_endpoint_model",
        "description": (
            "Call this ONCE when you have fully traced the endpoint. This is the "
            "only way to finish. Every field is required."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "route": {"type": "string"},

                "call_chain": {
                    "type": "array",
                    "description": "Every method executed, in order, from controller down.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string", "description": "e.g. CustomerService.search"},
                            "file":   {"type": "string"},
                            "line":   {"type": "integer"},
                            "does":   {"type": "string"},
                        },
                        "required": ["method", "file", "line", "does"],
                    },
                },

                "data_access": {
                    "type": "array",
                    "description": (
                        "EVERY database access this endpoint can trigger. Include lazy "
                        "collection loads and calls made inside loops, not only explicit "
                        "repository queries."
                    ),
                    "items": {
                        "type": "object",
                        "properties": {
                            "trigger": {"type": "string",
                                        "description": "What causes it, e.g. 'repo.findByFilter' "
                                                       "or 'lazy load of Customer.invoices'"},
                            "kind": {"type": "string",
                                     "enum": ["jpql_query", "native_query", "derived_query",
                                              "lazy_collection", "lazy_association", "other"]},
                            "file": {"type": "string"},
                            "line": {"type": "integer"},
                            "sql": {"type": "string", "description": "The query, if visible in code"},
                            "executes_per_row": {
                                "type": "boolean",
                                "description": "TRUE if this runs once per result row (inside a "
                                               "loop, stream, or .map over results). FALSE if it "
                                               "runs once per request.",
                            },
                        },
                        "required": ["trigger", "kind", "file", "line", "executes_per_row"],
                    },
                },

                "uncertainties": {
                    "type": "array",
                    "description": "Anything you could not determine from the code alone.",
                    "items": {"type": "string"},
                },
            },
            "required": ["route", "call_chain", "data_access", "uncertainties"],
        },
    },
}