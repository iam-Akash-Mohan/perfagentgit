"""Risk tiers and selection. Both are code reading a fixed table.

A model that could set its own risk tier would set it to LOW, and a model that
could pick the winning candidate would pick the one it wrote most confidently.
Neither field is in the tool schema; both are computed here from what the
candidate says it will do.

T4 is not a tier a candidate can be submitted with — it is a rejection at plan
time, before a worktree exists. T3 is allowed into the artifact, flagged, and
cannot be selected.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..contracts import CandidateFix, EndpointModel, RiskTier

# The tier table, by approach. Everything in T1 either adds something the old
# code path did not have (an index, a fetch plan) or narrows a transaction to
# read-only; nothing in it changes what a query returns.
TIER_BY_APPROACH: dict[str, RiskTier] = {
    "add_index": RiskTier.LOW,
    "entity_graph": RiskTier.LOW,
    "fetch_join": RiskTier.LOW,
    "read_only_transaction": RiskTier.LOW,
    "dto_projection": RiskTier.LOW,
    "batch_query": RiskTier.MEDIUM,
    "query_rewrite": RiskTier.MEDIUM,
    "remove_redundant_call": RiskTier.MEDIUM,
    "move_work_out_of_loop": RiskTier.MEDIUM,
    # An approach the table does not name cannot be certified as low risk.
    # HIGH rather than MEDIUM so `other` is flagged and never auto-selected.
    "other": RiskTier.HIGH,
}

# T3: correct-looking changes whose failure mode is invisible in a diff.
CACHING_WORDS = (
    "cache", "caching", "cacheable", "@cacheable", "ehcache", "caffeine",
    "redis", "memoize", "memoise", "second-level cache",
)
TX_BOUNDARY_WORDS = (
    "propagation", "requires_new", "transaction boundary",
    "transactional boundary", "new transaction", "separate transaction",
    "split the transaction", "commit boundary", "nested transaction",
)

# T4: forbidden outright.
AUTH_WORDS = (
    "spring security", "authentication", "authorization", "@preauthorize",
    "@secured", "authenticat", "principal check",
)
DDL_FORBIDDEN = re.compile(
    r"\b(create\s+table|alter\s+table|drop\s+(table|column|index|view)|"
    r"add\s+column|modify\s+column|change\s+column|rename\s+(table|column)|"
    r"truncate\s+table|create\s+view)\b", re.I,
)
# The one schema change the design permits.
CREATE_INDEX = re.compile(
    r"^\s*create\s+(unique\s+)?index\s+[`\"]?[\w.]+[`\"]?\s+on\s+"
    r"[`\"]?[\w.]+[`\"]?\s*\([^)]+\)\s*;?\s*$", re.I,
)
INDEX_PARTS = re.compile(
    r"create\s+(?:unique\s+)?index\s+[`\"]?(?P<name>[\w.]+)[`\"]?\s+on\s+"
    r"[`\"]?(?P<table>\w+)[`\"]?\s*\((?P<columns>[^)]+)\)", re.I,
)

TEST_TREE = "src/test"


def parse_index(sql: str) -> tuple[str, str, list[str]] | None:
    """(index name, table, columns) from a CREATE INDEX statement, or None."""
    match = INDEX_PARTS.search(sql or "")
    if match is None:
        return None
    columns = [
        part.strip().strip("`\"").split()[0]
        for part in match.group("columns").split(",")
        if part.strip()
    ]
    return match.group("name"), match.group("table"), columns


def is_additive_index(sql: str) -> bool:
    """One CREATE INDEX statement and nothing else.

    Checked with a regex over the whole string rather than by looking for the
    words `create index` somewhere in it: `CREATE INDEX ...; DROP TABLE x;`
    contains both.
    """
    text = (sql or "").strip()
    if not text:
        return False
    if DDL_FORBIDDEN.search(text):
        return False
    return CREATE_INDEX.match(text) is not None


# Techniques JPA/Spring confine to a PARTICULAR kind of file -- not a style
# preference, a fact about where the framework permits the annotation to
# live. c2 (COL-04) described adding @SqlResultSetMapping while scoping the
# patch to one repository interface; that annotation can only go on an
# entity class, so the candidate was unimplementable the moment it was
# written, and the patch agent spent a full green-worktree build and
# baseline measurement discovering that for itself.
_REQUIRES_ENTITY_FILE = (
    "@sqlresultsetmapping", "@constructorresult", "@namednativequery",
)


def _looks_like_entity_file(path: str) -> bool:
    normalised = path.replace("\\", "/").lower()
    name = normalised.rsplit("/", 1)[-1]
    if name.endswith(("repository.java", "controller.java", "service.java",
                      "dto.java", "mapper.java")):
        return False
    return "/domain/" in normalised or "/entity/" in normalised or "/model/" in normalised


def call_path_files(model: EndpointModel) -> list[str]:
    """Every file on the endpoint's traced call path, in call order.

    `call_chain` is recorded by comprehension from the controller down to
    wherever the trace stopped; this is the one list in the whole artifact
    that already says which files call which, so scope is derived from it
    rather than re-guessed from prose.
    """
    seen: list[str] = []
    for hop in model.call_chain:
        if hop.file not in seen:
            seen.append(hop.file)
    return seen


def derive_scope(
    files_to_touch: list[str], model: EndpointModel,
) -> tuple[list[str], dict[str, str]]:
    """Expand files_to_touch to the rest of the endpoint's own call path.

    COL-06: a query_rewrite that changed a repository method's parameter
    type broke the service calling it, and the service was never in scope —
    the candidate had no way to fix what its own change broke. The file it
    needed was not invented; it was one hop up the SAME traced call chain
    the candidate's own target file sits on.

    Only ever walks toward the entry point from the deepest file the
    candidate already names, and only along a path this endpoint itself was
    traced on — never a file unrelated to the candidate's own target.
    Returns the (possibly unchanged) file list and a reason for each file
    added, keyed by path, for `CandidateFix.scope_reasons`.
    """
    path = call_path_files(model)
    on_path = [i for i, f in enumerate(path) if f in files_to_touch]
    out = list(files_to_touch)
    reasons: dict[str, str] = {}
    if on_path:
        deepest = path[max(on_path)]
        for file in path[: max(on_path)]:
            if file not in out:
                out.append(file)
                reasons[file] = (
                    f"on this endpoint's own traced call path to {deepest}, "
                    f"which this candidate already touches — included so a "
                    f"change to {deepest} has its caller in scope"
                )
    return out, reasons


def scope_mismatch_reason(*, description: str, mechanism_changed: str,
                          sql_change: str | None,
                          files_to_touch: list[str]) -> str | None:
    """Does this candidate's OWN description name a technique its OWN
    files_to_touch cannot support? None if the scope and the approach agree.

    Checked here, before a worktree exists, so a candidate that could never
    be implemented within its own declared scope is replaced in the same
    plan-submission conversation instead of reaching the patch agent, which
    has no choice at that point but to abandon after spending a build
    iteration finding out.
    """
    text = f"{description} {mechanism_changed} {sql_change or ''}".lower()
    if not any(marker in text for marker in _REQUIRES_ENTITY_FILE):
        return None
    if any(_looks_like_entity_file(f) for f in files_to_touch):
        return None
    return (
        "it describes @SqlResultSetMapping, @ConstructorResult or "
        "@NamedNativeQuery, which JPA only allows on an entity class, but "
        f"files_to_touch names no entity file: {', '.join(files_to_touch)}. "
        "A JPQL constructor expression (`SELECT NEW` the DTO's fully "
        "qualified name) needs none of this and stays inside the "
        "repository interface — prefer it for a DTO projection."
    )


def forbidden_reason(*, approach: str, description: str, sql_change: str | None,
                     files_to_touch: list[str]) -> str | None:
    """Why this candidate is T4, or None. T4 is refused, never recorded."""
    tests = [f for f in files_to_touch if f.replace("\\", "/").startswith(TEST_TREE)]
    if tests:
        return (f"it writes to {', '.join(tests)}. Test files are immutable: "
                f"{TEST_TREE}/** is write-denied for every phase.")

    if sql_change and not is_additive_index(sql_change):
        return ("its sql_change is not a single additive CREATE INDEX "
                f"statement: {sql_change.strip()[:160]!r}. An additive index "
                "is the only schema change permitted.")

    text = f"{approach} {description}".lower()
    if DDL_FORBIDDEN.search(text):
        return ("it describes a schema change beyond an additive index. "
                "Tables, columns and views are fixed.")
    if any(word in text for word in AUTH_WORDS):
        return "it touches authentication or authorization, which is out of scope."
    return None


def assess_risk(*, approach: str, description: str,
                predicted_effect: str = "") -> tuple[RiskTier, bool, str]:
    """(tier, requires_flag, why) for a candidate that is not forbidden.

    Caching and transaction-boundary changes are detected from the prose
    rather than from the approach, because both arrive described as something
    else — "keep the result between requests" is a cache.
    """
    text = f"{approach} {description} {predicted_effect}".lower()

    cached = [w for w in CACHING_WORDS if w in text]
    if cached:
        return (RiskTier.HIGH, True,
                f"it adds caching ({cached[0]}), which changes what the "
                f"endpoint returns over time rather than how fast it returns it")

    boundary = [w for w in TX_BOUNDARY_WORDS if w in text]
    if boundary:
        return (RiskTier.HIGH, True,
                f"it changes transaction boundaries ({boundary[0]}), whose "
                f"failure mode does not show up in a diff")

    tier = TIER_BY_APPROACH.get(approach, RiskTier.HIGH)
    if tier is RiskTier.HIGH:
        return (tier, True,
                f"approach '{approach}' is not in the risk table and cannot be "
                f"certified low-risk")
    return (tier, False, f"approach '{approach}' is tier {tier.value}")


# ──────────────────────────  schema-change feasibility  ──────────────────


def schema_change_blocked(*, approach: str, sql_change: str | None,
                          files_to_touch: list[str], repo: Path) -> str | None:
    """Would THIS candidate's schema change be unmeasurable, right now?

    Asked at SELECTION time, before a green worktree exists — before the patch
    agent spends a single build iteration on a candidate that could never have
    been verified. The isolation guard in `measure/isolation.py` used to be
    the only place this was checked, and by then the plan had already picked
    the candidate: the knowledge arrived too late to change anything.

    Blue is compared against ITSELF here, because there is no green yet and
    nothing has diverged — the only question this early is whether the
    harness has ANY per-arm isolation mechanism at all. It does not today, so
    this fires for every schema-changing candidate. That is correct: the
    measure-time guard exists as a BACKSTOP for when the environment changes
    between plan and measure, not as the primary check — this is.

    Two signals, matching `touches_schema`'s definition plus the literal case
    named in the brief: the approach is `add_index`, the migration directory
    appears among the files to touch, OR `sql_change` is set at all (a
    candidate can in principle carry DDL without declaring `add_index`).
    """
    from ..measure.isolation import arms_share_one_schema, touches_schema

    changes_schema = bool(sql_change) or touches_schema(
        approach=approach, files_changed=files_to_touch)
    if not changes_schema:
        return None
    return arms_share_one_schema(repo, repo)


# ────────────────────────────────  selection  ────────────────────────────

MAX_SELECTABLE_TIER = RiskTier.MEDIUM


def selection_problem(candidate: CandidateFix) -> str | None:
    """Why this candidate cannot be the selected one, or None."""
    if candidate.not_selectable_because is not None:
        return candidate.not_selectable_because
    if candidate.risk_tier > MAX_SELECTABLE_TIER:
        return (f"risk tier {candidate.risk_tier.value} is above the "
                f"T{MAX_SELECTABLE_TIER.value} ceiling for an unattended fix")
    if not candidate.preserves_response_bytes:
        return ("it does not claim to preserve the response bytes, and the "
                "frozen oracle compares them byte for byte")
    return None
