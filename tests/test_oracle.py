"""P13: recording, freezing, hashing, the policy guards, and comparison.

Nothing here touches the network (httpx.MockTransport) or a real database
(`capture_db=False`), per CLAUDE.md.

Read in four parts: provenance and recording, freezing and hashing, the
tolerance policy, then `canonicalise()` / `diff_bodies()` / `compare()`. The
last group carries the phase's stated acceptance tests, including the one that
matters most — a patch that special-cases benchmark inputs being caught by the
held-back set.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import httpx
import pytest

from perfagent.contracts import (BehaviourOracle, RecordedResponse,
                                 RequestCorpus, RequestSpec, TolerancePolicy)
from perfagent.errors import ContractViolation, ToolError
from perfagent.measure.corpus import corpus_hash
from perfagent.oracle import canonicalise as canon
from perfagent.oracle import record as rec
from perfagent.oracle.compare import (MISSING, align, compare, diff_bodies,
                                      resolve_policy, verify_corpus)

EMPTY_POLICY = TolerancePolicy()


# ──────────────────────────────  fixtures  ───────────────────────────────


def make_corpus(n_benchmark: int = 3, n_held_back: int = 1) -> RequestCorpus:
    benchmark = [
        RequestSpec(id=f"bench-{i}", method="POST", path="/api/customers/search",
                    body={"name": "a", "page": i, "size": 1})
        for i in range(n_benchmark)
    ]
    held_back = [
        RequestSpec(id=f"held-{i}", method="POST", path="/api/customers/search",
                    body={"name": "z", "page": i, "size": 1})
        for i in range(n_held_back)
    ]
    return RequestCorpus(
        benchmark=benchmark,
        held_back=held_back,
        hash=corpus_hash(benchmark, held_back),
    )


@pytest.fixture
def git_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "target-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo, check=True)
    (repo / "README.md").write_text("x")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "init"], cwd=repo, check=True)
    return repo


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def json_handler(payload_for) -> httpx.Client:
    """A client that answers every request with JSON derived from its body."""
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else {}
        return httpx.Response(
            200,
            json=payload_for(body),
            headers={"content-type": "application/json", "date": "Wed, 03 Sep 2026 00:00:00 GMT"},
        )
    return mock_client(handler)


# ─────────────────────────  provenance / recording  ──────────────────────


def test_base_commit_marks_a_dirty_worktree(git_repo: Path) -> None:
    clean = rec.base_commit(git_repo)
    assert len(clean) == 40 and "+dirty" not in clean

    (git_repo / "README.md").write_text("changed")
    assert rec.base_commit(git_repo) == f"{clean}+dirty"


def test_base_commit_refuses_a_non_repo(tmp_path: Path) -> None:
    with pytest.raises(ContractViolation, match="not a git repository"):
        rec.base_commit(tmp_path)


def test_record_fires_every_corpus_request_once_and_in_order(
    tmp_path: Path, git_repo: Path
) -> None:
    seen: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"page": json.loads(request.content)["page"]},
                              headers={"content-type": "application/json"})

    corpus = make_corpus(n_benchmark=3, n_held_back=2)
    bundle = rec.record_oracle(
        "http://app", corpus,
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "oracle.json", client=mock_client(handler),
    )

    # Benchmark set first, then the held-back set, once each, in corpus order.
    assert len(seen) == 5
    assert [r["request_id"] for r in bundle["recorded"]] == [
        "bench-0", "bench-1", "bench-2", "held-0", "held-1"
    ]


def test_recorded_response_keeps_behavioural_headers_and_drops_noisy_ones(
    tmp_path: Path, git_repo: Path
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"ok": True},
            headers={
                "content-type": "application/json",
                "etag": "W/\"abc\"",
                "date": "Wed, 03 Sep 2026 00:00:00 GMT",
                "server": "netty",
            },
        )

    bundle = rec.record_oracle(
        "http://app", make_corpus(1, 0),
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "oracle.json", client=mock_client(handler),
    )
    headers = bundle["recorded"][0]["headers_of_interest"]
    assert headers == {"content-type": "application/json", "etag": 'W/"abc"'}
    # `date` changes every request and `content-length` changes with JSON
    # whitespace; recording either guarantees a false failure on every check.
    assert "date" not in rec.HEADERS_OF_INTEREST
    assert "content-length" not in rec.HEADERS_OF_INTEREST


def test_a_transport_error_records_nothing_at_all(tmp_path: Path, git_repo: Path) -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 2:
            raise httpx.ConnectError("connection refused")
        return httpx.Response(200, json={"ok": True},
                              headers={"content-type": "application/json"})

    out = tmp_path / "oracle.json"
    with pytest.raises(ToolError, match="bench-1"):
        rec.record_oracle(
            "http://app", make_corpus(3, 0),
            policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
            path=out, client=mock_client(handler),
        )
    # A partial oracle is not written: the hole would be exactly where a
    # regression hides.
    assert not out.exists()


def test_non_json_and_empty_bodies_round_trip(tmp_path: Path, git_repo: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        page = json.loads(request.content)["page"]
        if page == 0:
            return httpx.Response(204)
        if page == 1:
            return httpx.Response(200, text="plain", headers={"content-type": "text/plain"})
        # JSON content-type that will not parse: that IS today's behaviour.
        return httpx.Response(200, content=b"{not json",
                              headers={"content-type": "application/json"})

    bundle = rec.record_oracle(
        "http://app", make_corpus(3, 0),
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "oracle.json", client=mock_client(handler),
    )
    bodies = [r["body"] for r in bundle["recorded"]]
    assert bodies == [None, "plain", "{not json"]


# ────────────────────────────  freezing / hash  ──────────────────────────


def test_oracle_hash_changes_if_any_recorded_response_changes(
    tmp_path: Path, git_repo: Path
) -> None:
    corpus = make_corpus(2, 0)
    first = rec.record_oracle(
        "http://app", corpus,
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "a.json",
        client=json_handler(lambda body: {"page": body["page"], "total": 7}),
    )
    second = rec.record_oracle(
        "http://app", corpus,
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "b.json",
        client=json_handler(lambda body: {"page": body["page"], "total": 8}),
    )
    assert first["hash"] != second["hash"]

    # ... and identical behaviour hashes identically, or the hash would be
    # useless as an identity.
    third = rec.record_oracle(
        "http://app", corpus,
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False,
        path=tmp_path / "c.json",
        client=json_handler(lambda body: {"page": body["page"], "total": 7}),
    )
    assert third["hash"] == first["hash"]


def test_oracle_hash_covers_the_policy_and_the_base_commit(
    tmp_path: Path, git_repo: Path
) -> None:
    corpus = make_corpus(1, 0)
    kwargs = dict(
        repo_path=git_repo, capture_db=False,
        client=json_handler(lambda body: {"ok": True}),
    )
    strict = rec.record_oracle(
        "http://app", corpus, policy=EMPTY_POLICY, path=tmp_path / "s.json", **kwargs
    )
    lenient = rec.record_oracle(
        "http://app", corpus,
        policy=TolerancePolicy(ignore_paths=["$.ok"], rationale={"$.ok": "because"}),
        path=tmp_path / "l.json", **kwargs,
    )
    # A widened policy is a different oracle. Otherwise the tolerance could be
    # swapped under a bundle that still claims its original hash.
    assert strict["hash"] != lenient["hash"]


def test_load_oracle_rejects_a_tampered_bundle(tmp_path: Path, git_repo: Path) -> None:
    out = tmp_path / "oracle.json"
    rec.record_oracle(
        "http://app", make_corpus(1, 0),
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False, path=out,
        client=json_handler(lambda body: {"total": 7}),
    )
    assert rec.load_oracle(out).hash  # loads clean

    payload = json.loads(out.read_text())
    payload["recorded"][0]["body"]["total"] = 999   # hash left untouched
    out.write_text(json.dumps(payload))

    with pytest.raises(ContractViolation, match="has been modified"):
        rec.load_oracle(out)


def test_write_oracle_refuses_to_overwrite_without_force(
    tmp_path: Path, git_repo: Path
) -> None:
    out = tmp_path / "oracle.json"
    args = ("http://app", make_corpus(1, 0))
    kwargs = dict(policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False, path=out,
                  client=json_handler(lambda body: {"total": 7}))
    first = rec.record_oracle(*args, **kwargs)

    with pytest.raises(ContractViolation, match="already exists"):
        rec.record_oracle(*args, **kwargs)
    assert json.loads(out.read_text())["hash"] == first["hash"]


def test_force_archives_the_previous_bundle_rather_than_deleting_it(
    tmp_path: Path, git_repo: Path
) -> None:
    out = tmp_path / "oracle.json"
    kwargs = dict(policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False, path=out)
    first = rec.record_oracle(
        "http://app", make_corpus(1, 0),
        client=json_handler(lambda body: {"total": 7}), **kwargs
    )
    second = rec.record_oracle(
        "http://app", make_corpus(1, 0), force=True,
        client=json_handler(lambda body: {"total": 8}), **kwargs
    )

    archive = tmp_path / f"oracle.{first['hash']}.json"
    assert archive.exists(), "an overwritten oracle must remain recoverable"
    assert json.loads(archive.read_text())["hash"] == first["hash"]
    assert json.loads(out.read_text())["hash"] == second["hash"]


def test_dbstate_side_file_is_linked_to_its_bundle(tmp_path: Path, git_repo: Path) -> None:
    out = tmp_path / "oracle.json"
    bundle = rec.record_oracle(
        "http://app", make_corpus(1, 0),
        policy=EMPTY_POLICY, repo_path=git_repo, capture_db=False, path=out,
        client=json_handler(lambda body: {"total": 7}),
    )
    side = json.loads(rec.dbstate_path(out).read_text())
    # The advisory detail names the bundle it describes, so a stale side file
    # from an earlier recording is detectable rather than silently trusted.
    assert side["oracle_hash"] == bundle["hash"]


def test_db_state_hash_ignores_key_order_but_not_content() -> None:
    a = {"customer": {"count": 2, "sum_crc32": 9}, "invoice": {"count": 1, "sum_crc32": 3}}
    b = {"invoice": {"sum_crc32": 3, "count": 1}, "customer": {"sum_crc32": 9, "count": 2}}
    assert rec.db_state_hash(a) == rec.db_state_hash(b)

    c = {"customer": {"count": 3, "sum_crc32": 9}, "invoice": {"count": 1, "sum_crc32": 3}}
    assert rec.db_state_hash(a) != rec.db_state_hash(c)


def test_db_state_drift_names_the_table_and_the_row_counts() -> None:
    before = {"customer": {"count": 50000}, "invoice": {"count": 1200000}}
    after = {"customer": {"count": 50000}, "invoice": {"count": 1200001}}
    assert rec.db_state_drift(before, after) == ["invoice (count 1200000 -> 1200001)"]
    assert rec.db_state_drift(before, before) == []


def test_row_expression_marks_nulls_so_a_column_swap_is_visible() -> None:
    # CONCAT_WS skips NULLs, so without an explicit marker ('a', NULL, 'b') and
    # ('a', 'b', NULL) produce the same row text.
    assert "IFNULL" in rec._row_expression("name", "varchar")
    assert "CHAR(0)" in rec._row_expression("name", "varchar")
    assert rec._row_expression("payload", "blob").startswith("IFNULL(HEX(")


def test_snapshot_sql_refuses_a_backtick_in_an_identifier() -> None:
    with pytest.raises(ContractViolation, match="refusing to build SQL"):
        rec._row_expression("name`; DROP TABLE customer; --", "varchar")


# ────────────────────────────  tolerance policy  ─────────────────────────


def write_policy(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "tolerance.json"
    path.write_text(json.dumps(payload))
    return path


def test_the_committed_policy_tolerates_exactly_one_known_thing() -> None:
    """The tolerance ledger, as a test.

    This started empty and gained one entry: Spring's error-body `$.timestamp`,
    found by recording against unmodified code before any patch existed. Every
    future widening has to edit this test too, which is the point — a tolerance
    that can be added quietly is a tolerance that will be added to make a diff
    go away.
    """
    policy = canon.load_policy(canon.DEFAULT_TOLERANCE_PATH)
    assert policy.ignore_paths == ["$.timestamp"]
    assert policy.order_insensitive_paths == []
    assert policy.numeric_epsilon == 0.0
    # Not a placeholder: the reason has to be readable by a reviewer.
    assert len(policy.rationale["$.timestamp"]) > 80


def test_a_missing_policy_is_an_error_not_an_empty_default(tmp_path: Path) -> None:
    # An empty policy is a decision; a missing one is an omission, and silently
    # defaulting would hide that nobody ever wrote it.
    with pytest.raises(ContractViolation, match="tolerance policy not found"):
        canon.load_policy(tmp_path / "nope.json")


def test_a_tolerated_path_without_a_rationale_is_refused(tmp_path: Path) -> None:
    path = write_policy(tmp_path, {"ignore_paths": ["$.meta.generatedAt"]})
    with pytest.raises(ContractViolation, match="no rationale"):
        canon.load_policy(path)

    ok = write_policy(tmp_path, {
        "ignore_paths": ["$.meta.generatedAt"],
        "rationale": {"$.meta.generatedAt": "wall-clock stamp, not endpoint output"},
    })
    assert canon.load_policy(ok).ignore_paths == ["$.meta.generatedAt"]


def test_a_nonzero_epsilon_needs_a_rationale_too(tmp_path: Path) -> None:
    path = write_policy(tmp_path, {"numeric_epsilon": 0.01})
    with pytest.raises(ContractViolation, match="no rationale"):
        canon.load_policy(path)


def test_a_blank_rationale_does_not_count(tmp_path: Path) -> None:
    path = write_policy(tmp_path, {
        "ignore_paths": ["$.x"], "rationale": {"$.x": "   "},
    })
    with pytest.raises(ContractViolation, match="no rationale"):
        canon.load_policy(path)


def test_an_unknown_key_is_refused_rather_than_ignored(tmp_path: Path) -> None:
    # `ignore_path` (singular) would otherwise load as a silently empty policy.
    path = write_policy(tmp_path, {"ignore_path": ["$.x"]})
    with pytest.raises(ContractViolation, match="unknown key"):
        canon.load_policy(path)


def test_a_duplicate_path_is_refused(tmp_path: Path) -> None:
    path = write_policy(tmp_path, {
        "ignore_paths": ["$.x", "$.x"], "rationale": {"$.x": "reason"},
    })
    with pytest.raises(ContractViolation, match="same path twice"):
        canon.load_policy(path)


@pytest.mark.parametrize("bad", ["items[*].x", "$", "$..x", "$.items[*", "$.9lives"])
def test_unsupported_path_syntax_is_refused_at_load_time(bad: str) -> None:
    # A policy entry that matches nothing looks like a tolerance is in force
    # when comparison is in fact still strict. Fail at load, loudly.
    with pytest.raises(ContractViolation, match="unsupported tolerance path"):
        canon.parse_path(bad)


@pytest.mark.parametrize(
    "path,expected",
    [
        ("$.total", ["total"]),
        ("$.items[*].updatedAt", ["items", "*", "updatedAt"]),
        ("$.content[0].id", ["content", 0, "id"]),
        ("$.a.b.c", ["a", "b", "c"]),
    ],
)
def test_parse_path_segments(path: str, expected: list) -> None:
    assert canon.parse_path(path) == expected


def test_there_is_no_api_that_widens_the_policy() -> None:
    # Changing what is tolerated must be a deliberate human edit of
    # tolerance.json, reviewed like any other change.
    for name in ("save_policy", "set_policy", "add_ignore_path", "widen_policy"):
        assert not hasattr(canon, name), f"{name} must not exist"


# ────────────────────────────  comparison guards  ────────────────────────


def make_oracle(policy: TolerancePolicy | None = None, **overrides) -> BehaviourOracle:
    corpus = make_corpus(2, 1)
    fields = {
        "corpus_hash": corpus.hash,
        "recorded": [
            RecordedResponse(request_id=spec.id, status=200, body={"id": spec.id}).model_dump(
                mode="json"
            )
            for spec in [*corpus.benchmark, *corpus.held_back]
        ],
        "policy": (policy or EMPTY_POLICY).model_dump(mode="json"),
        "base_commit": "a" * 40,
    }
    fields.update(overrides)
    return BehaviourOracle.model_validate({**fields, "hash": rec.bundle_hash(fields)})


def test_resolve_policy_returns_the_frozen_policy() -> None:
    oracle = make_oracle()
    assert resolve_policy(oracle) is oracle.policy
    assert resolve_policy(oracle, EMPTY_POLICY) is oracle.policy


def test_resolve_policy_refuses_a_policy_that_differs_from_the_recorded_one() -> None:
    # This is the hole in the original spec: a comparison whose tolerance comes
    # from the caller is a comparison the caller can win.
    oracle = make_oracle()
    widened = TolerancePolicy(ignore_paths=["$.id"], rationale={"$.id": "convenient"})
    with pytest.raises(ContractViolation, match="differs from the one"):
        resolve_policy(oracle, widened)


def test_verify_corpus_refuses_a_corpus_the_oracle_was_not_recorded_from() -> None:
    oracle = make_oracle()
    verify_corpus(oracle, make_corpus(2, 1))  # same shape, same hash

    with pytest.raises(ContractViolation, match="does not match"):
        verify_corpus(oracle, make_corpus(3, 1))


def test_align_pairs_by_request_id_and_reports_both_kinds_of_gap() -> None:
    oracle = make_oracle()
    live = [
        RecordedResponse(request_id="bench-0", status=200, body={}),
        RecordedResponse(request_id="surprise", status=200, body={}),
    ]
    pairs, missing, unexpected = align(oracle, live)

    assert [r.request_id for _, r in pairs] == ["bench-0"]
    assert missing == ["bench-1", "held-0"]
    assert unexpected == ["surprise"]


def test_align_refuses_a_duplicated_replay_response() -> None:
    oracle = make_oracle()
    twice = [RecordedResponse(request_id="bench-0", status=200, body={})] * 2
    with pytest.raises(ContractViolation, match="two responses"):
        align(oracle, twice)


def test_missing_sentinel_is_distinct_from_none() -> None:
    # "field set to null" and "field absent" are different behaviours.
    assert MISSING is not None and MISSING != "null"


# ───────────────────────────────  invariants  ────────────────────────────


def test_the_oracle_is_not_in_any_agent_tool_surface() -> None:
    """CLAUDE.md invariant 2, enforced mechanically rather than by review."""
    root = Path(__file__).resolve().parents[1] / "perfagent"
    agent_facing = [root / "tools.py", *(root / "diagnose").glob("*.py")]
    for path in agent_facing:
        if not path.exists():
            continue
        source = path.read_text(encoding="utf-8")
        assert "perfagent.oracle" not in source and "from .oracle" not in source, (
            f"{path.name} imports the oracle. No agent-facing module may reach "
            "the recording, or the grader becomes something the agent can touch."
        )
        assert "oracle.json" not in source, f"{path.name} names the oracle bundle"


def test_record_and_replay_share_one_firing_path() -> None:
    """If they diverged, a difference in how WE send a request would surface as
    a behaviour change in the APPLICATION."""
    import inspect
    for func in (rec.record_oracle, rec.replay_live):
        assert "_capture(" in inspect.getsource(func)


# ────────────────────────────  canonicalise()  ───────────────────────────


def policy_of(**kwargs) -> TolerancePolicy:
    """A policy with rationales filled in, so tests are not about the loader."""
    paths = [*kwargs.get("ignore_paths", []), *kwargs.get("order_insensitive_paths", [])]
    rationale = {p: "test" for p in paths}
    if kwargs.get("numeric_epsilon"):
        rationale["numeric_epsilon"] = "test"
    return TolerancePolicy(**kwargs, rationale=rationale)


def test_canonicalise_returns_the_body_untouched_under_an_empty_policy() -> None:
    body = {"a": [3, 1, 2], "b": {"c": None}}
    assert canon.canonicalise(body, EMPTY_POLICY) == body


def test_canonicalise_drops_an_ignored_key_entirely() -> None:
    body = {"timestamp": "2026-01-01", "status": 400}
    out = canon.canonicalise(body, policy_of(ignore_paths=["$.timestamp"]))
    # Absent, not None: "the patch removed this field" and "policy ignores this
    # field" must not become the same shape.
    assert out == {"status": 400}
    assert "timestamp" not in out


def test_canonicalise_does_not_mutate_its_input() -> None:
    body = {"timestamp": "x", "items": [{"n": 2}, {"n": 1}]}
    snapshot = json.loads(json.dumps(body))
    canon.canonicalise(
        body, policy_of(ignore_paths=["$.timestamp"], order_insensitive_paths=["$.items"])
    )
    # The caller still holds the recorded oracle body and has to be able to
    # render it after a comparison.
    assert body == snapshot


def test_canonicalise_wildcard_reaches_every_element() -> None:
    body = {"items": [{"id": 1, "at": "t1"}, {"id": 2, "at": "t2"}]}
    out = canon.canonicalise(body, policy_of(ignore_paths=["$.items[*].at"]))
    assert out == {"items": [{"id": 1}, {"id": 2}]}


def test_canonicalise_fixed_index_targets_one_element() -> None:
    body = {"items": [{"id": 1}, {"id": 2}, {"id": 3}]}
    out = canon.canonicalise(body, policy_of(ignore_paths=["$.items[1]"]))
    assert out == {"items": [{"id": 1}, {"id": 3}]}


def test_canonicalise_sorts_a_tolerated_array() -> None:
    a = {"items": [{"id": 2}, {"id": 1}, {"id": 3}]}
    b = {"items": [{"id": 1}, {"id": 3}, {"id": 2}]}
    policy = policy_of(order_insensitive_paths=["$.items"])
    assert canon.canonicalise(a, policy) == canon.canonicalise(b, policy)


def test_canonicalise_sorts_a_heterogeneous_array_without_a_type_error() -> None:
    # Sorting by the element itself raises TypeError comparing dict to int, and
    # that crash would surface as a mysterious oracle failure.
    a = {"mixed": [{"k": 1}, 3, "s", None, [1]]}
    b = {"mixed": ["s", [1], None, {"k": 1}, 3]}
    policy = policy_of(order_insensitive_paths=["$.mixed"])
    assert canon.canonicalise(a, policy) == canon.canonicalise(b, policy)


def test_canonicalise_sorts_by_the_canonicalised_element() -> None:
    # Two elements differing ONLY in an ignored field must sort together. If the
    # sort key were the raw element they would flap and produce phantom diffs.
    a = {"items": [{"id": 1, "at": "zzz"}, {"id": 2, "at": "aaa"}]}
    b = {"items": [{"id": 2, "at": "qqq"}, {"id": 1, "at": "bbb"}]}
    policy = policy_of(
        ignore_paths=["$.items[*].at"], order_insensitive_paths=["$.items"]
    )
    assert canon.canonicalise(a, policy) == canon.canonicalise(b, policy)
    assert canon.canonicalise(a, policy) == {"items": [{"id": 1}, {"id": 2}]}


def test_canonicalise_is_idempotent() -> None:
    body = {
        "timestamp": "x",
        "items": [{"id": 2, "at": "t"}, {"id": 1, "at": "t"}],
        "nested": {"deep": [3, 1, 2]},
    }
    policy = policy_of(
        ignore_paths=["$.timestamp", "$.items[*].at"],
        order_insensitive_paths=["$.items", "$.nested.deep"],
    )
    once = canon.canonicalise(body, policy)
    assert canon.canonicalise(once, policy) == once


@pytest.mark.parametrize("scalar", ["plain", 7, 1.5, None, True])
def test_canonicalise_passes_scalars_through(scalar) -> None:
    # A 204 with no body is None and must stay None.
    assert canon.canonicalise(scalar, policy_of(ignore_paths=["$.x"])) == scalar


def test_canonicalise_does_not_apply_numeric_epsilon() -> None:
    # Rounding here is a trap: with eps 0.001, 1.0049 and 1.0051 round apart yet
    # are within tolerance of each other. Epsilon is a leaf rule in compare().
    policy = policy_of(numeric_epsilon=0.5)
    assert canon.canonicalise({"v": 1.0049}, policy) == {"v": 1.0049}


def test_an_index_inside_a_sorted_array_is_refused(tmp_path: Path) -> None:
    # Sorting moves the element that [0] refers to, so canonicalise() would stop
    # being idempotent.
    path = write_policy(tmp_path, {
        "order_insensitive_paths": ["$.items"],
        "ignore_paths": ["$.items[0].at"],
        "rationale": {"$.items": "r", "$.items[0].at": "r"},
    })
    with pytest.raises(ContractViolation, match="fixed index inside"):
        canon.load_policy(path)


def test_unmatched_policy_paths_reports_a_stale_tolerance() -> None:
    # A tolerance that matches nothing reads as "accounted for" while
    # comparison is in fact still strict on that field.
    bodies = [{"timestamp": "x", "status": 400}]
    policy = policy_of(ignore_paths=["$.timestamp", "$.renamedAway"])
    assert canon.unmatched_policy_paths(bodies, policy) == ["$.renamedAway"]
    assert canon.unmatched_policy_paths(bodies, policy_of(ignore_paths=["$.timestamp"])) == []


# ────────────────────────────  diff_bodies()  ────────────────────────────


def diffs_of(before, after, policy: TolerancePolicy | None = None) -> list:
    p = policy or EMPTY_POLICY
    return diff_bodies("r1", canon.canonicalise(before, p), canon.canonicalise(after, p), p)


def test_identical_bodies_produce_no_diffs() -> None:
    body = {"content": [{"id": 1}], "total": 3, "nested": {"a": [1, 2]}}
    assert diffs_of(body, json.loads(json.dumps(body))) == []


def test_a_changed_field_names_the_exact_path() -> None:
    before = {"content": [{"id": 1}, {"id": 2, "total": 10}], "totalElements": 5}
    after = {"content": [{"id": 1}, {"id": 2, "total": 11}], "totalElements": 5}
    diffs = diffs_of(before, after)
    assert len(diffs) == 1
    assert diffs[0].path == "$.content[1].total"
    assert (diffs[0].before, diffs[0].after) == (10, 11)


def test_a_renamed_field_is_two_diffs_not_the_whole_subtree() -> None:
    diffs = diffs_of({"oldName": 1, "keep": 2}, {"newName": 1, "keep": 2})
    assert sorted(d.path for d in diffs) == ["$.newName", "$.oldName"]
    assert len(diffs) == 2


def test_absent_and_null_are_different_behaviours() -> None:
    diffs = diffs_of({"a": 1}, {"a": 1, "b": None})
    assert len(diffs) == 1
    assert diffs[0].path == "$.b"
    assert diffs[0].before is MISSING
    assert diffs[0].after is None


def test_a_body_containing_the_sentinel_string_is_not_read_as_absent() -> None:
    # Absence is decided by key membership, never by comparing to MISSING.
    assert diffs_of({"a": MISSING}, {"a": MISSING}) == []
    diffs = diffs_of({"a": MISSING}, {})
    assert len(diffs) == 1 and diffs[0].after is MISSING


def test_true_and_one_are_a_difference() -> None:
    # Equal in Python (bool subclasses int), different on the wire.
    diffs = diffs_of({"flag": True}, {"flag": 1})
    assert len(diffs) == 1 and diffs[0].path == "$.flag"


def test_int_and_float_are_a_difference() -> None:
    # 1 == 1.0 in Python; a Long that became a Double is a behaviour change.
    diffs = diffs_of({"v": 1}, {"v": 1.0})
    assert len(diffs) == 1 and diffs[0].path == "$.v"


def test_a_type_change_does_not_report_the_whole_subtree() -> None:
    diffs = diffs_of({"a": {"x": 1, "y": 2}}, {"a": "gone"})
    assert len(diffs) == 1 and diffs[0].path == "$.a"


def test_lists_of_different_length_are_one_diff_at_the_list() -> None:
    diffs = diffs_of({"items": [1, 2, 3]}, {"items": [1, 2, 3, 4]})
    assert len(diffs) == 1
    assert diffs[0].path == "$.items"
    assert (diffs[0].before, diffs[0].after) == ("<list of 3>", "<list of 4>")


def test_epsilon_tolerates_a_float_inside_it_and_not_outside() -> None:
    policy = policy_of(numeric_epsilon=0.01)
    assert diffs_of({"v": 1.0}, {"v": 1.005}, policy) == []
    outside = diffs_of({"v": 1.0}, {"v": 1.5}, policy)
    assert len(outside) == 1 and outside[0].path == "$.v"


def test_epsilon_does_not_apply_to_ints_or_to_numbers_in_strings() -> None:
    policy = policy_of(numeric_epsilon=5.0)
    # An int change is a row count, a page size or an id.
    assert len(diffs_of({"total": 100}, {"total": 101}, policy)) == 1
    assert len(diffs_of({"total": "100"}, {"total": "101"}, policy)) == 1


def test_a_reordered_array_passes_with_the_policy_and_fails_without() -> None:
    before = {"items": [{"id": 1}, {"id": 2}]}
    after = {"items": [{"id": 2}, {"id": 1}]}
    assert diffs_of(before, after) != []
    assert diffs_of(before, after, policy_of(order_insensitive_paths=["$.items"])) == []


def test_a_timestamp_passes_with_the_policy_and_fails_without() -> None:
    before = {"timestamp": "2026-09-03T13:54:18.647+00:00", "status": 400}
    after = {"timestamp": "2026-09-03T14:14:23.981+00:00", "status": 400}
    diffs = diffs_of(before, after)
    assert len(diffs) == 1 and diffs[0].path == "$.timestamp"
    assert diffs_of(before, after, policy_of(ignore_paths=["$.timestamp"])) == []


# ──────────────────────────────  compare()  ──────────────────────────────


def responses_from(oracle: BehaviourOracle, **overrides) -> list[RecordedResponse]:
    """The recorded set, replayed verbatim except where a test changes it."""
    out = []
    for recorded in oracle.recorded:
        payload = recorded.model_dump(mode="json")
        payload.update(overrides.pop(recorded.request_id, {}))
        out.append(RecordedResponse.model_validate(payload))
    return out


def oracle_with_db(policy: TolerancePolicy | None = None) -> BehaviourOracle:
    """An oracle whose responses carry a database state hash."""
    corpus = make_corpus(2, 1)
    fields = {
        "corpus_hash": corpus.hash,
        "recorded": [
            RecordedResponse(
                request_id=spec.id, status=200, body={"id": spec.id, "total": 3},
                headers_of_interest={"content-type": "application/json"},
                db_state_hash="db0",
            ).model_dump(mode="json")
            for spec in [*corpus.benchmark, *corpus.held_back]
        ],
        "policy": (policy or EMPTY_POLICY).model_dump(mode="json"),
        "base_commit": "a" * 40,
    }
    return BehaviourOracle.model_validate({**fields, "hash": rec.bundle_hash(fields)})


def test_an_identical_replay_passes() -> None:
    oracle = oracle_with_db()
    verdict = compare(oracle, responses_from(oracle), make_corpus(2, 1))
    assert verdict.passed
    assert verdict.compared == 3 and verdict.identical == 3
    assert verdict.diffs == []
    assert verdict.db_state_matched and verdict.held_back_passed
    assert verdict.external_calls_matched
    assert verdict.oracle_hash == oracle.hash
    assert verdict.decided_by == "code"


def test_one_changed_field_fails_and_names_the_path() -> None:
    oracle = oracle_with_db()
    live = responses_from(oracle, **{"bench-0": {"body": {"id": "bench-0", "total": 4}}})
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.passed
    assert verdict.compared == 3 and verdict.identical == 2
    assert [(d.request_id, d.path, d.before, d.after) for d in verdict.diffs] == [
        ("bench-0", "$.total", 3, 4)
    ]


def test_a_status_change_fails_and_skips_the_body() -> None:
    oracle = oracle_with_db()
    # 400 -> 500 with a body that ALSO differs: only the status is reported,
    # because the bodies of two different error paths tell you nothing.
    live = responses_from(oracle, **{"bench-0": {"status": 500, "body": {"other": 1}}})
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.passed
    paths = [(d.request_id, d.path) for d in verdict.diffs]
    assert paths == [("bench-0", "$status")]


def test_an_identical_body_under_a_changed_status_still_fails() -> None:
    oracle = oracle_with_db()
    live = responses_from(oracle, **{"bench-0": {"status": 500}})
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.passed
    assert verdict.diffs[0].path == "$status"
    assert (verdict.diffs[0].before, verdict.diffs[0].after) == (200, 500)


def test_a_content_type_change_fails() -> None:
    oracle = oracle_with_db()
    live = responses_from(
        oracle, **{"bench-1": {"headers_of_interest": {"content-type": "text/plain"}}}
    )
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.passed
    assert verdict.diffs[0].path == "$headers.content-type"


def test_a_database_write_fails_the_verdict() -> None:
    oracle = oracle_with_db()
    live = responses_from(oracle, **{"bench-1": {"db_state_hash": "db1"}})
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.db_state_matched
    assert not verdict.passed
    assert [d.path for d in verdict.diffs] == ["$db_state"]


def test_no_db_capture_cannot_produce_a_pass() -> None:
    # Reporting db_state_matched True would be a clean bill of health for a
    # check that never looked.
    oracle = make_oracle()  # every db_state_hash is None
    verdict = compare(oracle, responses_from(oracle), make_corpus(2, 1))
    assert verdict.diffs == []
    assert not verdict.db_state_matched
    assert not verdict.passed


def test_an_external_call_change_fails() -> None:
    oracle = oracle_with_db()
    live = responses_from(
        oracle, **{"bench-0": {"external_calls_made": [{"url": "http://x"}]}}
    )
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.external_calls_matched
    assert not verdict.passed


def test_a_missing_replay_response_fails_and_is_visible() -> None:
    oracle = oracle_with_db()
    live = [r for r in responses_from(oracle) if r.request_id != "bench-1"]
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.passed
    assert verdict.compared == 2
    gap = [d for d in verdict.diffs if d.request_id == "bench-1"]
    assert len(gap) == 1 and gap[0].after is MISSING


def test_a_patch_that_special_cases_benchmark_inputs_is_caught_by_the_held_back_set() -> None:
    """The important one. Benchmark requests answer perfectly; the held-back
    request — never used for measurement, so never seen by the agent — does not."""
    oracle = oracle_with_db()
    live = responses_from(oracle, **{"held-0": {"body": {"id": "held-0", "total": 99}}})
    verdict = compare(oracle, live, make_corpus(2, 1))

    assert not verdict.held_back_passed
    assert not verdict.passed
    # Every benchmark request looked perfect, which is exactly how this failure
    # mode presents itself.
    assert verdict.identical == 2
    assert [d.request_id for d in verdict.diffs] == ["held-0"]


def test_a_held_back_request_that_never_ran_does_not_count_as_passed() -> None:
    oracle = oracle_with_db()
    live = [r for r in responses_from(oracle) if r.request_id != "held-0"]
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert not verdict.held_back_passed
    assert not verdict.passed


def test_compare_refuses_a_mismatched_corpus() -> None:
    oracle = oracle_with_db()
    with pytest.raises(ContractViolation, match="does not match"):
        compare(oracle, responses_from(oracle), make_corpus(3, 1))


def test_compare_refuses_a_widened_policy() -> None:
    oracle = oracle_with_db()
    widened = policy_of(ignore_paths=["$.total"])
    with pytest.raises(ContractViolation, match="differs from the one"):
        compare(oracle, responses_from(oracle), make_corpus(2, 1), widened)


def test_compare_uses_the_frozen_policy_not_the_strictest_one() -> None:
    # A tolerance recorded WITH the oracle is honoured; that is the only way a
    # tolerance can be honoured.
    lenient = policy_of(ignore_paths=["$.total"])
    oracle = oracle_with_db(lenient)
    live = responses_from(oracle, **{"bench-0": {"body": {"id": "bench-0", "total": 999}}})
    verdict = compare(oracle, live, make_corpus(2, 1))
    assert verdict.passed


@pytest.mark.parametrize(
    "override,expected_false",
    [
        ({"bench-0": {"body": {"id": "bench-0", "total": 4}}}, "body"),
        ({"bench-0": {"status": 500}}, "status"),
        ({"bench-0": {"headers_of_interest": {}}}, "headers"),
        ({"bench-0": {"db_state_hash": "db9"}}, "db"),
        ({"bench-0": {"external_calls_made": [{"u": 1}]}}, "external"),
        ({"held-0": {"body": {"id": "held-0", "total": 4}}}, "held-back"),
    ],
)
def test_every_failure_kind_alone_is_enough_to_fail(override, expected_false) -> None:
    """No branch may set `passed` True early. Every `return True` inside a
    verifier is a place a future edit makes it lie."""
    oracle = oracle_with_db()
    verdict = compare(oracle, responses_from(oracle, **override), make_corpus(2, 1))
    assert not verdict.passed, f"{expected_false} failure did not fail the verdict"


def test_passed_cannot_be_set_by_anything_but_this_module() -> None:
    # The contract itself refuses any other author of the verdict.
    from pydantic import ValidationError

    from perfagent.contracts import EquivalenceVerdict
    with pytest.raises(ValidationError, match="decided_by"):
        EquivalenceVerdict(
            passed=True, compared=1, identical=1, db_state_matched=True,
            external_calls_matched=True, held_back_passed=True, oracle_hash="x",
            decided_by="model",
        )


# ──────────────────  bookkeeping tables, excluded and reported  ──────────


def test_classify_tables_excludes_only_exact_sealed_names() -> None:
    app, excluded = rec.classify_tables(
        ["customer", "invoice", "flyway_schema_history"])
    assert app == ["customer", "invoice"]
    assert excluded == ["flyway_schema_history"]


def test_classify_tables_recognises_every_sealed_name() -> None:
    app, excluded = rec.classify_tables(sorted(rec.BOOKKEEPING_TABLES))
    assert app == []
    assert excluded == sorted(rec.BOOKKEEPING_TABLES)


def test_a_table_with_no_bookkeeping_name_is_an_application_table() -> None:
    app, excluded = rec.classify_tables(["customer", "audit_log"])
    assert sorted(app) == ["audit_log", "customer"]
    assert excluded == []


def test_a_case_near_miss_is_an_error_not_a_guess() -> None:
    """Neither silently excluded (that IS the pattern-matching loophole this
    exists to prevent) nor silently hashed as application data (which would
    hide a real migration tool's bookkeeping under a different OS's table-name
    casing). Told, not guessed, either way."""
    with pytest.raises(ContractViolation, match="do not match exactly"):
        rec.classify_tables(["customer", "FLYWAY_SCHEMA_HISTORY"])
    with pytest.raises(ContractViolation, match="do not match exactly"):
        rec.classify_tables(["databasechangelog"])


def test_the_sealed_list_is_names_only_no_pattern() -> None:
    """A table that merely CONTAINS a bookkeeping name is an application
    table. If this ever starts matching substrings, the exclusion has become
    exactly the loophole it was built not to be."""
    app, excluded = rec.classify_tables(["my_flyway_schema_history_archive"])
    assert app == ["my_flyway_schema_history_archive"]
    assert excluded == []


class _FakeCursor:
    """Answers information_schema lookups, including an injected bookkeeping
    table, without a real database."""

    def __init__(self, tables: list[str]) -> None:
        self.tables = tables

    def execute(self, sql: str, params=()) -> None:
        self._sql = sql

    def fetchall(self):
        if "TABLE_NAME" in self._sql and "COLUMNS" not in self._sql:
            return [(t,) for t in self.tables]
        if "COLUMNS" in self._sql:
            return [("id", "bigint")]
        return []

    def fetchone(self):
        return (1, 10, 10)          # count, sum_crc32, xor_crc32

    def close(self) -> None:
        pass


def test_db_table_snapshot_excludes_bookkeeping_from_the_hashed_state(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """The proof for bullet 1: the table that broke F4 never reaches the hash."""
    from contextlib import contextmanager

    cursor = _FakeCursor(["customer", "invoice", "flyway_schema_history"])

    @contextmanager
    def fake_connect(target):
        yield type("Conn", (), {"cursor": lambda self: cursor})()

    monkeypatch.setattr(rec, "connect", fake_connect)
    from perfagent.measure.sql import MySQLTarget

    snapshot, excluded = rec.db_table_snapshot(MySQLTarget())
    assert excluded == ["flyway_schema_history"]
    assert set(snapshot) == {"customer", "invoice"}
    assert "flyway_schema_history" not in snapshot


# ──────────────────  the oracle hash is a behaviour fingerprint  ─────────


def test_two_recordings_differing_only_in_an_ignored_field_hash_identically(
    tmp_path: Path, git_repo: Path
) -> None:
    """Exactly the mini-spring finding: two back-to-back recordings of
    UNMODIFIED code produced different oracle hashes because $.timestamp on
    the 400 response differs between them — a field the policy already
    ignores. 'The oracle hash changed' must mean behaviour changed."""
    policy = TolerancePolicy(
        ignore_paths=["$.timestamp"],
        rationale={"$.timestamp": "server clock, not behaviour"},
    )
    corpus = make_corpus(1, 0)

    ticks = iter(range(100))

    def handler(body):
        return {"timestamp": f"2026-01-01T00:00:{next(ticks):02d}Z", "status": 400}

    first = rec.record_oracle(
        "http://app", corpus, policy=policy, repo_path=git_repo,
        capture_db=False, path=tmp_path / "a.json", client=json_handler(handler))
    second = rec.record_oracle(
        "http://app", corpus, policy=policy, repo_path=git_repo,
        capture_db=False, path=tmp_path / "b.json", client=json_handler(handler))

    assert (first["recorded"][0]["body"]["timestamp"]
            != second["recorded"][0]["body"]["timestamp"])
    assert first["hash"] == second["hash"]


def test_a_difference_the_policy_does_not_ignore_still_changes_the_hash(
    tmp_path: Path, git_repo: Path
) -> None:
    """The fingerprint is of BEHAVIOUR, not a rubber stamp: a field the policy
    does not tolerate still has to move the hash."""
    corpus = make_corpus(1, 0)
    first = rec.record_oracle(
        "http://app", corpus, policy=EMPTY_POLICY, repo_path=git_repo,
        capture_db=False, path=tmp_path / "a.json",
        client=json_handler(lambda body: {"total": 7}))
    second = rec.record_oracle(
        "http://app", corpus, policy=EMPTY_POLICY, repo_path=git_repo,
        capture_db=False, path=tmp_path / "b.json",
        client=json_handler(lambda body: {"total": 8}))
    assert first["hash"] != second["hash"]


def test_raw_bodies_still_live_on_disk_for_diagnosis(
    tmp_path: Path, git_repo: Path
) -> None:
    """The canonical form is what gets HASHED; the raw, uncanonicalised body
    is still what gets WRITTEN, so a human can read what the server actually
    said."""
    policy = TolerancePolicy(ignore_paths=["$.timestamp"],
                             rationale={"$.timestamp": "server clock"})
    out = tmp_path / "oracle.json"
    rec.record_oracle(
        "http://app", make_corpus(1, 0), policy=policy, repo_path=git_repo,
        capture_db=False, path=out,
        client=json_handler(lambda body: {"timestamp": "2026-01-01T00:00:00Z",
                                          "status": 400}))
    on_disk = json.loads(out.read_text())
    assert on_disk["recorded"][0]["body"]["timestamp"] == "2026-01-01T00:00:00Z"


def test_tampering_with_a_field_the_policy_does_not_ignore_is_still_caught(
    tmp_path: Path, git_repo: Path
) -> None:
    """The canonical-hash scheme must not weaken tamper-detection for
    anything that actually matters — only for what the policy already
    excused."""
    out = tmp_path / "oracle.json"
    rec.record_oracle(
        "http://app", make_corpus(1, 0), policy=EMPTY_POLICY,
        repo_path=git_repo, capture_db=False, path=out,
        client=json_handler(lambda body: {"total": 7}))
    assert rec.load_oracle(out).hash

    payload = json.loads(out.read_text())
    payload["recorded"][0]["body"]["total"] = 999
    out.write_text(json.dumps(payload))

    with pytest.raises(ContractViolation, match="has been modified"):
        rec.load_oracle(out)


def test_comparable_recording_reduces_to_exactly_what_compare_checks() -> None:
    response = RecordedResponse(
        request_id="r1", status=200, body={"timestamp": "now", "total": 7},
        headers_of_interest={"content-type": "application/json"},
        db_state_hash="abc", external_calls_made=[])
    policy = TolerancePolicy(ignore_paths=["$.timestamp"],
                             rationale={"$.timestamp": "x"})
    reduced = rec.comparable_recording(response, policy)
    assert reduced["body"] == {"total": 7}
    assert reduced["db_state_hash"] == "abc"
    assert reduced["status"] == 200


def test_excluded_tables_are_sealed_into_the_hash() -> None:
    """A later run claiming to exclude MORE tables than were actually found at
    record time must not silently inherit a passing hash."""
    base = dict(corpus_hash="c", recorded=[], policy=EMPTY_POLICY,
               base_commit="a" * 40)
    narrow = rec.oracle_hash(excluded_tables=[], **base)
    wide = rec.oracle_hash(excluded_tables=["flyway_schema_history"], **base)
    assert narrow != wide
