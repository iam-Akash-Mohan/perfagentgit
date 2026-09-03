from __future__ import annotations
import pytest
from pydantic import ValidationError
from perfagent.contracts import (
    Citation, EndpointModel, HypothesisSet, Hypothesis,
    RiskTier, TerminalState, EquivalenceVerdict, RequestCorpus, RequestSpec,
)


def test_citation_ref():
    assert Citation(file="a/B.java", line=34).ref() == "a/B.java:34"


def test_hypothesis_set_requires_three():
    """Forced competition: fewer than 3 hypotheses is a contract violation,
    because a single hypothesis means the model locked onto its first idea."""
    one = Hypothesis(id="h1", claim="c", mechanism="m", layer="database_query",
                     evidence=["e"], attributed_ms=100, prediction="p",
                     confidence=0.5)
    with pytest.raises(ValidationError):
        HypothesisSet(hypotheses=[one], ranked_ids=["h1"], measured_total_ms=200)


def test_attribution_validity():
    def h(i, ms):
        return Hypothesis(id=f"h{i}", claim="c", mechanism="m",
                          layer="database_query", evidence=["e"],
                          attributed_ms=ms, prediction="p", confidence=0.5)
    ok = HypothesisSet(hypotheses=[h(1, 100), h(2, 50), h(3, 20)],
                       ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert ok.attribution_valid()
    bad = HypothesisSet(hypotheses=[h(1, 5000), h(2, 50), h(3, 20)],
                        ranked_ids=["h1", "h2", "h3"], measured_total_ms=200)
    assert not bad.attribution_valid()


def test_confidence_bounded():
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", layer="database_query",
                   evidence=[], attributed_ms=1, prediction="p", confidence=1.4)


def test_verdict_decided_by_code_only():
    """A verdict must never be attributable to the model."""
    v = EquivalenceVerdict(passed=True, compared=10, identical=10,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="abc")
    assert v.decided_by == "code"
    with pytest.raises(ValidationError):
        EquivalenceVerdict(passed=True, compared=1, identical=1,
                           db_state_matched=True, external_calls_matched=True,
                           held_back_passed=True, oracle_hash="a",
                           decided_by="model")


def test_endpoint_model_collects_citations():
    m = EndpointModel(route="POST /x",
                      entry=Citation(file="C.java", line=1),
                      path=[Citation(file="C.java", line=1),
                            Citation(file="S.java", line=9)])
    assert len(m.all_citations()) >= 2
    assert m.uncertainties == []


def test_corpus_held_back_defaults_empty_but_present():
    c = RequestCorpus(benchmark=[RequestSpec(id="r1", method="GET", path="/x")],
                      hash="h")
    assert c.held_back == []


def test_terminal_states_cover_all_eight():
    assert len(TerminalState) == 8
    assert TerminalState.NO_BOTTLENECK_FOUND in TerminalState


def test_risk_tiers_ordered():
    assert RiskTier.LOW < RiskTier.HIGH < RiskTier.FORBIDDEN


def test_hypothesis_requires_a_layer():
    """A hypothesis that does not say WHERE the time goes cannot be checked
    for layer diversity, so the field is required rather than defaulted."""
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", evidence=[],
                   attributed_ms=1, prediction="p", confidence=0.5)


def test_hypothesis_rejects_an_unknown_layer():
    with pytest.raises(ValidationError):
        Hypothesis(id="h", claim="c", mechanism="m", layer="gpu",
                   evidence=[], attributed_ms=1, prediction="p", confidence=0.5)
