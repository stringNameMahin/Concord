"""End to end over a small invented corpus: block, judge, count.

These tests are the shape of the Phase 4 exit criterion. The real one runs on
the starter corpus once extraction is unblocked; the logic it exercises is
exactly what runs here.
"""

from factories import make_fact
from test_block import BagOfWords

from concord.compare.engine import compare


def corpus():
    """Three documents that between them contain each required case."""
    return [
        # Case 1: the same figure at two scales, in two documents.
        make_fact(
            fact_id="f_deck",
            doc="deck",
            raw="8,142 Cr",
            period="FY 2023-24",
            claim="Revenue from services in FY24 was 8,142 Cr.",
        ),
        make_fact(
            fact_id="f_report",
            doc="report",
            raw="81,415.38 mn",
            period="FY 2023-24",
            consolidation="consolidated",
            claim="Revenue from services for the year ended March 31, 2024 was 81,415.38 million.",
        ),
        # Case 3: same metric, different period.
        make_fact(
            fact_id="f_prior",
            doc="report",
            raw="72,236.47 mn",
            period="FY 2022-23",
            consolidation="consolidated",
            claim="Revenue from services for the year ended March 31, 2023 was 72,236.47 million.",
        ),
        # Case 2: same metric, same stated context, incompatible figures.
        make_fact(
            fact_id="f_third",
            doc="filing",
            raw="79,900.00 mn",
            period="FY 2023-24",
            consolidation="consolidated",
            claim="Revenue from services for the year ended March 31, 2024 was 79,900.00 million.",
        ),
        # An unrelated metric, to prove blocking does not sweep everything in.
        make_fact(
            fact_id="f_other",
            doc="filing",
            predicate="employee_count",
            raw="84,000",
            period="FY 2023-24",
            claim="The company employed 84,000 people at year end.",
        ),
    ]


def verdicts(run):
    return {(r.fact_a, r.fact_b): r.verdict for r in run.relations}


def test_the_three_cases_come_out_of_one_pass():
    run = compare(corpus(), encoder=BagOfWords())
    found = verdicts(run)

    assert found[("f_deck", "f_report")] == "corroborates"
    assert found[("f_prior", "f_report")] == "reconciled_by_context"
    assert found[("f_report", "f_third")] == "contradicts"


def test_only_genuine_value_conflicts_reach_the_llm():
    run = compare(corpus(), encoder=BagOfWords())
    queued = {(r.fact_a, r.fact_b) for r in run.llm_queue}

    assert ("f_report", "f_third") in queued
    assert ("f_deck", "f_report") not in queued
    assert len(run.llm_queue) < run.blocking.candidates


def test_a_different_metric_is_never_related_to_the_others():
    run = compare(corpus(), encoder=BagOfWords())
    touched = {r.fact_a for r in run.relations} | {r.fact_b for r in run.relations}
    assert "f_other" not in touched


def test_unrelated_pairs_are_counted_but_not_stored():
    run = compare(corpus(), encoder=BagOfWords())
    assert run.dropped_unrelated == run.verdicts["unrelated"]
    assert all(relation.verdict != "unrelated" for relation in run.relations)

    kept = compare(corpus(), encoder=BagOfWords(), keep_unrelated=True)
    assert len(kept.relations) == len(run.relations) + run.dropped_unrelated


def test_relations_record_provenance_for_the_ui():
    run = compare(corpus(), encoder=BagOfWords())
    headline = next(
        r for r in run.relations if (r.fact_a, r.fact_b) == ("f_deck", "f_report")
    )
    assert headline.cross_document
    assert "value" in headline.blocked_by
    assert headline.decided_by == "deterministic"
    assert "overlap" in headline.explanation


def test_the_run_reports_the_numbers_the_readme_needs():
    run = compare(corpus(), encoder=BagOfWords())
    summary = run.summary()
    assert f"{run.blocking.theoretical_pairs} theoretical pairs" in summary
    assert "pairs to the LLM" in summary
    assert sum(run.verdicts.values()) == run.blocking.candidates


def test_an_alias_map_pulls_a_renamed_predicate_into_the_same_key():
    facts = [
        make_fact(fact_id="f_a", predicate="revenue_from_services", raw="8,142 Cr", period="FY24"),
        make_fact(
            fact_id="f_b",
            doc="d2",
            predicate="revenue_from_operations",
            raw="81,415.38 mn",
            period="FY24",
        ),
    ]
    plain = compare(facts)
    assert plain.verdicts["unrelated"] == 1

    aliased = compare(facts, aliases={"revenue_from_operations": "revenue_from_services"})
    assert aliased.relations[0].verdict == "corroborates"


def test_comparison_is_stable_across_runs():
    first = compare(corpus(), encoder=BagOfWords())
    second = compare(corpus(), encoder=BagOfWords())
    assert [(r.fact_a, r.fact_b, r.verdict) for r in first.relations] == [
        (r.fact_a, r.fact_b, r.verdict) for r in second.relations
    ]
