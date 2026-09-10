"""Complementary categories, and the arithmetic that recognises them.

A breakdown by region, segment or class produces a pair between every two of
its rows. Each pair has the same subject, the same predicate, one differing
qualifier and two disagreeing values - the exact shape of an apparent
contradiction explained by context - and calling them `reconciled_by_context`
is noise, because nothing ever looked like a conflict. These tests pin what
the detector accepts and, more importantly, what it refuses.
"""

import itertools

import pytest
from factories import make_fact

from concord.compare.decide import decide
from concord.compare.partition import index


def breakdown(predicate, key, shares, **common):
    return [
        make_fact(raw=value, predicate=predicate, **{key: category}, **common)
        for category, value in shares.items()
    ]


# --- what a partition is ----------------------------------------------------

def test_shares_that_add_up_to_a_whole_are_one_distribution():
    facts = breakdown(
        "workforce_share",
        "region",
        {"Americas": "63%", "Europe": "15%", "Asia Pacific": "22%"},
        as_at="March 31, 2024",
    )
    found = index(facts)
    assert len(found) == 1
    assert all(found.groups[fact.fact_id] for fact in facts)
    assert found.key_for(facts[0], facts[1]) == "region"


def test_every_pair_inside_a_distribution_is_unrelated_not_reconciled():
    facts = breakdown(
        "workforce_share",
        "region",
        {"Americas": "63%", "Europe": "15%", "Asia Pacific": "22%"},
    )
    found = index(facts)
    for a, b in itertools.combinations(facts, 2):
        assert decide(a, b).verdict == "reconciled_by_context"
        decision = decide(a, b, partitions=found)
        assert decision.verdict == "unrelated"
        assert decision.rule_fired == "complementary_categories"
        assert decision.qualifier_key == "region"


def test_a_two_way_split_is_still_a_distribution():
    facts = breakdown("revenue_share", "channel", {"Domestic": "58%", "Overseas": "42%"})
    assert len(index(facts)) == 1


def test_rounding_is_carried_by_the_precision_intervals():
    """33.3 + 33.3 + 33.3 is 99.9 written out, and a whole as measured."""
    facts = breakdown(
        "holding_share", "holder", {"A": "33.3%", "B": "33.3%", "C": "33.3%"}
    )
    assert len(index(facts)) == 1


# --- what a partition is not ------------------------------------------------

def test_measurements_that_do_not_add_up_stay_reconcilable():
    """Coverage percentages of four groups are four measurements, not parts."""
    facts = breakdown(
        "training_coverage",
        "cohort",
        {"Directors": "100%", "Managers": "97%", "Workers": "96%"},
    )
    assert len(index(facts)) == 0
    assert decide(facts[0], facts[1]).verdict == "reconciled_by_context"


def test_absolute_quantities_are_never_a_partition():
    """Without the total in hand there is nothing to check the parts against."""
    facts = breakdown(
        "headcount", "region", {"Americas": "630", "Europe": "150", "Asia Pacific": "220"}
    )
    assert len(index(facts)) == 0


def test_one_category_is_not_a_distribution():
    facts = breakdown("workforce_share", "region", {"Americas": "100%"})
    assert len(index(facts)) == 0


def test_a_partition_does_not_reach_across_a_differing_context():
    """Two years of one breakdown are two distributions, not six parts of one."""
    this_year = breakdown(
        "workforce_share", "region", {"Americas": "63%", "Europe": "37%"}, period="FY24"
    )
    last_year = breakdown(
        "workforce_share", "region", {"Americas": "60%", "Europe": "40%"}, period="FY23"
    )
    found = index(this_year + last_year)
    assert len(found) == 2
    assert found.key_for(this_year[0], last_year[0]) is None


def test_the_same_category_in_one_distribution_is_still_compared():
    """Two documents reporting the same slice are a corroboration, not parts."""
    a, b = breakdown("workforce_share", "region", {"Americas": "63%", "Europe": "37%"})
    restated = make_fact(raw="63%", predicate="workforce_share", region="Americas", doc="d2")
    found = index([a, b, restated])
    assert found.key_for(a, restated) is None
    assert decide(a, restated, partitions=found).verdict == "corroborates"


def test_a_different_subject_is_a_different_distribution():
    ours = breakdown("workforce_share", "region", {"Americas": "63%", "Europe": "37%"})
    theirs = breakdown(
        "workforce_share",
        "region",
        {"Americas": "40%", "Europe": "60%"},
        subject="Borealis Freight Limited",
    )
    found = index(ours + theirs)
    assert len(found) == 2
    assert found.key_for(ours[0], theirs[1]) is None


@pytest.mark.parametrize("share", ["-20%", "160%"])
def test_a_figure_outside_the_whole_is_not_a_share(share):
    facts = breakdown("growth_share", "region", {"Americas": share, "Europe": "40%"})
    assert len(index(facts)) == 0
