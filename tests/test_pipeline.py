"""The ingest seam: what survives from one document into the ledger."""

from factories import make_fact

from concord.pipeline import dedupe


def test_the_same_claim_extracted_twice_collapses_to_one_record():
    """Observed on the real deck: identical facts on two extraction passes."""
    twice = [make_fact(fact_id="f_x", span=100), make_fact(fact_id="f_x", span=100)]
    kept, dropped = dedupe(twice)
    assert len(kept) == 1 and dropped == 1


def test_the_better_qualified_copy_wins():
    """The impoverished copy would drag pairs into insufficient_context."""
    bare = make_fact(fact_id="f_x", span=100)
    qualified = make_fact(fact_id="f_x", span=100, period="FY24", consolidation="consolidated")
    for order in ([bare, qualified], [qualified, bare]):
        kept, dropped = dedupe(order)
        assert dropped == 1
        assert kept[0].qualifier_keys() == {"period", "consolidation"}


def test_distinct_facts_are_left_alone():
    facts = [make_fact(fact_id="f_a"), make_fact(fact_id="f_b")]
    kept, dropped = dedupe(facts)
    assert len(kept) == 2 and dropped == 0
