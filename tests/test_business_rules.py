"""
Tests for business_rules.py.

This is the shared module both entry points import their categorization
thresholds and marketing logic from (see AUDIT_REPORT.md, section 5.4 and
the remediation log). Pinning down the threshold boundaries here means a
future edit that shifts one by accident gets caught immediately, instead
of quietly changing both scripts' output at once.
"""

import pytest

from business_rules import (
    AgeCategory,
    IncomeCategory,
    SpendingCategory,
    SegmentPriority,
    categorize_age,
    categorize_income,
    categorize_spending,
    generate_marketing_strategy,
    generate_product_suggestions,
    segment_priority,
)


# Age boundaries: < 30 Young, < 50 Middle-aged, else Senior


def test_categorize_age_young():
    assert categorize_age(18) == AgeCategory.YOUNG
    assert categorize_age(29.9) == AgeCategory.YOUNG


def test_categorize_age_boundary_is_middle_aged_not_young():
    # 30 itself belongs to the next bucket, not "Young"
    assert categorize_age(30) == AgeCategory.MIDDLE_AGED


def test_categorize_age_middle_aged():
    assert categorize_age(35) == AgeCategory.MIDDLE_AGED
    assert categorize_age(49.9) == AgeCategory.MIDDLE_AGED


def test_categorize_age_senior():
    assert categorize_age(50) == AgeCategory.SENIOR
    assert categorize_age(80) == AgeCategory.SENIOR


# Income boundaries: < 500k Low, < 1M Medium, else High


def test_categorize_income_low():
    assert categorize_income(0) == IncomeCategory.LOW
    assert categorize_income(499_999) == IncomeCategory.LOW


def test_categorize_income_medium():
    assert categorize_income(500_000) == IncomeCategory.MEDIUM
    assert categorize_income(999_999) == IncomeCategory.MEDIUM


def test_categorize_income_high():
    assert categorize_income(1_000_000) == IncomeCategory.HIGH
    assert categorize_income(5_000_000) == IncomeCategory.HIGH


# Spending boundaries: < 40 Low, < 70 Medium, else High


def test_categorize_spending_low():
    assert categorize_spending(0) == SpendingCategory.LOW
    assert categorize_spending(39.9) == SpendingCategory.LOW


def test_categorize_spending_medium():
    assert categorize_spending(40) == SpendingCategory.MEDIUM
    assert categorize_spending(69.9) == SpendingCategory.MEDIUM


def test_categorize_spending_high():
    assert categorize_spending(70) == SpendingCategory.HIGH
    assert categorize_spending(100) == SpendingCategory.HIGH


# segment_priority - the four branches used by the dashboard


def test_segment_priority_high_income_high_spending_is_high():
    assert (
        segment_priority(IncomeCategory.HIGH, SpendingCategory.HIGH)
        == SegmentPriority.HIGH
    )


def test_segment_priority_high_spending_alone_is_medium():
    assert (
        segment_priority(IncomeCategory.MEDIUM, SpendingCategory.HIGH)
        == SegmentPriority.MEDIUM
    )


def test_segment_priority_high_income_low_spending_is_medium():
    assert (
        segment_priority(IncomeCategory.HIGH, SpendingCategory.LOW)
        == SegmentPriority.MEDIUM
    )


def test_segment_priority_default_is_low():
    assert (
        segment_priority(IncomeCategory.LOW, SpendingCategory.MEDIUM)
        == SegmentPriority.LOW
    )


# generate_marketing_strategy - one check per branch, plus the
# market_share_pct split that only applies to the fallback branch


def test_marketing_strategy_premium_segment():
    strategy = generate_marketing_strategy(IncomeCategory.HIGH, SpendingCategory.HIGH)
    assert "Premium" in strategy


def test_marketing_strategy_high_spending_not_high_income():
    strategy = generate_marketing_strategy(
        IncomeCategory.MEDIUM, SpendingCategory.HIGH
    )
    assert "loyalty" in strategy.lower()


def test_marketing_strategy_untapped_potential_segment():
    strategy = generate_marketing_strategy(IncomeCategory.HIGH, SpendingCategory.LOW)
    assert "trust" in strategy.lower()


def test_marketing_strategy_fallback_small_segment_gets_entry_level_copy():
    strategy = generate_marketing_strategy(
        IncomeCategory.LOW, SpendingCategory.MEDIUM, market_share_pct=10
    )
    assert "entry-level" in strategy.lower()


def test_marketing_strategy_fallback_large_segment_gets_mass_market_copy():
    strategy = generate_marketing_strategy(
        IncomeCategory.LOW, SpendingCategory.MEDIUM, market_share_pct=35
    )
    assert "mass market" in strategy.lower()


def test_marketing_strategy_fallback_with_no_market_share_defaults_to_mass_market():
    strategy = generate_marketing_strategy(IncomeCategory.LOW, SpendingCategory.MEDIUM)
    assert "mass market" in strategy.lower()


# generate_product_suggestions - checks the one genuinely data-derived
# number (spending_index) and that every branch returns usable suggestions


@pytest.mark.parametrize(
    "income_cat,spending_cat",
    [
        (IncomeCategory.HIGH, SpendingCategory.HIGH),
        (IncomeCategory.HIGH, SpendingCategory.LOW),
        (IncomeCategory.MEDIUM, SpendingCategory.HIGH),
        (IncomeCategory.LOW, SpendingCategory.LOW),
    ],
)
def test_generate_product_suggestions_returns_at_least_one_suggestion(
    income_cat, spending_cat
):
    suggestions = generate_product_suggestions(
        income_cat, spending_cat, cluster_avg_spending=60, population_avg_spending=50
    )
    assert len(suggestions) >= 1
    for s in suggestions:
        assert s.category
        assert s.price_tier
        assert s.priority
        assert s.reason
        assert s.spending_index


def test_spending_index_reflects_above_average_cluster():
    # A cluster spending twice the population average should show a ~2.0x index
    suggestions = generate_product_suggestions(
        IncomeCategory.HIGH,
        SpendingCategory.HIGH,
        cluster_avg_spending=100,
        population_avg_spending=50,
    )
    assert "2.00x" in suggestions[0].spending_index


def test_spending_index_handles_zero_population_average():
    # Shouldn't divide by zero - should fall back to an explicit "n/a" label
    suggestions = generate_product_suggestions(
        IncomeCategory.LOW,
        SpendingCategory.LOW,
        cluster_avg_spending=10,
        population_avg_spending=0,
    )
    assert "n/a" in suggestions[0].spending_index.lower()
