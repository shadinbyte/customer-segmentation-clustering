"""
Shared segmentation and business-intelligence logic.

Both entry points, customer_segmentation_analysis.py and
customer_segmentation_dashboard.py, used to reimplement the same
age/income/spending thresholds and marketing copy independently.
That duplication had already drifted (see AUDIT_REPORT.md, section
5.4): the two files described the same "high spending" segment with
slightly different wording. Import the functions below from both
places instead of copying them again.

Honesty note on product suggestions: the dataset behind this project
has three columns, Age, Annual Income, and Spending Score. There is
no purchase history, product catalog, or transaction log anywhere in
the pipeline. The category suggestions below are heuristic examples
for a marketer to start from, not something a model derived from
actual purchases. The one number in this file that is genuinely
computed from data is spending_index, which compares a segment's
average Spending Score to the whole population's average. Everything
else (category text, priority, reason) is a label chosen from the
income/spending combination.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

# Category thresholds - the single source of truth for both scripts
AGE_YOUNG_MAX = 30
AGE_MIDDLE_AGED_MAX = 50

INCOME_LOW_MAX = 500_000
INCOME_MEDIUM_MAX = 1_000_000

SPENDING_LOW_MAX = 40
SPENDING_MEDIUM_MAX = 70


class AgeCategory(str, Enum):
    """Age group categories."""

    YOUNG = "Young"
    MIDDLE_AGED = "Middle-aged"
    SENIOR = "Senior"


class IncomeCategory(str, Enum):
    """Income level categories."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SpendingCategory(str, Enum):
    """Spending behavior categories."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class SegmentPriority(str, Enum):
    """How commercially attractive a segment looks, from income/spending alone."""

    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


def categorize_age(age: float) -> AgeCategory:
    """Categorize age into groups."""
    if age < AGE_YOUNG_MAX:
        return AgeCategory.YOUNG
    if age < AGE_MIDDLE_AGED_MAX:
        return AgeCategory.MIDDLE_AGED
    return AgeCategory.SENIOR


def categorize_income(income: float) -> IncomeCategory:
    """Categorize income into groups."""
    if income < INCOME_LOW_MAX:
        return IncomeCategory.LOW
    if income < INCOME_MEDIUM_MAX:
        return IncomeCategory.MEDIUM
    return IncomeCategory.HIGH


def categorize_spending(spending: float) -> SpendingCategory:
    """Categorize spending into groups."""
    if spending < SPENDING_LOW_MAX:
        return SpendingCategory.LOW
    if spending < SPENDING_MEDIUM_MAX:
        return SpendingCategory.MEDIUM
    return SpendingCategory.HIGH


def segment_priority(
    income_cat: IncomeCategory, spending_cat: SpendingCategory
) -> SegmentPriority:
    """Rank how commercially attractive a segment looks, from income/spending alone."""
    if spending_cat == SpendingCategory.HIGH and income_cat == IncomeCategory.HIGH:
        return SegmentPriority.HIGH
    if spending_cat == SpendingCategory.HIGH:
        return SegmentPriority.MEDIUM
    if spending_cat == SpendingCategory.LOW and income_cat == IncomeCategory.HIGH:
        return SegmentPriority.MEDIUM
    return SegmentPriority.LOW


def generate_marketing_strategy(
    income_cat: IncomeCategory,
    spending_cat: SpendingCategory,
    market_share_pct: Optional[float] = None,
) -> str:
    """
    Return a marketing strategy blurb for a segment.

    Args:
        income_cat: Segment's income category.
        spending_cat: Segment's spending category.
        market_share_pct: Segment size as a percentage of all customers.
            Used to tell a mass-market medium-spending segment apart
            from a small niche one. Pass None to skip that distinction
            (falls back to the mass-market wording).

    Returns:
        A short strategy description.
    """
    if spending_cat == SpendingCategory.HIGH and income_cat == IncomeCategory.HIGH:
        return "Premium products, VIP programs, exclusive offers, personalized service"
    if spending_cat == SpendingCategory.HIGH:
        return "Value bundles, loyalty rewards, installment plans, quality assurance"
    if spending_cat == SpendingCategory.LOW and income_cat == IncomeCategory.HIGH:
        return "Trust-building campaigns, product demonstrations, value proposition focus"
    if market_share_pct is not None and market_share_pct <= 20:
        return "Entry-level products, first-purchase discounts, education campaigns"
    return "Mass market campaigns, seasonal promotions, volume discounts"


@dataclass
class ProductSuggestion:
    """
    One illustrative product-category suggestion for a segment.

    This is a heuristic example, not a data-derived recommendation.
    See the module docstring for why.
    """

    category: str
    price_tier: str
    priority: str
    reason: str
    spending_index: str


def _spending_index_label(cluster_avg_spending: float, population_avg_spending: float) -> str:
    """Compare a segment's average Spending Score to the population average."""
    if not population_avg_spending:
        return "n/a (no population baseline)"
    index = cluster_avg_spending / population_avg_spending
    return f"{index:.2f}x population avg spending score"


def generate_product_suggestions(
    income_cat: IncomeCategory,
    spending_cat: SpendingCategory,
    cluster_avg_spending: float,
    population_avg_spending: float,
) -> List[ProductSuggestion]:
    """
    Build illustrative product-category suggestions for a segment.

    Args:
        income_cat: Segment's income category.
        spending_cat: Segment's spending category.
        cluster_avg_spending: Segment's average Spending Score.
        population_avg_spending: Whole dataset's average Spending Score.

    Returns:
        A short list of ProductSuggestion objects. category/price_tier/
        reason are heuristic labels, not model output. spending_index
        is the one number here that is actually computed from data.
    """
    index_label = _spending_index_label(cluster_avg_spending, population_avg_spending)

    if income_cat == IncomeCategory.HIGH and spending_cat == SpendingCategory.HIGH:
        return [
            ProductSuggestion(
                "Premium electronics (flagship phones, laptops)",
                "Premium",
                "Primary",
                "Premium segment with high purchasing power",
                index_label,
            ),
            ProductSuggestion(
                "Luxury lifestyle products",
                "Premium",
                "Primary",
                "Affluent profile, high average spending",
                index_label,
            ),
        ]

    if income_cat == IncomeCategory.HIGH:
        return [
            ProductSuggestion(
                "Mid-range business electronics",
                "Mid-range",
                "Primary",
                "High income but a more price-conscious spending pattern",
                index_label,
            ),
            ProductSuggestion(
                "Practical accessories",
                "Budget",
                "Secondary",
                "Low-risk entry point for this profile",
                index_label,
            ),
        ]

    if income_cat == IncomeCategory.MEDIUM and spending_cat == SpendingCategory.HIGH:
        return [
            ProductSuggestion(
                "Mid-range electronics, financing available",
                "Mid-range",
                "Primary",
                "Aspirational buyers; financing may help conversion",
                index_label,
            ),
            ProductSuggestion(
                "Lifestyle accessories",
                "Mid-range",
                "Secondary",
                "Affordable within this segment's typical budget",
                index_label,
            ),
        ]

    return [
        ProductSuggestion(
            "Budget-friendly essentials",
            "Budget",
            "Primary",
            "Accessible price point matches this segment's spending",
            index_label,
        ),
        ProductSuggestion(
            "Entry-level lifestyle products",
            "Budget",
            "Secondary",
            "Low commitment, broad appeal",
            index_label,
        ),
    ]
