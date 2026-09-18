# Technical Audit Report: Customer Segmentation Clustering Project

**Auditor role:** Senior AI Engineer / Solution Architect / Technical Reviewer
**Audit date:** 2026-09-17
**Artifact audited:** `customer-segmentation-clustering.zip`
**Audit method:** Full static review of every source file, dependency verification against live PyPI metadata, and dynamic verification — the analysis pipeline and the Streamlit dashboard were both actually executed in a sandboxed environment to confirm the claimed results reproduce.

---

## 1. Executive Summary

This is a BSc capstone project that performs customer segmentation on a synthetic 3,000-row "mall customer" dataset using four clustering algorithms (K-Means, Hierarchical/Agglomerative, DBSCAN, Gaussian Mixture Model), and exposes the results through a Streamlit dashboard.

**Overall verdict: solid engineering execution, overstated business claims.**

- The code architecture (dataclasses, enums, structured logging, single-responsibility classes) is well above the typical student-project bar and the pipeline is genuinely reproducible — I re-ran it end-to-end and every headline metric in the README matched exactly.
- The most serious problem is not a bug, it's a **methodology and framing issue**: the "96.89% validation accuracy" is measured against ground-truth labels that come from a generator explicitly designed to produce well-separated Gaussian blobs, and the per-segment "conversion rates" (e.g. "85–90%") and named products (iPhone Pro Max, MacBook Pro) shown in the dashboard are hardcoded strings with no statistical basis in the data at all. Both are presented to the reader as if they were measured outcomes.
- There are also a handful of concrete, fixable defects: a broken quick-start command, a Python-version claim that contradicts the pinned dependencies, duplicated business logic that has already drifted between the two scripts, and zero automated tests despite the "production-ready" label.

None of this is disqualifying — it's a good academic project — but if this is being positioned to hiring managers as evidence of production-grade ML engineering (as the README's "Why This Matters for Hiring" section explicitly does), the gaps below are exactly what an experienced reviewer will probe first.

---

## 2. What Was Actually Verified (not just read)

| Check | Method | Result |
|---|---|---|
| Does `customer_segmentation_analysis.py` run end-to-end? | Installed pinned deps, executed the script in a clean sandbox | ✅ Runs clean, no exceptions |
| Do the headline metrics in the README reproduce? | Compared script output to README Table | ✅ Exact match (see §4.1) |
| Does the Streamlit dashboard start and serve traffic? | Launched `streamlit run` headless, curled `/_stcore/health` | ✅ `200 OK`, health check `ok` |
| Is the codebase free of unused imports / dead code? | Ran `pyflakes` on all three `.py` files | ⚠️ Several unused imports found |
| Do declared dependency versions match reality? | Queried live PyPI metadata for `requires_python` and latest versions | ❌ Contradiction found (§5.2) |
| Is `generator.py` (per README) actually present? | `ls` the extracted archive | ❌ File is named `genarator.py` (typo) |
| Do the two "business intelligence" modules agree? | Diffed logic in both scripts | ❌ Already drifted (§4.4) |

---

## 3. Project Structure (as received)

```
customer-segmentation-clustering/
├── customer_segmentation_analysis.py       # 1,921 lines — batch analysis pipeline
├── customer_segmentation_dashboard.py      # 1,616 lines — Streamlit app
├── genarator.py                            # 350 lines  — synthetic data generator (typo in filename)
├── requirements.txt                        # pinned dependencies (looks like a raw `pip freeze`)
├── README.md                               # marketing-style write-up, numerically accurate
├── LICENSE                                 # MIT
├── .gitignore
├── Synthetic_Mall_Customers_3000.csv               # clustering input (no true labels)
├── Synthetic_Mall_Customers_3000_with_labels.csv   # validation-only file (True_Segment column)
├── data_generation_validation.png          # generator QA plot
├── screenshots/
│   └── dashboard_main.png
└── clustering_results_3000/                # pre-generated run artifacts
    ├── analysis.log / analysis_summary.txt
    ├── {kmeans,hierarchical,dbscan,gmm}_results.csv
    ├── KMeans_Cluster_Summary.csv
    └── *.png  (EDA, correlation heatmap, optimal-k, dendrogram, per-algorithm clusters, comparison)
```

No `tests/`, no CI config (`.github/workflows`, etc.), no `Dockerfile`, no `pyproject.toml`. Everything is single-machine, script-based, and CSV-driven — consistent with an academic deliverable, not a deployable service, despite some README language to the contrary.

---

## 4. Findings — Methodology & Data Science Validity

This is the section that matters most for a clustering project, so it comes first.

### 4.1 Headline metrics are reproducible and accurate ✅

I re-ran `customer_segmentation_analysis.py` from a clean environment. Output matched the README exactly:

| Algorithm | Silhouette | Davies-Bouldin | Calinski-Harabasz | Clusters |
|---|---|---|---|---|
| K-Means | 0.5492 | 0.6507 | 6,504 | 5 |
| Validation ARI (K-Means vs. true labels) | 0.9689 | — | — | — |

Cluster sizes and demographic averages (age/income/spending per cluster) also matched the README's segment table to within rounding. This is worth stating plainly: **the numbers are not fabricated or cherry-picked** — that's a meaningfully positive signal about the author's rigor, and it's rarer than it should be in portfolio projects.

### 4.2 The 96.89% validation score is close to circular — this is the core issue

Look at `genarator.py`. The synthetic data isn't neutral — it's constructed as five explicit Gaussian blobs with tight, hand-tuned standard deviations, and the code comments say so directly:

```python
segments = {
    "Young Premium Spenders": {
        "age_std": 3,       # Tighter, younger
        "income_std": 120000,   # High income, tight
        "spending_std": 7,      # Very high spending, tight
        ...
```

The class docstring literally states the goal: *"Generate 5 distinct customer segments with CLEAR SEPARATION... Optimized for clustering algorithms to discover 5 distinct groups... Reduced correlations to emphasize segment differences."*

That means the "ground truth" used to compute the 96.89% Adjusted Rand Index is itself generated by carving out five well-separated Euclidean clusters — the exact structure K-Means and GMM are built to detect. A 0.97 ARI here demonstrates that K-Means can recover clusters that were deliberately shaped for K-Means to find. It does **not** demonstrate that the pipeline would perform this well on messy, real-world purchase data with overlapping segments, non-Gaussian distributions, or genuine noise — which is the actual hard problem in customer segmentation.

**Why this matters:** the README presents 96.89% as if it were an accuracy figure a business could rely on, and the "Why This Matters for Hiring" section leans on it as evidence of validation rigor. It is rigor — but rigor on a self-fulfilling benchmark. A one- or two-sentence caveat ("this validates the pipeline mechanics, not real-world segment separability") would turn this from an overstatement into an honest and still-impressive result.

### 4.3 Product recommendations and conversion rates are fabricated, not measured

In `customer_segmentation_analysis.py` (`BusinessIntelligenceEngine._generate_product_recommendations`) and its dashboard counterpart, every cluster gets hardcoded output like:

```python
{
    "product": "iPhone Pro Max",
    "priority": "Primary",
    "reason": "Premium segment with high purchasing power",
    "conversion": "85-90%",
},
```

The dataset contains exactly three features: Age, Annual Income, and Spending Score. There is no purchase history, no product catalog, no transaction log, and no conversion data anywhere in the pipeline. The "85–90%" conversion figures and specific branded product names are static strings selected by an if/elif chain on income/spending category — they are illustrative placeholders, not outputs of any model or statistical estimate.

This is the single most important thing to fix or caveat before showing this to anyone in a hiring or business context: as written, the dashboard presents fabricated numbers with the same visual authority as the genuinely-computed clustering metrics. Recommendation: either clearly label this section "illustrative recommendations, not data-derived" everywhere it appears, or replace hardcoded conversion percentages with something honestly derived (e.g., relative spending propensity within the segment).

### 4.4 Optimal-k selection quietly overrides a disagreeing metric

`OptimalClusterFinder._determine_optimal_k` computes k via three independent methods. Re-running the pipeline and inspecting the log shows:

```
Silhouette Score:        k = 5
Davies-Bouldin Index:    k = 4
Calinski-Harabasz Index: k = 5
```

The code always takes the Silhouette-based k with no reconciliation logic (`optimal_k = optimal_k_sil`), and the final report/README present "5 segments" as if all metrics agreed. Two out of three do — but the disagreement is silently dropped rather than surfaced. This is a minor statistical honesty issue: a single added log line ("Davies-Bouldin suggests k=4; proceeding with Silhouette/CH consensus of k=5") would close the gap.

### 4.5 DBSCAN's failure is honestly reported, but under-investigated

The README is admirably candid that DBSCAN found effectively one cluster on this data — that kind of transparency about a failed approach is a genuine strength of this project. That said, the auto-tuning heuristic (`eps` = 90th percentile of k-NN distances, no grid search) is a single point estimate. Given that the data is scaled, dense, and near-uniform in local density by construction, it's unsurprising DBSCAN collapses — but the analysis stops at "it didn't work" rather than trying 2–3 eps/min_samples combinations to show whether the failure is fundamental to density-based clustering on this data shape, or just this one parameter choice. A small grid search (already half-supported by the dashboard's manual eps/min_samples sliders) would make this a much stronger point.

---

## 5. Findings — Code Quality & Engineering

### 5.1 Architecture: genuinely good

`customer_segmentation_analysis.py` is organized into single-responsibility classes (`DataLoader`, `DataPreprocessor`, `ExploratoryDataAnalysis`, `OptimalClusterFinder`, `ClusteringEngine`, `BusinessIntelligenceEngine`, `VisualizationEngine`, `ReportGenerator`, `SegmentationPipeline`), uses frozen dataclasses for config, `Enum` for categorical constants, and consistent file+console logging. Type hints are present on nearly every function signature, and docstrings follow a consistent Args/Returns convention. This is meaningfully more mature than the typical "one big notebook" clustering project.

### 5.2 Broken quick-start: `generator.py` vs. `genarator.py`

The README's Quick Start says:

```bash
python generator.py
```

and the "Project Structure" section in the same README lists `generator.py`. The actual file in the archive is **`genarator.py`** (missing the "e"). The error message inside `customer_segmentation_analysis.py` even repeats the wrong name: `"Please run generator.py first to create the dataset."` Anyone following the README verbatim on a fresh clone gets an immediate `python: can't open file 'generator.py': No such file or directory`. This is a one-line fix (rename the file, or fix the three references) but it's the first thing a new user or reviewer will hit.

### 5.3 Stated Python version contradicts the pinned dependencies

The README claims **"Python 3.8+."** I checked the actual PyPI metadata for the pinned versions in `requirements.txt`:

| Package (pinned) | `requires_python` (from PyPI) |
|---|---|
| `numpy==2.3.3` | `>=3.11` |
| `pandas==2.3.3` | `>=3.9` |

A genuine Python 3.8 (or even 3.9/3.10) environment cannot install `numpy==2.3.3` at all — `pip install -r requirements.txt` would fail outright, not just behave oddly. The README should say **Python 3.11+**, or the pins should be relaxed if 3.8 support is actually intended.

### 5.4 Duplicated business logic has already drifted between the two scripts

`BusinessIntelligenceEngine` in the analysis script and `BusinessIntelligence` / `ProductRecommendationEngine` in the dashboard both independently implement `categorize_age`, `categorize_income`, `categorize_spending`, and a marketing-strategy generator with the same thresholds (age <30/<50, income <500k/<1M, spending <40/<70). This is a clear DRY violation, and it isn't hypothetical — the two copies **have already diverged**: the analysis script's "high spending" strategy text reads *"Value bundles, loyalty rewards, installment plans, quality assurance"* while the dashboard's reads *"Value bundles, loyalty rewards, quality products at competitive prices."* This is exactly the kind of silent drift duplicated logic produces. Extracting a shared `business_rules.py` module used by both entry points would eliminate the risk entirely.

### 5.5 Fragile cache key in the dashboard

```python
@st.cache_data(show_spinner=False, hash_funcs={pd.Series: lambda x: id(x)})
def perform_clustering(...):
```

Hashing a `pd.Series` by `id()` keys the cache on Python object identity, not content. It happens to work here because the same in-memory `validation_labels` Series object is reused across Streamlit reruns within a session — but it's not a correct general-purpose cache key: two Series with identical values but different identities would be treated as a cache miss, and (in principle) a garbage-collected id could be reused for an unrelated object later in the process. A content hash (e.g. `pd.util.hash_pandas_object(x).values.tobytes()`) would be both correct and still fast at this data size.

### 5.6 Minor reproducibility gap

Every clustering algorithm correctly uses `random_state=self.config.RANDOM_STATE` (42). However, `VisualizationEngine.plot_dendrogram` samples rows with `np.random.choice(...)` against the *global* NumPy RNG, and `np.random.seed(...)` is never called anywhere in `customer_segmentation_analysis.py` (only in `genarator.py`, a separate process). This means the exact 100-row sample drawn for the dendrogram — and therefore the exact dendrogram image — is not reproducible across runs, unlike literally every other output in the pipeline. Passing an explicit `random_state`/`np.random.RandomState(42)` into that one call would close the gap.

### 5.7 Dependency hygiene

- `requirements.txt` reads like a raw `pip freeze` of the whole virtual environment rather than a curated list of direct dependencies — it includes Streamlit's own transitive dependencies (`altair`, `blinker`, `cachetools`, `pydeck`, `tenacity`, `watchdog`, `GitPython`, `toml`, `jsonschema*`, `rpds-py`, etc.). This makes it hard to tell, at a glance, what the project actually needs versus what got pulled in incidentally.
- `scikit-fuzzy==0.5.0` is listed but **never imported anywhere** in any of the three Python files — confirmed via `grep`. Dead dependency; pure install-time risk for zero benefit.
- `pyflakes` flagged unused imports: `sys` (analysis script), `Any`/`Dict`/`matplotlib.pyplot` (dashboard), `seaborn` (generator). Harmless, but easy cleanup.

### 5.8 Version currency (checked against live PyPI, 2026-09-17)

| Package | Pinned | Current latest stable | Gap |
|---|---|---|---|
| pandas | 2.3.3 | 3.0.5 | Major version behind |
| scikit-learn | 1.7.2 | 1.9.1 | Two minors behind |
| streamlit | 1.50.0 | 1.64.0 | 14 minors behind |
| plotly | 6.3.0 | 7.1.0 | Major version behind |
| matplotlib | 3.10.6 | 3.11.2 | One minor behind |
| scipy | 1.16.2 | 1.18.1 | Two minors behind |
| seaborn | 0.13.2 | 0.13.2 | Current |

Nothing here is broken today, and pinning exact versions for reproducibility is a reasonable choice in general — but the pandas 2.x → 3.0 jump in particular carries breaking changes, so a deliberate, tested upgrade sooner rather than later is worth planning for.

### 5.9 No automated tests, no CI

There is no `tests/` directory and no CI configuration of any kind. For an academic capstone this is normal and not a real ding — but it directly undercuts the README's own "production-ready" framing. At minimum, a handful of unit tests around `DataPreprocessor`, the categorization thresholds, and the DBSCAN parameter heuristic would substantially strengthen that claim if it's meant to be taken literally by a hiring audience.

---

## 6. Findings — Documentation

- The README is unusually well-written for this kind of project — clear narrative, an honest "what I'd do differently," and (per §4.1) numerically accurate. That's genuinely above average.
- Unfilled template placeholder: `Thanks to my thesis supervisor [Supervisor Name]` — easy to miss before publishing.
- Framing oversell: phrases like *"No guesswork. Just data-driven insights"* and *"something a business could actually deploy tomorrow"* sit uncomfortably next to §4.2 and §4.3 above. The engineering underneath those claims is solid; the claims themselves are currently a notch ahead of what the evidence supports.
- The Quick Start's third command (`streamlit run customer_segmentation_dashboard.py`) is accurate and I confirmed it works — only the second command (`python generator.py`) is broken (§5.2).

---

## 7. Prioritized Recommendations

**Fix now (cheap, high impact on credibility):**
1. Rename `genarator.py` → `generator.py` (or fix the three README/code references) so the documented quick-start actually works.
2. Add one caveat sentence next to the 96.89% ARI figure clarifying it validates pipeline mechanics against a deliberately well-separated synthetic ground truth, not real-world segment recoverability.
3. Label the product/conversion-rate recommendations as illustrative placeholders, or derive them from something in the data instead of hardcoding brand names and percentages.
4. Correct the README's "Python 3.8+" claim (pinned `numpy` requires 3.11+).

**Fix soon (maintainability):**
5. Extract the duplicated categorization/strategy logic into a single shared module imported by both scripts; reconcile the wording that has already drifted.
6. Replace the `id()`-based cache hash in the dashboard with a content-based hash.
7. Seed the dendrogram's row sampling for full run-to-run reproducibility.
8. Trim `requirements.txt` to actual direct dependencies (or clearly separate "direct" vs. "transitive/pinned-for-reproducibility").
9. Drop the unused `scikit-fuzzy` dependency and the imports flagged by `pyflakes`.

**Worth doing if this becomes a real portfolio centerpiece:**
10. Add a small DBSCAN parameter grid search rather than a single heuristic point estimate, to make the "DBSCAN struggled" conclusion more rigorous.
11. Add a minimal `tests/` suite (preprocessing, categorization thresholds, DBSCAN param logic) and a basic CI workflow if the "production-ready" language is meant literally.
12. Log/report the Davies-Bouldin vs. Silhouette/Calinski-Harabasz disagreement on optimal k instead of silently picking one.

---

## 8. Scorecard

| Dimension | Rating | Notes |
|---|---|---|
| Code architecture | Strong | Clean separation of concerns, good use of dataclasses/enums/logging |
| Reproducibility of stated results | Strong | Independently re-ran and confirmed every headline number |
| Statistical/methodological rigor | Needs work | Validation is close to circular; k-selection disagreement is hidden |
| Business-insight honesty | Needs work | Conversion rates and product picks are fabricated, not derived |
| Documentation quality | Good | Clear and accurate on the numbers, but oversells scope/impact |
| Dependency & environment hygiene | Fair | Version claim contradiction, dead dependency, unpinned rationale |
| Testing / production readiness | Weak | No tests, no CI — fine for academic scope, inconsistent with "production-ready" claim |

**Bottom line:** this is a well-engineered academic clustering project with real, verifiable results — the code is not the problem. The gap between what the code actually demonstrates and what the README claims it demonstrates is the thing worth closing before using this as a "production-ready, hire-me" artifact.

---

## 9. Remediation Log (2026-09-18)

Everything above is preserved as originally written — it's the record of what was found on first pass. This section documents what was actually changed afterward, verified by re-running the pipeline and dashboard rather than by inspection alone.

| # | Finding | Status | What was done |
|---|---|---|---|
| 5.2 | `genarator.py` typo | Fixed | Renamed to `generator.py`. Re-ran it standalone to confirm the quick-start command in the README now works as written. |
| 4.1/4.2 | Circular 96.89% ARI validation | Addressed | Added a caveat next to the figure in the README, and a matching note in `generator.py`'s docstring, explaining that the synthetic ground truth is deliberately well-separated and that the score mainly validates pipeline mechanics. The number itself was re-verified and still holds (0.9689 for K-Means). |
| 4.3 | Fabricated product names / conversion rates | Fixed | Replaced with `business_rules.py`'s `generate_product_suggestions()`. Specific branded products (iPhone Pro Max, MacBook Pro, etc.) and invented conversion percentages are gone. What's left is a generic price-tier category (Premium/Mid-range/Budget) plus one number that's actually computed from the data: `spending_index`, the segment's average Spending Score against the population average. Both the analysis script's log output and the dashboard now carry an explicit "illustrative, not data-derived" disclaimer at the point these suggestions are shown. |
| README §6 | "Python 3.8+" claim | Fixed, and re-checked after §5.8's version bump | Queried PyPI's package metadata directly for the pinned versions rather than assuming. The actual floor is now **Python 3.12+** (driven by numpy 2.5.3 and scipy 1.18.1, both `>=3.12`), not 3.11+ as this report originally found — that was correct for the old pins, not the new ones. README updated to 3.12+. |
| 5.4 | Duplicated, drifted categorization/strategy logic | Fixed | Extracted into a new shared module, `business_rules.py`: the age/income/spending thresholds, `categorize_*` functions, `generate_marketing_strategy`, `segment_priority`, and `generate_product_suggestions`. Both `customer_segmentation_analysis.py` and `customer_segmentation_dashboard.py` now import from it instead of keeping their own copies. The wording that had drifted between the two files (the "high spending, not high income" strategy text) is now identical by construction. |
| 5.5 | `id()`-based cache hash | Fixed | Replaced with `pd.util.hash_pandas_object(x).values.tobytes()`, which hashes on the Series' actual values instead of its object identity. |
| 5.6 | Unseeded dendrogram sampling | Fixed | `VisualizationEngine.plot_dendrogram` now draws its sample with `np.random.RandomState(self.config.RANDOM_STATE)` instead of the global `np.random.choice`, so the sampled rows (and the resulting image) are reproducible across runs. |
| 5.7 | Dependency hygiene (dead `scikit-fuzzy`, unused imports) | Fixed | `scikit-fuzzy` was never imported anywhere and is dropped. Confirmed-unused imports removed: `sys` (analysis script), `matplotlib.pyplot`, `Any`, `Dict` (dashboard), `seaborn` (generator). Re-ran `pyflakes` and `vulture` across all four scripts after every change; both come back clean of unused imports and dead code as of this log. |
| 5.8 | Stale/unverified pinned versions | Re-verified and updated | Rebuilt `requirements.txt` down to the 8 actual direct dependencies (numpy, pandas, scikit-learn, scipy, matplotlib, seaborn, plotly, streamlit — confirmed by grepping every `import` line across the four scripts). Installed the current latest stable release of each with `uv`, then ran `generator.py`, the full `customer_segmentation_analysis.py` pipeline, and the Streamlit dashboard (headless, health-checked) against them in a venv containing nothing else. All three ran clean, including pandas 3.0.6 — the jump this report flagged as carrying breaking changes turned out not to affect this codebase in practice, though that was worth confirming rather than assuming either way. |
| 10 | DBSCAN single-point heuristic | Fixed, and changed the actual finding | `_optimize_dbscan_params` now runs a small grid search (4 eps percentiles × 2 min_samples values) around the k-distance estimate instead of trusting one point. Result: DBSCAN now finds all 5 segments (Silhouette 0.4647, Davies-Bouldin 1.4948, Calinski-Harabasz 2578, ~11% noise, ARI 0.8289) where the old single-guess heuristic collapsed to 1 cluster. This is a genuine change to the pipeline's behavior, not just a caveat — the README's algorithm comparison table and DBSCAN writeup were both updated with these re-measured numbers. |
| 12 | Silent Davies-Bouldin/Silhouette disagreement on optimal k | Fixed | `_determine_optimal_k` now logs a note when the Davies-Bouldin-suggested k differs from the Silhouette/Calinski-Harabasz consensus, instead of picking the latter without comment. |
| — | Two pieces of dead code not caught in the original pass | Fixed | `vulture` turned up an unused `spending_category` field on the dashboard's `ClusterInsight` dataclass (removed), and an `is_valid` property defined on both `ClusteringResult` classes but never called anywhere. In the analysis script it's now actually used, to skip business-insight generation for a degenerate result instead of silently profiling a meaningless single cluster. In the dashboard it was removed, since `warning_message` already covers the same cases with clearer per-case text. Also wired up `additional_metrics`, a field that was being populated (DBSCAN's chosen eps/min_samples/noise count, GMM's BIC/AIC/convergence) but never read anywhere — it now appears in the summary report. |
| 11 | No automated tests | Partially addressed | Added a small `tests/` suite covering `business_rules.py` (the categorization thresholds, marketing strategy branches, and product suggestions) — see `tests/README.md`. No CI workflow was added; that's still open if this is meant to be a literal "production-ready" artifact. |
| 6 | Unfilled `[Supervisor Name]` placeholder | Fixed | Removed the bracket; the sentence now reads "Thanks to my thesis supervisor for guidance" without a fabricated or placeholder name. |
| 6 | Framing oversell ("No guesswork", DBSCAN "failure" language) | Softened | Reworded in the README to acknowledge which parts are algorithm output versus the author's own marketing judgment, and updated the DBSCAN-related claims throughout to match the grid-search finding above. |

**What's still open:** a CI workflow (part of item 11) was not added. The core numeric claims in the README (segment sizes, all four algorithms' metrics, the visualization file count) were re-verified against a fresh run rather than carried over from the original text.
