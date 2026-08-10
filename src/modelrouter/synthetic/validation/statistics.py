"""The statistical primitives behind Figure 4's validation layer: the
two-sample Kolmogorov–Smirnov test, the Population Stability Index, and
correlation agreement.

**Pure stdlib, no SciPy.** Same reasoning as the profiler: validation runs beside
generation inside the customer's environment, and a stage with a heavy dependency
tree is a stage that gets deployed somewhere more convenient. Both statistics are
a few dozen lines and are fully specified, so implementing them is not
reimplementing a library — it is avoiding a 40 MB dependency for two functions.

**Why both KS and PSI**, rather than picking one:

- **KS** is a proper hypothesis test over *continuous* distributions: it compares
  empirical CDFs, is sensitive to shifts anywhere in the distribution, and yields
  a p-value. It is meaningless for categorical data, where there is no ordering to
  build a CDF from.
- **PSI** is a binned divergence used for exactly the categorical case (and for
  drift monitoring generally). It has no p-value; it is read against conventional
  thresholds.

Applying KS to a categorical column, or PSI alone to a continuous one, is the
common mistake that produces confident-looking numbers that mean nothing. Figure 4
shows both for that reason, and `checks.py` routes each column by its profiled
kind rather than letting a caller choose wrongly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "KsResult", "PsiResult",
    "ks_two_sample", "kolmogorov_p_value", "population_stability_index",
    "PSI_NO_SHIFT", "PSI_MODERATE_SHIFT",
    "DEFAULT_KS_ALPHA",
]

# Conventional PSI reading, from credit-risk practice where the index originates:
#   < 0.10  no material shift
#   0.10-0.25 moderate shift, worth investigating
#   > 0.25  significant shift
PSI_NO_SHIFT = 0.10
PSI_MODERATE_SHIFT = 0.25

# The KS null hypothesis is "these came from the same distribution", so a p-value
# ABOVE alpha is the passing outcome — the opposite of the usual reading, and the
# single easiest thing to get backwards here.
DEFAULT_KS_ALPHA = 0.05

_PSI_EPSILON = 1e-6         # keeps ln() finite when a bin is empty on one side


@dataclass(frozen=True)
class KsResult:
    statistic: float
    p_value: float
    sample_sizes: tuple[int, int]

    def passes(self, alpha: float = DEFAULT_KS_ALPHA) -> bool:
        """True when we CANNOT reject "same distribution" — i.e. p > alpha.

        Worth being explicit that this is not proof of similarity: failing to
        reject a null is weak evidence, and with small samples KS fails to reject
        almost anything. `sample_sizes` is carried so a report can say so rather
        than presenting a pass from 12 rows as equivalent to a pass from 120,000."""
        return self.p_value > alpha

    def as_dict(self) -> dict:
        return {
            "statistic": self.statistic, "p_value": self.p_value,
            "sample_sizes": list(self.sample_sizes),
        }


@dataclass(frozen=True)
class PsiResult:
    psi: float
    bin_count: int

    @property
    def shift(self) -> str:
        if self.psi < PSI_NO_SHIFT:
            return "none"
        return "moderate" if self.psi < PSI_MODERATE_SHIFT else "significant"

    def passes(self, threshold: float = PSI_MODERATE_SHIFT) -> bool:
        return self.psi < threshold

    def as_dict(self) -> dict:
        return {"psi": self.psi, "bin_count": self.bin_count, "shift": self.shift}


def kolmogorov_p_value(lam: float) -> float:
    """The asymptotic Kolmogorov distribution:

        Q(λ) = 2 · Σ_{k=1..∞} (−1)^(k−1) · exp(−2k²λ²)

    Summed until terms stop contributing, with a hard iteration cap so a
    pathological λ can never spin. Very small λ is short-circuited to 1.0: the
    series converges to 1 there but does so slowly and with cancellation error, and
    "the distributions are indistinguishable" is the correct answer anyway.

    Asymptotic rather than exact, and that limitation is real: for very small
    samples the true p-value differs. `KsResult.sample_sizes` exists so a report
    can flag a small-sample verdict instead of over-trusting it."""
    if lam <= 0:
        return 1.0
    if lam < 0.04:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * (k ** 2) * (lam ** 2))
        total += ((-1) ** (k - 1)) * term
        if term < 1e-12:
            break
    return max(0.0, min(1.0, 2.0 * total))


def ks_two_sample(real: list[float], synthetic: list[float]) -> KsResult:
    """Two-sample KS by walking both sorted samples once — O(n log n) for the
    sorts, O(n) for the walk.

    An empty sample on either side yields `statistic=0.0, p_value=1.0` rather than
    raising: an all-NULL column is a legitimate thing to encounter, and the honest
    reading is "nothing to compare", which the caller distinguishes via
    `sample_sizes` rather than by catching an exception."""
    if not real or not synthetic:
        return KsResult(statistic=0.0, p_value=1.0, sample_sizes=(len(real), len(synthetic)))

    first = sorted(real)
    second = sorted(synthetic)
    n1, n2 = len(first), len(second)
    i = j = 0
    cdf1 = cdf2 = 0.0
    statistic = 0.0

    while i < n1 and j < n2:
        # Advance whichever sample is behind; on a tie advance BOTH, so the
        # comparison is made after all copies of an equal value are consumed.
        # Advancing only one would report a spurious gap at every tie — which is
        # every discrete-valued column.
        if first[i] <= second[j]:
            value = first[i]
            while i < n1 and first[i] == value:
                i += 1
                cdf1 = i / n1
            while j < n2 and second[j] == value:
                j += 1
                cdf2 = j / n2
        else:
            value = second[j]
            while j < n2 and second[j] == value:
                j += 1
                cdf2 = j / n2
        statistic = max(statistic, abs(cdf1 - cdf2))

    effective_n = math.sqrt((n1 * n2) / (n1 + n2))
    return KsResult(
        statistic=statistic,
        p_value=kolmogorov_p_value(effective_n * statistic),
        sample_sizes=(n1, n2),
    )


def population_stability_index(
    real_frequencies: dict[str, float], synthetic_frequencies: dict[str, float],
) -> PsiResult:
    """PSI over the UNION of both category sets:

        PSI = Σ (a_i − e_i) · ln(a_i / e_i)

    The union matters. Binning over only the real categories would score a
    synthetic dataset that invented three new categories as perfect, and binning
    over only the synthetic ones would hide categories it dropped entirely. Those
    are the two failure modes worth catching, so both must be in the sum — a
    category present on one side and absent on the other contributes a large term
    via the epsilon floor, which is the intended behavior."""
    categories = sorted(set(real_frequencies) | set(synthetic_frequencies))
    if not categories:
        return PsiResult(psi=0.0, bin_count=0)

    total = 0.0
    for category in categories:
        expected = max(real_frequencies.get(category, 0.0), _PSI_EPSILON)
        actual = max(synthetic_frequencies.get(category, 0.0), _PSI_EPSILON)
        total += (actual - expected) * math.log(actual / expected)
    return PsiResult(psi=total, bin_count=len(categories))


def frequencies_from_values(values: list) -> dict[str, float]:
    """Observed relative frequencies, NULLs excluded.

    Excluded because missingness is validated separately (Figure 4's Data Quality
    layer): folding NULLs in as a pseudo-category would let a matching null rate
    mask a genuine shift in the real categories, and vice versa."""
    counts: dict[str, int] = {}
    for value in values:
        if value is None:
            continue
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if not total:
        return {}
    return {key: count / total for key, count in counts.items()}
