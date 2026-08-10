"""The validation report — Figure 4's "Rules Passed / Critical Failures /
Warnings" verdict, plus the overall statistical summary.

**Three severities, because two are not enough.** A binary pass/fail forces every
finding into one of two boxes, and the result is either a report that fails on
cosmetic drift (so people stop reading it) or one that passes on a broken join (so
people stop trusting it). The distinction that actually matters in practice:

- **CRITICAL** — the dataset is not usable for its purpose. A dangling foreign key,
  a duplicated primary key, a synthetic record that matches a real one. These are
  correctness and privacy failures, and any one of them fails the run.
- **WARNING** — usable, with a caveat worth knowing. Moderate distribution drift,
  an unverifiable CHECK constraint, correlation loss from an engine that never
  claimed to preserve correlations.
- **INFO** — measured and recorded, no judgement. The numbers a trend line is
  built from.

**A run's verdict is `critical_failures == 0`, never a score threshold.** An
aggregate score can average a catastrophic privacy leak into a comfortable 94%,
which is precisely how a validation framework becomes decorative. Figure 4 shows
"Critical Failures: 0" as its own field for that reason.

**Expected limitations are not failures.** When an engine declares it does not
preserve correlations, correlation drift is reported as INFO with that engine named
— not as a WARNING. Flagging a system for not doing something it explicitly never
claimed to do is how false alarms accumulate until the report is ignored.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

__all__ = [
    "Severity", "SEVERITY_CRITICAL", "SEVERITY_WARNING", "SEVERITY_INFO",
    "CheckResult", "LayerReport", "ValidationReport",
    "LAYER_STATISTICAL", "LAYER_STRUCTURAL", "LAYER_PRIVACY", "LAYER_QUALITY",
    "LAYER_TSTR",
]

SEVERITY_CRITICAL = "critical"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

Severity = str

LAYER_STATISTICAL = "statistical"
LAYER_STRUCTURAL = "structural"
LAYER_PRIVACY = "privacy"
LAYER_QUALITY = "quality"
# A 5th layer, opt-in (see `validation/tstr.py`): whether a model trained on
# the synthetic data actually generalizes to real data, not just whether the
# marginal distributions resemble it.
LAYER_TSTR = "tstr"


@dataclass(frozen=True)
class CheckResult:
    """One rule, one verdict.

    `passed` and `severity` are independent: a passing check still carries the
    severity it WOULD have had, so a report can say "342 of 356 rules passed, and
    the 14 that didn't are all warnings" — which is the difference between a
    dataset to investigate and one to discard.

    `metrics` holds the numbers behind the verdict (a KS statistic, a PSI, a
    distance) so a reader never has to trust a bare boolean."""

    layer: str
    name: str
    passed: bool
    severity: Severity = SEVERITY_WARNING
    table: str | None = None
    column: str | None = None
    detail: str = ""
    metrics: dict = field(default_factory=dict)

    @property
    def is_critical_failure(self) -> bool:
        return not self.passed and self.severity == SEVERITY_CRITICAL

    def as_dict(self) -> dict:
        return {
            "layer": self.layer, "name": self.name, "passed": self.passed,
            "severity": self.severity, "table": self.table, "column": self.column,
            "detail": self.detail, "metrics": self.metrics,
        }


@dataclass(frozen=True)
class LayerReport:
    layer: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed_count(self) -> int:
        return sum(1 for c in self.checks if c.passed)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if c.is_critical_failure]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == SEVERITY_WARNING]

    @property
    def ok(self) -> bool:
        return not self.critical_failures

    def as_dict(self) -> dict:
        return {
            "layer": self.layer,
            "total": len(self.checks),
            "passed": self.passed_count,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "ok": self.ok,
            "checks": [c.as_dict() for c in self.checks],
        }


@dataclass(frozen=True)
class ValidationReport:
    layers: list[LayerReport] = field(default_factory=list)
    engine: dict = field(default_factory=dict)     # EngineCapabilities.as_dict()
    generated_at: datetime | None = None

    @property
    def all_checks(self) -> list[CheckResult]:
        return [check for layer in self.layers for check in layer.checks]

    @property
    def total_rules(self) -> int:
        return len(self.all_checks)

    @property
    def rules_passed(self) -> int:
        return sum(1 for c in self.all_checks if c.passed)

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.all_checks if c.is_critical_failure]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.all_checks if not c.passed and c.severity == SEVERITY_WARNING]

    @property
    def trustworthy(self) -> bool:
        """The verdict. Zero critical failures — deliberately NOT a pass-rate
        threshold; see the module docstring on why an aggregate score can hide a
        privacy leak."""
        return not self.critical_failures

    @property
    def pass_rate(self) -> float:
        """Reported for readability, never used as the verdict. 1.0 for an empty
        report rather than a ZeroDivisionError — though `total_rules == 0` is
        itself worth noticing, which is why the field is exposed too."""
        return self.rules_passed / self.total_rules if self.total_rules else 1.0

    def summary(self) -> dict:
        """Figure 4's "Overall Statistical Summary" — averages computed from the
        raw per-check metrics, never from other averages.

        Only checks that actually recorded a given metric contribute to its mean,
        so a dataset with three numeric and twenty categorical columns doesn't
        dilute its average KS with seventeen zeroes."""
        ks_values = [
            c.metrics["ks_statistic"] for c in self.all_checks
            if "ks_statistic" in c.metrics
        ]
        psi_values = [c.metrics["psi"] for c in self.all_checks if "psi" in c.metrics]
        correlation_deltas = [
            c.metrics["correlation_delta"] for c in self.all_checks
            if "correlation_delta" in c.metrics
        ]
        distribution_checks = [
            c for c in self.all_checks
            if c.layer == LAYER_STATISTICAL and ("ks_statistic" in c.metrics or "psi" in c.metrics)
        ]
        matched = sum(1 for c in distribution_checks if c.passed)
        return {
            "avg_ks_statistic": (sum(ks_values) / len(ks_values)) if ks_values else None,
            "avg_psi": (sum(psi_values) / len(psi_values)) if psi_values else None,
            # 1.0 minus mean absolute deviation, so it reads like Figure 4's
            # "Correlation Match: 0.97" rather than as an error term.
            "correlation_match": (
                1.0 - (sum(correlation_deltas) / len(correlation_deltas))
                if correlation_deltas else None
            ),
            "distributions_matched_fraction": (
                matched / len(distribution_checks) if distribution_checks else None
            ),
            "distribution_checks": len(distribution_checks),
        }

    def as_dict(self) -> dict:
        return {
            "trustworthy": self.trustworthy,
            "rules_passed": self.rules_passed,
            "total_rules": self.total_rules,
            "pass_rate": self.pass_rate,
            "critical_failures": [c.as_dict() for c in self.critical_failures],
            "warning_count": len(self.warnings),
            "summary": self.summary(),
            "engine": self.engine,
            "layers": [layer.as_dict() for layer in self.layers],
            "generated_at": (self.generated_at or datetime.now(timezone.utc)).isoformat(),
        }
