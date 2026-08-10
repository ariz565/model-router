# Metadata-Driven Synthetic Data Generation

Generate privacy-safe synthetic data that is **structurally usable**, not just
statistically plausible.

> **Verification status.** 131 tests, all executed, running against real SQLite
> databases and in-memory fixtures created per test — discovery, profiling,
> dependency ordering, datetime/chronology handling, generation (both
> `StatisticalEngine` and `CopulaEngine`), reconstruction, TSTR, and all five
> validation layers. Two exceptions, both named rather than hidden: the
> **RCTGAN adapter**, which needs a deep-learning stack not installed here, so
> it is compile-checked and *never executed*; and the **HTTP request-wiring
> tests** (`tests/test_synthetic_http.py`), gated on `fastapi`/`pydantic`, which
> skip in this dev environment and run in any environment with the `[server]`
> extra installed. Every non-HTTP engine, validation, and orchestrator path is
> the verified one. Details in §6 and §5.

---

## 1. The problem this solves

The naive approach is to train a generative model on a production table and sample
from it. The values come out realistic and the **dataset comes out broken**: joins
fail, foreign keys dangle, and business processes reach states that could never
occur. The model learned the data and knew nothing about the *system* the data
belonged to.

So the generative model is treated as one replaceable component, and the platform
around it does the work:

> **First understand the system. Then learn the data. Then generate something that
> preserves both.**

## 2. The five stages

```
        ┌──────────────────────────────────────────────────────────┐
        │  SOURCE DATABASE  (read only here, and nowhere else)     │
        └───────────────────────────┬──────────────────────────────┘
                                    │  discovery.py — the ONLY module
                                    │  that touches real rows
   ① DISCOVER   ┌────────────────────▼─────────────────────────────┐
                │ schema · PK/FK · constraints · nullability ·      │
                │ relationship cardinality (MEASURED, not assumed) │
                └────────────────────┬─────────────────────────────┘
   ② UNDERSTAND ┌────────────────────▼─────────────────────────────┐
                │ numeric quantiles · categorical frequencies ·    │
                │ null rates · correlations · outlier COUNTS       │
                │        ── the privacy boundary lives here ──     │
                └────────────────────┬─────────────────────────────┘
                ┌────────────────────▼─────────────────────────────┐
                │ DEPENDENCY GRAPH → topological generation order  │
                │ cycles broken deliberately · self-refs handled   │
                └────────────────────┬─────────────────────────────┘
   ③ GENERATE   ┌────────────────────▼─────────────────────────────┐
                │  GeneratorEngine (pluggable)                     │
                │  statistical · copula · rctgan · <yours>         │
                │  → UNLINKED tables. Knows nothing about keys.    │
                └────────────────────┬─────────────────────────────┘
   ④ RECONSTRUCT┌────────────────────▼─────────────────────────────┐
                │ mint PKs → remap FKs from the real key pool →    │
                │ uniqueness → nullability → CHECK reporting →     │
                │ chronology enforcement (opt-in, see §4b)         │
                │        ── deterministic, not probabilistic ──    │
                └────────────────────┬─────────────────────────────┘
   ⑤ VALIDATE   ┌────────────────────▼─────────────────────────────┐
                │ statistical · structural · privacy · quality ·   │
                │ TSTR (opt-in, see §5b)                           │
                │ → report: rules passed / critical / warnings     │
                └──────────────────────────────────────────────────┘
```

**The order is the architecture.** Each stage consumes only what earlier stages
produced and never reaches backwards. Generation cannot see the source (it gets
profiles). Reconstruction cannot see the engine (it gets rows). Validation cannot
see the source either. That is what makes each stage replaceable, and it is what
makes "no real data leaves the secure environment" checkable by reading interfaces
rather than auditing the whole platform.

## 3. Two ideas doing most of the work

### Generation is probabilistic; reconstruction is deterministic

An engine cannot guarantee a join works — that is not a statistical property. So it
doesn't try. It emits unlinked tables with placeholder keys, and
`reconstruction.py` then assigns fresh primary keys and remaps every foreign key by
**sampling from the parent's actual generated key pool**. A child can only
reference a key that exists, because the pool it draws from *is* the set of keys
that exist. Joins are correct by construction, not by luck.

Cardinality is honored too, not just referential integrity. Uniform random parent
assignment satisfies every FK constraint and still describes an impossible
business — one customer with 40,000 addresses. One-to-one gets a permutation;
one-to-many gets a skewed draw, because real fan-out is skewed.

### The generative model is a port, not the product

`ports.GeneratorEngine` is a four-method Protocol. An engine learns per-table
distributions and is explicitly **not** responsible for keys, constraints,
ordering, or linking. Swapping RCTGAN for CTGAN, TVAE, or something new means
writing one file; nothing else changes.

`EngineCapabilities` makes each engine **declare** what it preserves, so the
validation report can label correlation drift as *expected for this engine*
rather than flagging it. Grading a component against a capability it disclaims is
how a report earns a reputation for false alarms.

## 4. The privacy boundary

Profiles carry **shapes, never rows** — quantiles, frequencies, null rates, outlier
*counts*. One deliberate exception: categorical labels are real, because a
synthetic `region` column full of invented labels breaks every downstream query.

That exception is bounded by a rule in `profiling.py`, and it is the single most
important line in this module:

```
a text column becomes CATEGORICAL only if its distinct count is
    below max_categories (50)        AND
    below max_distinct_fraction (20%) of its rows
otherwise it stays TEXT and NO labels are retained at all
```

Both conditions are required. The absolute cap alone would retain 40 labels from a
45-row table (near-unique, therefore identifying). The ratio alone would retain
50,000 labels from a million rows. `email` and free-text notes fail this test and
are never characterized — there is a test asserting exactly that.

## 4b. Datetimes and chronology

A timestamp is profiled as **epoch-second quantiles** — the same quantile-ladder
machinery `NumericProfile` already has, since a timestamp is a monotonic numeric
quantity once converted (`datetimes.py`). `date_only` is detected from the DATA
(every observed instant at midnight), never from the declared SQL type, so a
`TIMESTAMP` column that has only ever held date-only values still round-trips as a
bare date. `median_gap_seconds` — the median gap between chronologically adjacent
observed values — drives realistic-looking corrections rather than an arbitrary
constant offset.

**Chronology is enforced only when a caller says so — `ChronologyConstraint`,
explicit and opt-in**, matching this platform's standing rule (already applied to
composite UNIQUE and CHECK expressions): nothing infers business semantics from a
schema. Two shapes:

- **Same-table**: `orders.ship_date >= orders.order_date` (same row).
- **Cross-table, via a foreign key**: `orders.order_date >= customer.created_at`.
  `via_fk_column` disambiguates when a child has more than one FK to the same
  parent; left unset with more than one candidate, the constraint is reported
  unenforceable rather than guessed.

A violation is corrected forward by adding a realistically-sized, exponentially
sampled gap — never truncated back onto the same instant, and never onto the SAME
calendar day for a date-only column (truncating a small correction to a bare date
can land back on the day being corrected against, which would silently re-violate
the constraint; the fix floors to the next calendar day). Multiple constraints that
chain (`ship_date` depends on `order_date`, which itself depends on
`customer.created_at`) are applied in **dependency order, not caller-supplied
order** — the reconstructor topologically sorts them so a column is finalized
before anything reads it as a reference point.

## 5. Validation: four always-on layers, three severities

| Layer | Question | Method | Failure |
|---|---|---|---|
| **Statistical** | does it *look* like the source? | KS (numeric), PSI (categorical), correlation | WARNING |
| **Structural** | can it be *queried* like the source? | referential integrity, uniqueness, nullability, schema | **CRITICAL** |
| **Privacy** | does it *leak* the source? | k-anonymity-aware exact match, DCR vs. baseline | **CRITICAL** |
| **Quality** | is it *realistically imperfect*? | missingness preserved, value ranges, duplicates | WARNING / INFO |

These four always run. A 5th, opt-in layer — **TSTR**, does a model trained on
this actually work? — is covered on its own in §5b, because unlike these four it
needs a caller-specified target column and (optionally) real rows.

A run's verdict is **`critical_failures == 0`, never a score threshold** — an
aggregate score can average a privacy leak into a comfortable 94%.

Three design decisions here were forced by *running* the framework and finding it
crying wolf. All three are cases where the naive implementation produces confident
numbers that mean nothing:

**KS and PSI are routed by column kind, and identifiers are excluded.** KS on a
categorical column and PSI on a continuous one both look fine and mean nothing.
Foreign-key columns are excluded from distribution checks entirely: their values
are assigned by reconstruction from a *different key space* than the source's, so
comparing them reports an enormous false shift. Relationship shape is validated by
the structural layer, where it belongs.

**Exact-match leakage is k-anonymity-aware.** A synthetic row matching a real one
is only a disclosure if that real row was rare. On a low-cardinality table,
`(North, 42, NULL)` collides by pure chance and may be shared by 30 real
customers — flagging it is a false positive, and a check that fires every run gets
switched off. Only matches against real signatures occurring ≤ *k* times (default
1) are critical.

**Nearest-neighbour distance is measured against the real data's own spacing**
(Distance to Closest Record vs. a real→real baseline), not an absolute threshold.
"Closer than 2% of the range" is meaningless without knowing how dense the real
data is: in a one-column, 200-row table real records are already ~0.5% apart, so
every synthetic value trips it. If synthetic rows are no closer to real records
than real records already are to *each other*, there is no excess disclosure risk.
Memorization is exactly the case where that ratio collapses.

**The privacy layer is tested adversarially.** A "memorizing engine" that emits real
rows verbatim must produce critical failures, and does — DCR ratio 0.00×,
exact-match firing. Independent data passes at 871×. A privacy check that has never
been shown to *fail* is decorative.

## 5b. TSTR — the one layer that measures ML utility, not resemblance

The other four layers can all pass while a model trained on the synthetic data
still fails on real data: two columns can each match their marginal distribution
perfectly while their *joint* relationship is destroyed — exactly
`StatisticalEngine`'s declared limitation. Statistical fidelity does not imply ML
utility, and the only way to know the difference is to measure it.

**Train-Synthetic-Test-Real**, `validation/tstr.py`: train a k-nearest-neighbours
model on the *synthetic* data, evaluate it on a *real* held-out split, and compare
that score against training the same model on a *real* split and evaluating on the
same real holdout (**TRTR**, the baseline). k-NN in pure stdlib — no numpy, no
scikit-learn — because it is the simplest model that is genuinely a model: a
distance computation and an aggregation, for both regression (R², k-neighbour mean)
and classification (accuracy, k-neighbour majority vote).

Like `ChronologyConstraint`, this is **explicit and opt-in per `(table,
target_column)`** — nothing infers which column is an ML target — and it needs real
rows in the process, gated by its *own* toggle
(`include_real_rows_for_tstr`/`verify_ml_utility_against_real_rows`), independent
of the privacy layer's real-rows toggle. Absent real rows, a task is reported
`tstr_unverified`, never silently skipped — the same discipline the privacy layer
already applies when real rows aren't supplied.

Measured on a fixture with a genuine `income ≈ f(age)` relationship: a model
trained on `StatisticalEngine`'s independent output scores **R² = −0.11** against
a real baseline of **0.98** (worse than predicting the mean — the relationship is
gone). The identical setup through `CopulaEngine` scores **R² = 0.98**, a utility
ratio of **1.00**. That gap is what "does this synthetic data solve the ML
engineer's problem" means, made measurable.

## 6. Engines

### `statistical` — the default, zero dependencies

Pure stdlib. Samples each column independently from its profiled distribution:
inverse-transform over the empirical quantile ladder for numerics, frequency
sampling for categoricals, and reproduced null rates.

Preserves: per-column distributions, null rates, category sets, value ranges,
integrality, and the source's own boolean *representation* (a column stored as
`0`/`1` emits `0`/`1`, not Python `True`/`False` — otherwise `WHERE active = 1`
matches nothing). Deterministic: same seed ⇒ byte-identical output, which is what
makes a dataset usable as a test fixture.

**Does not preserve cross-column correlations.** Each column is drawn
independently, so `age` and `product_category` are each valid and unrelated to one
another. This is declared in `capabilities`, and it is the specific gap a
CTGAN/RCTGAN-class model — or `copula`, below — exists to close. It is also why
this engine is genuinely sufficient for software testing, SQL validation, and
sandbox seeding — three of the six consumers in the source architecture — none of
which needs a learned joint distribution.

Quantile sampling **interpolates between adjacent quantiles**. That was a privacy
fix prompted by this platform's own validation: reading the ladder by nearest rank
returns a value that literally occurred in the source, and combined with two
low-cardinality columns that reconstructed whole real records. Interpolating keeps
the property that mattered — values stay inside the observed range — without
reproducing observations verbatim.

### `copula` — zero dependencies, preserves correlation

Also pure stdlib (`mathstats.py`: `erf`-based normal CDF, Acklam's algorithm for
the inverse, Cholesky decomposition with an automatic ridge-jitter fallback for a
not-quite-positive-definite empirical correlation matrix). A **Gaussian copula**:
draw correlated standard-normal noise via the Cholesky factor of the table's
recorded Pearson correlations, map each coordinate through `normal_cdf`, then
through *that column's own* empirical quantile ladder — so every individual
marginal is exactly as faithful as `statistical`'s, and only the JOINT relationship
changes.

**Scope, stated plainly:** only numeric and boolean columns are copula-modelled —
the only kinds `profiling.py` ever records a correlation for in the first place.
Correlation is preserved *within* a table, never across tables (that is
`reconstruction.py`'s job, via FK assignment, not an engine's). Boolean
correlations are preserved but **attenuated**, not exact — thresholding a
continuous correlated normal into a binary value is a known correlation-shrinking
operation (the same reason point-biserial correlation is weaker than the
underlying continuous relationship). Measured on a fixture with true correlations
of 0.99 (numeric-numeric) and 0.75 (boolean-numeric): the copula recovers ~0.99 and
~0.61 respectively, against `statistical`'s ~0.0 for both.

A table with fewer than two eligible columns falls back to fully independent
sampling automatically — there is nothing to correlate, not a special case to
avoid.

### `rctgan` — written, **never executed**

Requires `torch` plus an SDV-family package. Trains one model per table, excludes
identifier columns from training (a PK is a label, not a distribution, and letting
a GAN learn one risks reproducing real identifiers).

It is the one engine that **breaks the profile-only contract**: a GAN needs real
rows to train a discriminator, so it takes a `DataSource` at construction. That is a
visible exception rather than a quiet widening of the port, and it has an
operational consequence — an RCTGAN-backed run **must execute inside the customer's
secure environment**.

Risks the surrounding platform is expected to catch, none of which the adapter can
address alone: *memorization* (→ the privacy layer), *mode collapse* (→ PSI and
distinct counts), and *non-determinism* (GPU nondeterminism means its output is not
a stable fixture; `capabilities.requires_training` says so).

## 7. HTTP surface

All under `/v1/synthetic`, gated on `org:manage` (owner-level) for anything that
reads a schema, `org:read` for reports.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/sources` | Registered source **names** — never DSNs or paths |
| `GET` | `/sources/{name}/metadata` | Stage 1 alone: schema, keys, cardinality, dependency graph |
| `POST` | `/generate` | Run the pipeline; returns the run record |
| `GET` | `/runs` | Run history (summaries) |
| `GET` | `/runs/{id}` | One full run record |
| `GET` | `/runs/{id}/report` | The validation report on its own |

**A source database is never accepted over HTTP.** The endpoint takes a
server-configured source *name*, resolved against `app.state.synthetic_sources`. If
a caller could POST a connection string, this would be a server-side-request-forgery
primitive that reads arbitrary reachable databases and returns their profiled
contents.

```python
from modelrouter.synthetic import SqliteDataSource
app.state.synthetic_sources = {"crm": SqliteDataSource("/secure/crm.db")}
```

`POST /generate`'s body carries the same opt-in knobs as `GenerationConfig`,
translated one-for-one: `engine` (`statistical` | `copula` | `rctgan`),
`chronology_constraints` (a list mirroring `ChronologyConstraint` field-for-field),
`tstr_tasks` (mirroring `TSTRTaskConfig`), and a *third*, independent real-rows
toggle, `verify_ml_utility_against_real_rows` — separate from
`verify_privacy_against_real_rows`, because a caller may want one without the
other. Every one of these fields is validated by a Pydantic model with
`extra="forbid"`, so a typo in a constraint or task is a `422`, never a silently
ignored field.

## 8. Removability

Nothing in the codebase imports this package except two lines in `server.py`.
Delete `synthetic/`, those lines, and the `[synthetic]` extra — nothing else
changes behavior. Same property as `identity/sso/`. `synthetic/http.py` is the only
file here that imports FastAPI, so the package stays usable as a library.

## 9. Not built — named, not implied

- **CHECK constraint enforcement.** Reported as unverified, never interpreted. A
  partial SQL expression evaluator is worse than none: it would silently pass the
  expressions it failed to parse, which is indistinguishable from having verified
  them.
- **Composite-key foreign keys.** The FK model is single-column — this also bounds
  chronology enforcement across tables and TSTR's parent-key lookups, both of which
  require a single-column parent key.
- **Durable run history.** Runs are in-memory; a restart loses them. Persisting a
  run record (metadata + profile + full report) is a real storage decision, not a
  detail to guess at.
- **Asynchronous generation.** `POST /generate` runs synchronously and will exceed
  an HTTP timeout on a large database. A half-built job queue that loses runs on
  restart would be worse; scope with `tables`/`scale` until it exists.
- **Cross-table joint distributions** are approximated by per-table copula/marginal
  learning plus deterministic relinking, not modelled jointly. `ChronologyConstraint`
  narrows the gap for *ordering* specifically, but a child table's numeric columns
  still don't correlate with their parent's.
- **Copula correlation for categorical/text columns.** Only numeric and boolean
  pairs are copula-modelled (see §6) — the only kinds `profiling.py` ever records a
  correlation for. A categorical column's relationship to a numeric one (e.g.
  `region` and `income`) is not preserved by either engine today.
- **TSTR feature encoding is numeric/boolean-only.** A categorical or text feature
  column is not usable as a k-NN input without a real encoding scheme (one-hot,
  embeddings); today a caller supplying `feature_columns` must restrict them to
  numeric/boolean, and auto-selection already does this.
- **Membership-inference testing** beyond DCR and exact-match.
- **Known limitation of the DCR check:** on highly regular or lattice-like numeric
  data (an arithmetic sequence, heavily rounded values), real→real spacing is
  artificially wide and interpolated synthetic values can look anomalously close.
  The ratio is a heuristic, not a proof.
