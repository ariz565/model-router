# Laya Decision

`laya_decision` is a production wrapper for the upstream `laya` and
`laya-mlx` runtimes. It intentionally does not use the package name `laya`,
because that would shadow the official runtime.

It provides:

- validated `choice`, `score`, and `noul` contracts;
- automatic English, multilingual, and typed-workflow checkpoint routing;
- Transformers/PyTorch and Apple-Silicon MLX backends;
- lazy loading, selective preload, attach, unload, and upstream LRU behavior;
- pinned/custom checkpoint maps and runtime provenance;
- bounded state, question, and choice cardinality controls;
- sync and concurrency-bounded async clients;
- routing inspection without model inference;
- per-checkpoint, question-shape, and domain temperature calibration;
- local-first fallback cascades with explicit abstention reasons;
- reviewable routing, guardrail, moderation, triage, observability, invoice,
  customer-service, and security question presets;
- state hashing, safe telemetry, and signed evidence records; and
- dependency/readiness health reporting.

## Install

Standard Laya works on supported PyTorch platforms:

```bash
pip install laya
```

Laya-MLX requires Apple Silicon, macOS 14+, and Python 3.11+:

```bash
pip install laya-mlx
```

The model checkpoints are downloaded by their respective upstream runtimes.
Pin revisions or mirror approved artifacts for reproducible production builds.

## Transformers quickstart

```python
from laya_decision import BackendConfig, LayaBackend, LayaClient, triage_questions

backend = LayaBackend(BackendConfig(
    device="cuda",
    preload=("english", "multilingual"),
    max_loaded=2,
))
client = LayaClient(backend)

result = client.predict(
    state={"message": "I was charged twice. Please refund it."},
    questions=triage_questions(),
)
print(result.routing.checkpoint)
print(result.answers["category"])
```

## MLX quickstart

```python
from laya_decision import BackendConfig, BackendKind, LayaBackend, LayaClient

backend = LayaBackend(BackendConfig(
    kind=BackendKind.MLX,
    dtype="float16",
    device="gpu",
    batch_size=32,
    compile=True,
    pad_to_multiple=16,
    cache_prompts=True,
))
client = LayaClient(backend)
```

MLX is not a Windows or NVIDIA runtime. Use the Transformers backend there.

## Calibration

Laya's published model card warns that shipped probabilities can be
overconfident. Fit temperatures only on held-out, representative labels:

```python
from laya_decision import CalibrationKey, CalibrationRegistry, fit_temperature

temperature = fit_temperature([
    ({"billing": 0.90, "technical": 0.10}, "billing"),
    ({"billing": 0.80, "technical": 0.20}, "technical"),
])
registry = CalibrationRegistry({
    CalibrationKey("english", "choice", 2, "support-v1"): temperature,
})
```

Do not reuse one calibration profile across checkpoints, domains, option
counts, languages, or materially changed question definitions.

## Production cautions

- Preload the checkpoints used by latency-sensitive services; cold loads take
  seconds and consume substantial memory.
- Prefer multilingual routing for non-English input. Confidence cannot repair
  a checkpoint that does not understand the script.
- Treat more than roughly 20 choices as a measured-risk configuration. Laya's
  own benchmarks report degradation for high-cardinality option spaces.
- Keep deterministic authorization, irreversible actions, money calculations,
  and safety vetoes outside the model.
- Run evaluation, calibration, option-order, language-segment, and drift tests
  against every pinned checkpoint before promotion.
- `laya-mlx` is an independent port, not an official ConvAI release. Validate
  numerical fidelity for the exact checkpoint, precision, and runtime version.
