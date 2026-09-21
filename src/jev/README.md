# Standalone Jev module

`jev` is an independent, zero-required-dependency Python package for TypeSafe
System One evaluation. It does not import ModelRouter.

It provides:

- validated Choice, Score, and Noul requests and responses;
- synchronous and native connection-pooled asynchronous clients with per-attempt
  timeouts, total deadlines, cancellation, bounded payloads, exponential backoff,
  HTTP-date/seconds `Retry-After`, typed errors, model listing, and safe telemetry;
- versioned speculative fan-out plans with explicit relevance rules;
- consequence-aware confidence gating;
- transparent composite scoring; and
- intent-to-destination routing that returns a decision but performs no side
  effects.
- sync and async engines that execute a versioned plan and return an immutable
  run containing relevant and ignored answers plus usage metadata.
- unified versioned policies with contextual role, amount, jurisdiction,
  sensitivity, authentication, and reversibility requirements;
- Noul thresholds and hard veto constraints that override model confidence;
- typed database, LLM, tool, human, block, and clarification destinations;
- controlled async handler execution with timeouts and idempotency-required retries;
- signed, append-only SQLite decision evidence; and
- an end-to-end runtime that writes evidence before executing side effects; and
- offline confusion matrices, precision/recall/F1, Brier score, calibration
  error/curves, threshold sweeps, segment reports, and drift reports.

```python
from jev import Choice, JevClient, Noul, Score

client = JevClient()  # reads TYPESAFE_API_KEY
result = client.system_one(
    state={"message": "My payment was taken twice. Please refund it."},
    questions={
        "intent": Choice("Which handler is appropriate?", {
            "billing": "Charges and refunds", "technical": "Software failures", "other": None,
        }),
        "refund": Noul("Is a refund explicitly requested?"),
        "urgency": Score("How urgent is this?", ["Routine", "Urgent", "Critical"]),
    },
)
```

The package treats model output as evidence, not authority. Authorization,
irreversible actions, exact arithmetic, and policy enforcement remain in
deterministic application code.

## Async lifecycle

Install `modelrouter[jev]` to use the native `httpx` transport. Reuse one client
so its connection pool is reused, and close it explicitly:

```python
from jev import AsyncJevClient, Noul

async with AsyncJevClient(total_timeout_s=20) as client:
    result = await client.system_one(
        state={"message": "Please refund the duplicate charge"},
        questions={"refund_requested": Noul("Is a refund requested?")},
    )
```

Cancellation of the calling task cancels the in-flight native request. Existing
synchronous custom transports remain supported by the async client through a
compatibility worker adapter.

## Offline release gates

Thresholds should come from representative labeled outcomes, not intuition:

```python
from jev import ChoiceEvaluationRecord, evaluate_choices, sweep_thresholds

records = [
    ChoiceEvaluationRecord("billing", {"billing": 0.92, "technical": 0.08}, "en"),
    ChoiceEvaluationRecord("technical", {"billing": 0.61, "technical": 0.39}, "en"),
]

report = evaluate_choices(records, threshold=0.85)
candidates = sweep_thresholds(records, [0.70, 0.80, 0.85, 0.90, 0.95])

# Example release policy; choose targets appropriate to the action's risk.
safe_to_release = (
    report.false_action_rate <= 0.01
    and report.coverage >= 0.50
    and report.expected_calibration_error <= 0.05
)
```

Use `evaluate_segments` to expose tenant/language/use-case regressions,
`evaluate_noul` for binary judgments, and `detect_drift` to compare model or
plan versions. Dataset collection must be consented, access-controlled, and
redacted before these records are persisted.
--
The central idea
A normal LLM workflow often looks like this:
Application state
      ↓
Prompt an LLM
      ↓
Generated JSON-like text
      ↓
Parse, validate, repair
      ↓
Application decision
Jev changes that contract:
State + typed questions
          ↓
         Jev
          ↓
Typed answers + probability distributions
          ↓
Deterministic application logic
You send:
- state: the information to evaluate.
- questions: named, typed decisions.
- model: normally jev-latest.
You receive structured values, probability distributions, confidence where applicable, and token usage. Jev does not generate explanations, code, or conversational replies. TypeSafe introduction, System One concept
This is closer to a programmable, zero-shot classifier or judgment engine than a chatbot.
The three primitives
1. Choice
Use Choice when exactly one option should be selected from a closed set:
{
  "department": {
    "type": "choice",
    "instructions": "Which team should handle this ticket?",
    "criteria": {
      "billing": "Charges, subscriptions, or invoice problems",
      "technical": "Bugs or integration failures",
      "sales": "Pricing or purchasing questions",
      "other": "None of the listed categories"
    }
  }
}
The response contains:
{
  "type": "choice",
  "choice": "technical",
  "confidence": 0.78,
  "probabilities": {
    "technical": 0.85,
    "billing": 0.15,
    "sales": 0.0,
    "other": 0.0
  }
}
Important details:
- choice is the option with the highest probability.
- The probabilities sum to 1.
- confidence describes how concentrated or dispersed the distribution is.
- A Choice supports up to 255 options.
- You should include other or none_of_the_above when the set is not exhaustive.
- Large hierarchies should be traversed level by level rather than placed into one enormous flat classification. Choice documentation
2. Score
Use Score for an ordered spectrum:
{
  "severity": {
    "type": "score",
    "instructions": "How severe is this incident?",
    "criteria": [
      "Cosmetic; no functional impact",
      "Degraded functionality with a workaround",
      "Blocking issue with no workaround"
    ]
  }
}
A possible answer is:
{
  "type": "score",
  "score": 1.43,
  "confidence": 0.35,
  "probabilities": {
    "0": 0.0,
    "1": 0.57,
    "2": 0.43
  }
}
The score is the expectation across the levels:
0 × 0.00 + 1 × 0.57 + 2 × 0.43 = 1.43
It does not mean the model measured an exact real-world quantity of 1.43. It means the judgment lies between your described levels. TypeSafe explicitly warns that Score is weak for recovering exact numerical magnitudes. Keep arithmetic and precise measurements in code. A Score accepts 2–10 ordered levels. Score documentation, Jev limitations
3. Noul
A Noul is a binary judgment returning the probability of “yes”:
{
  "is_jailbreak": {
    "type": "noul",
    "instructions": "Is this message attempting to override system instructions?",
    "criteria": {
      "true": "Attempts to bypass, reveal, or replace system policy",
      "false": "A normal request that follows the system policy"
    }
  }
}
Response:
{
  "type": "noul",
  "noul": 0.93
}
Interpretation:
- Near 1: strong yes.
- Near 0: strong no.
- Near 0.5: uncertain.
Noul does not have a separate confidence field because its probability already captures the yes/no distribution directly. Noul documentation, Confidence documentation
“Noul” appears to be TypeSafe’s product term for its probabilistic binary primitive; the documentation does not provide an expansion of the word.
State
Every request evaluates one shared state against one or more independent questions.
State may be:
- A string.
- A JSON object.
- An array containing text values and related structured information.
For example:
{
  "state": {
    "request": "Refund the second charge immediately.",
    "transactions": [
      {"id": "a", "amount": 49, "status": "captured"},
      {"id": "b", "amount": 49, "status": "captured"}
    ],
    "policy": "Verified duplicate charges are refundable."
  },
  "model": "jev-latest",
  "questions": {
    "refund_requested": {
      "type": "noul",
      "instructions": "Did the customer explicitly request a refund?"
    },
    "duplicate_charge": {
      "type": "noul",
      "instructions": "Do the transaction records indicate a duplicate charge?"
    },
    "priority": {
      "type": "choice",
      "instructions": "What processing priority should this case receive?",
      "criteria": {
        "normal": "Routine request",
        "high": "Time-sensitive financial correction",
        "manual_review": "Evidence is insufficient or contradictory"
      }
    }
  }
}
A JSON object is generally preferable because field names preserve relationships and distinguish customer content, policies, records, and metadata. Jev currently accepts text only; images, audio, and video are unsupported. English is its strongest language. State documentation
Parallel questions
A key Jev feature is that all questions are evaluated independently against the same state.
                         ┌─ refund requested? ─── Noul
State ingested once ─────┼─ duplicate charge? ─── Noul
                         ├─ customer tone? ────── Choice
                         ├─ severity? ─────────── Score
                         └─ escalation needed? ── Noul
The company says adding questions only slightly increases latency because the state is ingested once and questions are evaluated in parallel. However, additional questions still consume tokens.
This enables “speculative fan-out”: ask every potentially useful atomic question in one call, then have ordinary code ignore irrelevant answers.
That is materially different from asking an LLM:
First decide the department, then conditionally generate another answer.

In Jev, the model does not control the workflow. Your code does.
Probability versus confidence
These should not be confused.
For Choice:
{
  "choice": "billing",
  "probabilities": {
    "billing": 0.51,
    "technical": 0.46,
    "sales": 0.03
  },
  "confidence": 0.19
}
- 0.51 is the probability assigned to billing.
- The low confidence says the whole distribution is ambiguous.
- Your software might send this case for manual review.
A more decisive distribution:
{
  "billing": 0.96,
  "technical": 0.03,
  "sales": 0.01
}
would have substantially higher confidence.
Confidence is calculated from the shape of the probability distribution; it is not a separate second model opinion. TypeSafe also provides the full probabilities so applications can use their own uncertainty statistic. Confidence documentation
Also, calibration is a population-level property. If predictions assigned approximately 0.8 are calibrated, roughly 80% should be correct across a representative group. It never means an individual answer is guaranteed to be correct.
The recommended application architecture
TypeSafe’s intended architecture is:
1. Gather relevant application state.
2. Ask several narrow, atomic questions.
3. Read probabilities and confidence.
4. Apply thresholds and deterministic rules in code.
5. Escalate uncertainty to a human or reasoning model.
6. Measure results against real labels.
7. Tune thresholds and question definitions based on observed performance.
For example:
if answers["is_jailbreak"].noul >= 0.90:
    block()
elif answers["is_jailbreak"].noul >= 0.55:
    send_to_review()
else:
    continue_to_llm()
The threshold is your policy, not Jev’s. A medical or financial workflow should use different thresholds from a document-tagging workflow.
A complex judgment should be decomposed:
Bad:
  "Should we approve this refund?"

Better:
  refund_requested       → Noul
  duplicate_charge       → Noul
  policy_support         → Noul
  fraud_risk             → Score
  evidence_completeness  → Score

Then:
  approval = deterministic policy code
This makes the weighting inspectable, testable, versionable, and changeable without rewriting a giant prompt.
What Jev is good at
Based on its interface and published examples, its strongest applications are:
- Intent classification.
- Support-ticket routing.
- Guardrails and moderation signals.
- RAG passage filtering.
- Reranking candidate passages.
- Citation-support checking.
- Skill or tool selection.
- Entity matching.
- Confidence-gated workflow routing.
- Triage and human escalation.
- Turning semantic judgments into numeric features.
- Pre-validating candidates found by deterministic extraction.
The documentation includes cookbooks for reranking, semantic search, function selection, prompt-injection screening, citation checking, hierarchical classification, and RAG passage classification. Documentation index
What Jev is not
It is not designed to:
- Write answers.
- Produce explanations.
- Generate code.
- Perform exact calculations.
- Reliably compare raw dates.
- Reconstruct exact numeric values from Score outputs.
- Process images, audio, or video.
- Perform long, multi-step reasoning.
- Automatically resist adversarial instructions inside state.
TypeSafe openly documents several Jev 1.13 failure modes:
- Literal interpretation.
- Weak numeric precision.
- Weak date/time comparisons.
- Difficulty with several levels of indirection.
- Accuracy degradation from irrelevant context.
- Susceptibility to adversarial content.
- Confusion when instructions and criteria conflict.
- No guarantee that logically equivalent question formulations produce structurally identical probabilities.
- No text generation. Jev 1.13 jaggedness
A particularly important warning: the state is treated as data, but Jev does not inherently consider that data hostile. Prompt injection or persuasive content inside state can influence the result. This means it cannot be treated as a perfect security boundary.
Current operational details
As documented on September 22, 2026:
- Current versioned model: jev-1.13.0.
- Recommended alias: jev-latest.
- Endpoint: POST https://api.typesafe.ai/v1/systemone.
- Context: 64k aggregate request budget, with a separate 32k limit for state plus the longest question.
- Input: text/JSON text structures only.
- Pricing: $0.042 per million input tokens; output tokens are listed as free.
- Published limits: 250,000 tokens/second and 1,200 requests/minute.
- Python package: typesafe-sdk.
- JavaScript package: @typesafe-ai/sdk.
- The company says customer requests and responses are not used for training, and enterprise customers can obtain zero-data-retention terms. Models and pricing
These values are current vendor-published details, not permanent protocol guarantees.
Is this truly a new kind of AI?
Conceptually, probabilistic classification is not new. Classifiers have returned class probabilities for decades, and LLM applications already use constrained decoding and structured outputs.
What is different is the product abstraction:
- Natural-language state and criteria.
- Zero-shot, user-defined classifications.
- Several typed question forms.
- Full probability distributions.
- Parallel independent questions.
- An API designed for application decisions instead of prose generation.
- Explicit confidence-gated orchestration.
So I would call Jev a useful new model/API category, but not an entirely new branch of machine learning. “System One model” is TypeSafe’s product category. The public documentation does not disclose enough of the underlying architecture or training process to independently determine how different the base model technology is from conventional language-model classifiers.
Jev versus ordinary LLM structured output
Concern	LLM with JSON schema	Jev
Primary job	Generate content	Make constrained judgments
Output	Tokens constrained into JSON	Native typed decisions
Options	Often prompt/schema-defined	Primitive-defined
Probabilities	Usually unavailable or unreliable	Central API output
Confidence routing	Must be derived separately	Built into Choice/Score
Explanations	Available	Not generated
Complex reasoning	Often capable	Explicitly not the goal
Exact arithmetic	Still unreliable	Explicitly delegated to code
Multiple questions	May interact within generation	Designed as independent evaluations
Best use	Content or complex reasoning	Classification, scoring, routing


How this fits ModelRouter
Jev should be integrated as a distinct capability, not disguised as chat completions.
Recommended shape:
POST /v1/evaluations
        │
        ├─ state
        ├─ questions
        ├─ model = typesafe/jev-latest
        └─ policy/confidence thresholds
                ↓
        ModelRouter evaluation pipeline
                ↓
        TypeSafe/Jev adapter
                ↓
        typed answers + probabilities
ModelRouter should add:
- An EvaluationPort or DecisionPort.
- Normalized ChoiceQuestion, ScoreQuestion, and NoulQuestion types.
- A TypeSafe/Jev provider adapter.
- A capability such as structured_decision, separate from chat.
- /v1/systemone compatibility and a ModelRouter-native /v1/evaluations.
- BYOK support for TYPESAFE_API_KEY.
- Per-tenant access policy.
- Input-token accounting using versioned Jev pricing.
- RPM/TPM/concurrency enforcement.
- Trace fields for question versions, probability distributions, thresholds, and final application action.
- Evaluation datasets that measure calibration, confusion matrices, abstention rates, and threshold performance.
- Explicit fallback from Jev to a reasoning model or human review when confidence is insufficient.
The highest-value initial ModelRouter use cases would be:
1. Pre-routing intent classification.
2. Prompt-injection and policy-risk signals.
3. RAG passage filtering and reranking.
4. Skill/tool selection.
5. Confidence-gated model routing.
6. Automated evaluation labels.
7. Provider incident and support-ticket triage.
My strongest recommendation: integrate Jev as a probabilistic decision provider, while keeping deterministic policy in ModelRouter. Jev should produce evidence; ModelRouter should remain the authority that decides whether to route, block, escalate, or request human review.

Overall philosophy
TypeSafe is designed to operate inside a larger software system—not to become the system’s autonomous controller.
Application state
       ↓
Atomic TypeSafe questions
       ↓
Typed answers + probabilities + confidence
       ↓
Deterministic policy code
       ↓
Database lookup / specialist LLM / tool / human / block
The key design principle is:
The model evaluates meaning; code controls behavior.

The documented pattern catalog contains four main patterns:
Pattern	Purpose	Main benefit
Speculative fan-out	Evaluate many potentially useful questions together	Latency and cost
Confidence-gated routing	Decide whether an answer is safe enough to use	Reliability and safety
Composite scoring	Combine several atomic dimensions in code	Explainability and tunability
Intent routing	Select the appropriate handler	Cost and speed


TypeSafe patterns overview
1. Speculative fan-out
Core feature
Instead of making sequential model calls:
Classify category
    ↓
If bug → ask severity
    ↓
If billing → ask refund intent
    ↓
Ask frustration
you send all questions in one request:
                         ┌─ category
                         ├─ bug severity
Shared state ────────────┼─ reproducible steps
                         ├─ refund requested
                         └─ frustration
The application examines category and ignores irrelevant answers.
TypeSafe says questions are evaluated in parallel, so adding questions usually has little effect on response time. The full decision tree can therefore be populated with one API round trip. Speculative fan-out
Features
Shared-state ingestion
The state is supplied once. Every question evaluates the same state independently.
This is especially valuable when the state is large:
- A support conversation.
- A retrieved document.
- An account history.
- A transaction record.
- A code review.
- A provider incident report.
You avoid resending the same state over several sequential requests.
Mixed question types
A fan-out request can mix:
- Choice for category or intent.
- Score for severity, complexity, quality, or frustration.
- Noul for binary properties such as refund intent or reproducible steps.
Example:
{
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "What kind of support request is this?",
      "criteria": {
        "bug_report": "Existing behavior is broken",
        "billing": "Payment or account charge problem",
        "feature_request": "Request for new functionality",
        "other": "None of the listed categories"
      }
    },
    "bug_severity": {
      "type": "score",
      "instructions": "How severe is the reported software issue?",
      "criteria": [
        "Minor inconvenience",
        "Important feature is degraded",
        "Critical workflow is blocked"
      ]
    },
    "has_reproducible_steps": {
      "type": "noul",
      "instructions": "Does the report contain enough steps to reproduce the issue?"
    },
    "refund_requested": {
      "type": "noul",
      "instructions": "Is the customer requesting a refund?"
    },
    "frustration": {
      "type": "score",
      "instructions": "How frustrated is the customer?",
      "criteria": ["Calm", "Frustrated", "Very angry"]
    }
  }
}
Speculative questions
A speculative question might be irrelevant to the final route:
- Bug severity is irrelevant for a billing issue.
- Refund intent is irrelevant for a feature request.
- Shipping type is irrelevant for a software failure.
The answer still exists, but code must apply an explicit relevance predicate:
if category.choice == "bug_report":
    use(bug_severity)
else:
    ignore(bug_severity)
Cross-cutting questions
Some answers remain useful for every route:
- User frustration.
- Security risk.
- Prompt injection probability.
- Urgency.
- Language.
- PII detection.
- Human-escalation request.
These can be evaluated in the same fan-out request and processed independently of the primary intent.
Fewer network round trips
Sequential conditional evaluation introduces cumulative latency:
Call 1 latency + Call 2 latency + Call 3 latency
Fan-out changes that to approximately:
One call latency containing all questions
This is one of the strongest practical features of the design.
Constraints and failure modes
Questions do not depend on one another
The model evaluates questions independently. A later question cannot safely assume that an earlier answer was a particular value.
Bad:
Question 1: Which department?
Question 2: Given your answer to question 1, what action should it take?
Better:
Question 1: Which department?
Question 2: If this is a billing problem, is a refund requested?
The application supplies the dependency through code.
Irrelevant answers may still appear plausible
Because a speculative question is always evaluated, it can return a confident-looking result even when its premise is irrelevant. Relevance must therefore be enforced by deterministic policy, not inferred from the existence of an answer.
More questions still consume tokens
TypeSafe says additional questions have little latency impact, but they are not free. Fan-out should include questions that are plausibly useful, not every conceivable judgment.
Question versioning matters
Changing:
- Instructions.
- Criteria.
- Options.
- Level descriptions.
- State construction.
can change the distribution. Production systems should version the complete decision plan.
2. Confidence-gated routing
Core feature
The selected answer and its confidence serve different purposes:
Answer     → What does the model think?
Confidence → Should the system trust that answer enough to act?
The documented example uses voice banking:
- Below 0.6: send every action to a human.
- Balance lookup: 0.6 is sufficient.
- Transfer approval: greater than 0.85 is required.
- Moderate-confidence transfer approval: ask the user to confirm.
The essential feature is that thresholds depend on the consequence of a mistake. Confidence-gated routing
Features
Global uncertainty floor
A minimum threshold catches broadly uncertain results:
if intent.confidence < GLOBAL_FLOOR:
    route_to_human()
This establishes an abstention path. The system is allowed to say, “I am not confident enough to automate this.”
Per-action thresholds
Each action can have a separate threshold:
THRESHOLDS = {
    "read_documentation": 0.55,
    "check_balance": 0.65,
    "send_email": 0.80,
    "change_permissions": 0.90,
    "transfer_money": 0.97,
}
This is much safer than one global threshold because classification mistakes have unequal consequences.
Multiple uncertainty outcomes
Confidence routing need not be binary. A useful production structure is:
Very high confidence → act automatically
Moderate confidence  → ask for confirmation
Low confidence       → use a stronger reasoning model
Very low confidence  → human review
For example:
if confidence >= auto_threshold:
    execute()
elif confidence >= confirmation_threshold:
    request_confirmation()
elif confidence >= reasoning_threshold:
    call_reasoning_model()
else:
    route_to_human()
Confidence as a policy input
Confidence should be combined with:
- Action risk.
- User role.
- Tenant policy.
- Transaction amount.
- Resource sensitivity.
- Regulatory jurisdiction.
- Whether the action is reversible.
- Historical model performance.
A 0.9 confidence result should not automatically authorize a financial transaction. Authentication, authorization, deterministic validation, and confirmation requirements remain separate controls.
Fail-safe routing
Every policy needs explicit behavior for:
- Missing answer.
- Missing confidence.
- Unknown intent.
- Unsupported action.
- Model timeout.
- Invalid probability distribution.
- Threshold exactly on a boundary.
The safest default for consequential operations is abstain, confirm, or escalate.
Critical analytical point
The thresholds in the TypeSafe examples—0.5, 0.6, and 0.85—are demonstrations, not universal safe defaults.
Production thresholds should be selected from evaluation data:
false-positive cost
false-negative cost
coverage
automatic-action rate
human-review rate
calibration
performance by language and customer segment
A threshold should be attached to a versioned policy and periodically re-evaluated.
3. Intent routing
Core feature
Intent routing places a fast classifier in front of multiple execution paths:
Incoming request
       ↓
Intent + complexity evaluation
       ↓
 ┌────────────┬──────────────┬──────────────┬─────────────┐
 │ Code/DB    │ Specialist   │ General LLM  │ Human       │
 │ handler    │ LLM          │ or agent     │ review      │
 └────────────┴──────────────┴──────────────┴─────────────┘
The TypeSafe example routes:
- Order status → deterministic code.
- Product questions → product-specialist LLM.
- Return/exchange → returns-specialist LLM.
- Complex complaint → human.
- Simple complaint → complaint-resolution LLM.
Intent and complexity are evaluated in one request. Intent routing
Features
Deterministic-handler routing
Some requests need no generative model:
- Order lookup.
- Password-reset initiation.
- Fetching an invoice.
- Checking job status.
- Reading an account balance.
- Returning a cached response.
- Running an approved API operation.
Intent routing can eliminate unnecessary LLM calls entirely.
Specialist-model routing
Different intents can route to different models or prompts:
coding problem     → code-specialist model
legal retrieval    → legal/RAG pipeline
simple summary     → inexpensive small model
research request   → reasoning model
image request      → image-generation provider
This naturally fits a model router.
Human routing
Human escalation becomes a first-class destination, not merely an exception:
- Classification uncertainty.
- High complexity.
- Regulated decision.
- Angry or vulnerable customer.
- Large financial impact.
- Suspected fraud.
- Unsupported request.
- Conflicting evidence.
Multiple routing dimensions
The pattern uses both:
- Intent: what kind of request is this?
- Complexity: how difficult or risky is it?
A production router could evaluate additional dimensions:
- Urgency.
- Sensitivity.
- Required modality.
- Tool requirement.
- Reasoning depth.
- Expected context size.
- Cost tolerance.
- Safety risk.
- Need for current web information.
- Need for human approval.
Cost-tier routing
Intent routing can reduce spend:
Deterministic request → no LLM
Simple semantic task  → small model
Specialist task       → domain model
Hard reasoning        → frontier model
Unsafe/uncertain      → human
Context isolation
Specialist models should receive only the context required for their task. That reduces:
- Token cost.
- Context confusion.
- Prompt-injection exposure.
- Accidental disclosure.
- Irrelevant instructions.
Routing is policy, not just classification
A classifier returns:
{
  "intent": "return_exchange",
  "confidence": 0.88
}
The application—not Jev—decides that this maps to:
returns-specialist model
+ returns knowledge base
+ permitted tools
+ maximum budget
+ escalation threshold
That mapping should be versioned and auditable.
4. Composite scoring
Although you did not link its individual page, it is the fourth feature listed on the main Patterns page, so it belongs in the complete analysis.
Core feature
Break a broad judgment into independent dimensions, normalize them, and combine them using code-controlled weights.
TypeSafe’s example separately scores:
- Python depth.
- Team leadership.
- System design.
- Generalist ability.
It then uses different weights for a senior individual-contributor role and an engineering-manager role. Composite scoring
python_depth = answers["python_depth"].score / 4
leadership = answers["leadership"].score / 4
architecture = answers["architecture"].score / 4
generalist = answers["generalist"].score / 4

senior_ic = (
    0.40 * python_depth
    + 0.10 * leadership
    + 0.40 * architecture
    + 0.10 * generalist
)

engineering_manager = (
    0.15 * python_depth
    + 0.40 * leadership
    + 0.20 * architecture
    + 0.25 * generalist
)
Features
- Independent semantic dimensions.
- Normalization to a common scale.
- Explicit weights in code.
- Different profiles using the same underlying answers.
- Inspectable contribution of each dimension.
- Easy weight changes without prompt rewriting.
- Ranking across several candidates.
- Ability to add hard deterministic constraints.
For example:
if security_risk.noul > 0.8:
    reject_regardless_of_composite_score()

if evidence_quality.confidence < 0.5:
    send_to_review()
Composite scores should not override mandatory safety or compliance rules.
Important limitation
Jev Score outputs represent semantic positions between described levels. They are not reliable exact measurements. Composite scoring works best for relative ranking and thresholding—not reconstructing exact quantities.
How the four patterns compose
These are not competing patterns. They form one decision pipeline:
1. SPECULATIVE FAN-OUT
   Evaluate intent, complexity, safety, urgency, modality,
   domain, and specialist dimensions in one call
                     ↓
2. CONFIDENCE GATING
   Decide which judgments are reliable enough to use
                     ↓
3. COMPOSITE SCORING
   Combine relevant dimensions with policy-owned weights
                     ↓
4. INTENT ROUTING
   Select deterministic code, model, provider, tool, or human
An end-to-end example:
User request
   │
   ▼
Jev fan-out
   ├─ intent: Choice
   ├─ task complexity: Score
   ├─ security risk: Score
   ├─ prompt injection: Noul
   ├─ requires tools: Noul
   ├─ needs fresh data: Noul
   ├─ requires vision: Noul
   └─ urgency: Score
   │
   ▼
Confidence policy
   ├─ insufficient intent confidence → human/general model
   ├─ possible prompt injection → guarded path
   └─ uncertain risk → review
   │
   ▼
Composite route score
   ├─ quality weight
   ├─ latency weight
   ├─ price weight
   └─ provider-health weight
   │
   ▼
ModelRouter route
   ├─ deterministic handler
   ├─ cheap model
   ├─ specialist model
   ├─ reasoning model
   ├─ tool/agent workflow
   └─ human review
What ModelRouter should adopt
1. A first-class decision operation
Do not represent Jev as ordinary chat. Add a separate capability:
class DecisionPort(Protocol):
    async def evaluate(
        self,
        state: JsonValue,
        questions: dict[str, DecisionQuestion],
    ) -> DecisionResponse:
        ...
Normalized question types:
ChoiceQuestion
ScoreQuestion
NoulQuestion
Normalized answers:
ChoiceAnswer
ScoreAnswer
NoulAnswer
DecisionUsage
2. Versioned decision plans
A decision plan should contain:
{
  "plan_id": "route-request",
  "version": "2026-09-22.1",
  "questions": {},
  "relevance_rules": {},
  "confidence_policy": {},
  "composite_formulas": {},
  "route_actions": {}
}
This prevents routing behavior from being scattered across prompts and handlers.
3. Relevance rules for speculative answers
ModelRouter should record which answers were:
- Evaluated.
- Relevant.
- Ignored.
- Used in the final decision.
- Rejected because of low confidence.
That distinction is essential for debugging fan-out behavior.
4. Risk-tiered confidence policy
For example:
global_floor: 0.50

actions:
  use_prompt_cache:
    auto: 0.55

  choose_small_model:
    auto: 0.65

  invoke_read_only_tool:
    confirm: 0.65
    auto: 0.85

  invoke_write_tool:
    confirm: 0.85
    auto: 0.97

  change_access_policy:
    automatic: false
    human_required: true
Some operations should remain human-authorized regardless of model confidence.
5. Routing destinations beyond models
ModelRouter’s route result should support:
model
provider
deterministic handler
tool workflow
agent
human queue
reject
request clarification
That makes intent routing a general execution router rather than only a model selector.
6. Decision evidence in traces
Each trace should preserve:
- Decision-plan ID and version.
- Jev model version, not only jev-latest.
- Question definitions or their hashes.
- Full probability distributions.
- Confidence values.
- Thresholds applied.
- Relevant versus ignored answers.
- Composite weights and result.
- Selected route.
- Whether the route was automatic, confirmed, or reviewed.
- Later outcome/feedback where available.
7. Offline threshold evaluation
Before deploying a threshold, ModelRouter should calculate:
- Precision and recall per intent.
- Confusion matrix.
- Automation coverage.
- Human-review rate.
- False-action rate.
- Performance by tenant, language, and use case.
- Calibration curves.
- Cost and latency savings.
- Drift between model versions.
8. Explicit failure behavior
A Jev dependency failure should have a configured policy:
fail closed
fall back to deterministic rules
fall back to a reasoning model
route to human
use last known decision only if request is idempotent and policy permits
It should never silently choose a default high-impact action.
Recommended ModelRouter implementation order
1. Add normalized decision request/response types.
2. Add DecisionPort.
3. Implement the TypeSafe/Jev adapter.
4. Add /v1/systemone compatibility.
5. Add a native /v1/decisions endpoint.
6. Add versioned decision plans.
7. Implement fan-out relevance rules.
8. Implement risk-tiered confidence policies.
9. Connect intent results to the existing routing engine.
10. Add composite route scoring.
11. Persist decision evidence in ModelRouter traces.
12. Add golden datasets and calibration evaluation.
13. Use Jev for guardrails, skill selection, RAG filtering, and provider/model routing.
