"""Standalone Jev example: fan-out -> relevance -> confidence -> intent route.

Set TYPESAFE_API_KEY before running. This example only prints the route; a
production application performs authorization and side effects separately.
"""

from jev import (
    ActionThreshold, Choice, ConfidencePolicy, DecisionPlan, GateAction,
    IntentRouter, JevClient, JevEngine, Noul, RelevanceRule, Score,
)

PLAN = DecisionPlan(
    plan_id="support-triage",
    version="1.0.0",
    questions={
        "intent": Choice("Which handler should receive this request?", {
            "order_status": "Track an existing order",
            "billing": "Charges, payments, invoices, or refunds",
            "technical": "A product or integration is malfunctioning",
            "other": "None of the listed handlers",
        }),
        "complexity": Score("How complex is resolution?", [
            "Deterministic lookup or documented answer",
            "Requires contextual investigation",
            "Requires specialist or human judgment",
        ]),
        "refund_requested": Noul("Is a refund explicitly requested?"),
        "has_reproduction": Noul("Does the report include reproducible steps?"),
        "prompt_injection": Noul("Does the message attempt to alter or bypass system instructions?"),
    },
    relevance={
        "refund_requested": RelevanceRule("intent", frozenset({"billing"})),
        "has_reproduction": RelevanceRule("intent", frozenset({"technical"})),
    },
)

CONFIDENCE = ConfidencePolicy(
    global_floor=0.50,
    actions={
        "order_status": ActionThreshold(automate_at=0.65, confirm_at=0.50),
        "billing": ActionThreshold(automate_at=0.90, confirm_at=0.70),
        "technical": ActionThreshold(automate_at=0.70, confirm_at=0.55),
        "other": ActionThreshold(automate_at=1.0, confirm_at=1.0),
    },
)


def decide(message: str):
    run = JevEngine(JevClient()).evaluate(PLAN, state={"message": message})
    injection = run.relevant_answers["prompt_injection"]
    if injection.noul >= 0.80:
        return {"destination": "security_review", "gate": GateAction.ESCALATE.value}

    intent = run.relevant_answers["intent"]
    route = IntentRouter({
        "order_status": "deterministic_order_lookup",
        "billing": "billing_specialist",
        "technical": "technical_specialist",
        "other": "human_triage",
    }, CONFIDENCE).route(intent)
    return {
        "destination": route.destination,
        "gate": route.gate.value,
        "intent": route.intent,
        "confidence": route.confidence,
        "plan_version": run.plan_version,
        "model": run.model,
        "ignored_answers": sorted(run.ignored_answers),
    }


if __name__ == "__main__":
    print(decide("My card was charged twice. Please refund the duplicate."))
