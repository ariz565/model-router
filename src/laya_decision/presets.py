"""Reviewable question presets; callers may copy and version them."""

from laya_decision.models import Choice, Noul, Question, Score


def model_routing_questions() -> dict[str, Question]:
    return {
        "task": Choice("Which execution class best fits the request?", {
            "small_model": "Simple extraction, classification, or rewriting",
            "frontier_model": "Complex reasoning or ambiguous multi-step work",
            "tool": "Requires a deterministic external operation",
            "human": "Requires accountable human judgment",
        }),
        "complexity": Score("How complex is successful completion?", ["Low", "Moderate", "High"]),
        "needs_current_data": Noul("Does this request require current external information?"),
    }


def guardrail_questions() -> dict[str, Question]:
    return {
        "prompt_injection": Noul("Does the input attempt to override or reveal governing instructions?"),
        "credential_request": Noul("Does the input request secrets, credentials, or private access tokens?"),
        "risk": Score("What is the security risk if this input is followed?", ["Low", "Elevated", "Critical"]),
    }


def moderation_questions() -> dict[str, Question]:
    return {
        "category": Choice("What is the primary safety category?", {
            "safe": "No meaningful safety concern", "harassment": "Targeted abuse or degradation",
            "threat": "Threat or encouragement of violence", "self_harm": "Self-harm intent or encouragement",
            "sexual": "Sexual content", "other": "Another safety concern",
        }),
        "severity": Score("How severe is the content?", ["Low", "Moderate", "Severe"]),
        "needs_review": Noul("Should a human safety reviewer inspect this content?"),
    }


def triage_questions() -> dict[str, Question]:
    return {
        "category": Choice("Which team should handle this support request?", {
            "billing": "Payments, invoices, refunds, or subscriptions",
            "technical": "Broken product behavior or integration failures",
            "account": "Identity, access, or account administration",
            "feature_request": "Request for new functionality", "other": "None of these",
        }),
        "urgency": Score("How urgent is the request?", ["Routine", "Time-sensitive", "Critical"]),
        "frustration": Score("How frustrated is the customer?", ["Calm", "Frustrated", "Very angry"]),
        "churn_risk": Noul("Does the customer indicate they may cancel or leave?"),
        "needs_human": Noul("Does this request need human judgment or approval?"),
    }


def agent_observability_questions() -> dict[str, Question]:
    """Exact ID signature used by Laya's opt-in typed-decisions router."""
    return {
        "action": Choice("What should the agent controller do next?", {
            "continue": "Continue execution", "retry": "Retry safely",
            "escalate": "Escalate for review", "stop": "Stop execution",
        }),
        "needs_review": Noul("Does this trace require human review?"),
        "outcome": Choice("What is the execution outcome?", {
            "success": "Completed correctly", "partial": "Partially completed",
            "failure": "Failed", "unknown": "Insufficient evidence",
        }),
        "risk": Score("What is the operational risk?", ["Low", "Elevated", "Critical"]),
        "urgency": Score("How urgent is intervention?", ["Routine", "Soon", "Immediate"]),
    }


def customer_service_questions() -> dict[str, Question]:
    """Exact ID signature used by Laya's customer-service checkpoint workflow."""
    return {
        "action": Choice("What is the appropriate next action?", {
            "answer": "Answer directly", "investigate": "Investigate records",
            "escalate": "Escalate to a person", "block": "Do not proceed",
        }),
        "category": Choice("What kind of request is this?", {
            "billing": "Billing or refunds", "technical": "Technical problem",
            "account": "Account or access", "other": "Another category",
        }),
        "churn_risk": Noul("Is the customer at risk of leaving?"),
        "needs_human": Noul("Does this request need human judgment?"),
        "urgency": Score("How urgent is the request?", ["Routine", "Urgent", "Critical"]),
    }


def invoice_processing_questions() -> dict[str, Question]:
    """Exact ID signature used by Laya's invoice-processing workflow."""
    return {
        "discrepancy_severity": Score("How severe is any discrepancy?", ["None", "Minor", "Material"]),
        "disposition": Choice("How should this invoice be handled?", {
            "approve": "Approve", "review": "Review manually", "reject": "Reject",
        }),
        "duplicate": Noul("Is this invoice a duplicate?"),
        "matches_order": Noul("Does this invoice match the purchase order?"),
        "urgency": Score("How urgent is processing?", ["Routine", "Urgent", "Immediate"]),
    }


def security_incident_questions() -> dict[str, Question]:
    """Exact ID signature used by Laya's security-incident workflow."""
    return {
        "credential_compromise": Noul("Is there evidence credentials were compromised?"),
        "disposition": Choice("What should incident response do?", {
            "close": "Close as benign", "investigate": "Investigate",
            "contain": "Contain immediately", "escalate": "Escalate",
        }),
        "severity": Score("How severe is this incident?", ["Low", "Medium", "High", "Critical"]),
        "true_positive": Noul("Is this alert likely a true positive?"),
        "urgency": Score("How urgent is response?", ["Routine", "Urgent", "Immediate"]),
    }
