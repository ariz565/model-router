"""Local typed decisions with routing, calibration, policy gating, and evidence.

Install ``laya`` and ensure its model artifacts are approved for your
environment before running. The first run may download a checkpoint.
"""

import os

from laya_decision import (
    BackendConfig, CalibrationKey, CalibrationRegistry, EvidenceSigner,
    LayaBackend, LayaClient, build_evidence, customer_service_questions,
)


def main() -> None:
    calibration = CalibrationRegistry({
        # Replace this illustrative value with one fitted on held-out outcomes.
        CalibrationKey("english", "choice", 4, "support-v1"): 1.0,
    })
    client = LayaClient(
        LayaBackend(BackendConfig(device=os.getenv("LAYA_DEVICE"), max_loaded=2)),
        calibration=calibration,
    )
    response = client.predict(
        state={"message": "My card was charged twice. Please refund the duplicate."},
        questions=customer_service_questions(),
        calibration_domain="support-v1",
    )
    category = response.answers["category"]
    if category.confidence < 0.85:
        action = "human_review"
    elif response.answers["needs_human"].noul >= 0.70:
        action = "human_review"
    else:
        action = category.choice

    secret = os.environ.get("LAYA_EVIDENCE_SECRET", "development-only-secret-change-me!").encode()
    evidence = build_evidence("example-request", response, EvidenceSigner(secret))
    print({"action": action, "routing": response.routing, "evidence": evidence})


if __name__ == "__main__":
    main()
