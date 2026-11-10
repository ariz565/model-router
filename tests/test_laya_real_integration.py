"""Opt-in real-checkpoint smoke tests.

These may download large model artifacts and therefore never run implicitly.
Set LAYA_INTEGRATION=1 after pinning/approving the desired artifacts.
"""

import os

import pytest

from laya_decision import BackendConfig, LayaBackend, LayaClient, Noul


pytestmark = pytest.mark.integration


@pytest.mark.skipif(os.environ.get("LAYA_INTEGRATION") != "1",
                    reason="set LAYA_INTEGRATION=1 to run real Laya inference")
def test_real_laya_checkpoint_returns_a_valid_typed_decision():
    pytest.importorskip("laya")
    client = LayaClient(LayaBackend(BackendConfig(
        device=os.environ.get("LAYA_DEVICE"),
        models={"english": os.environ.get("LAYA_ENGLISH_MODEL", "convaiinnovations/laya")},
    )))
    result = client.predict(
        state="The customer explicitly asks for a refund.",
        questions={"refund": Noul("Does the customer request a refund?")},
        model="english",
    )
    assert 0 <= result.answers["refund"].noul <= 1
    assert result.routing.checkpoint == "english"
