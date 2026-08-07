"""cli.py end-to-end — every subcommand run through main() with the fake
backend (zero network, zero real keys needed), stdout/stderr captured via
pytest's capsys. Argparse/config-error paths are also verified for the
correct exit code, since those are the two other supported --backend
failure/misconfiguration cases besides "the provider itself failed"."""

import os

import pytest

from modelrouter.cli import main


def test_chat_succeeds_with_fake_backend(capsys):
    exit_code = main(["chat", "hello", "--backend", "fake", "--models", "fake:model-a"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fake response" in captured.out


def test_chat_prints_metadata_when_requested(capsys):
    exit_code = main(["chat", "hello", "--backend", "fake", "--models", "fake:model-a", "--metadata"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "served_by" in captured.err
    assert "fake:model-a" in captured.err


# ── --stream (Phase 3) — the one streaming surface runnable without fastapi ──

def test_chat_stream_prints_content_incrementally(capsys):
    exit_code = main(["chat", "hello", "--backend", "fake", "--models", "fake:model-a", "--stream"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "fake response"


def test_chat_stream_prints_metadata_when_requested(capsys):
    exit_code = main(["chat", "hello", "--backend", "fake", "--models", "fake:model-a",
                       "--stream", "--metadata"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "served_by" in captured.err
    assert "fake:model-a" in captured.err


def test_chat_stream_reports_failure_when_every_candidate_is_unreachable(capsys):
    # An unknown provider prefix -> stream_chat() skips it, attempt:0,
    # served_by=None -- same failure-reporting path as the non-streaming case.
    exit_code = main(["chat", "hi", "--backend", "fake", "--models", "ghostprovider:x", "--stream"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "skipped" in captured.err.lower()
    assert "ghostprovider" in captured.err


def test_image_generation_succeeds_with_fake_backend(capsys):
    exit_code = main(["image", "a red bicycle", "--backend", "fake", "--models", "fake:img-model", "--n", "2"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fake.local" in captured.out


def test_speech_writes_audio_file(capsys, tmp_path):
    out_path = str(tmp_path / "out.mp3")
    exit_code = main(["speech", "hello world", "--backend", "fake", "--models", "fake:tts", "--out", out_path])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert os.path.exists(out_path)
    assert os.path.getsize(out_path) > 0
    assert "Wrote" in captured.out


def test_transcribe_reads_audio_file(capsys, tmp_path):
    audio_path = tmp_path / "in.mp3"
    audio_path.write_bytes(b"fake audio content")

    exit_code = main(["transcribe", str(audio_path), "--backend", "fake", "--models", "fake:whisper"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "fake response" in captured.out


def test_transcribe_missing_file_returns_exit_code_2(capsys):
    exit_code = main(["transcribe", "/definitely/does/not/exist.mp3", "--backend", "fake", "--models", "fake:whisper"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "not found" in captured.err.lower()


def test_providers_command_lists_every_known_provider(capsys):
    exit_code = main(["providers"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "OPENAI_API_KEY" in captured.out
    assert "ANTHROPIC_API_KEY" in captured.out
    assert "GROQ_API_KEY" in captured.out
    assert "Ollama needs no API key" in captured.out


def test_invalid_backend_choice_exits_with_argparse_error():
    with pytest.raises(SystemExit) as exc_info:
        main(["chat", "hi", "--backend", "not-a-real-backend", "--models", "x:y"])
    assert exc_info.value.code == 2


def test_no_adapters_for_unconfigured_real_backend_returns_exit_1(capsys, monkeypatch):
    # Force a clean slate so this doesn't depend on the real developer
    # machine's actual environment having (or not having) OPENAI_API_KEY set.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    exit_code = main(["chat", "hi", "--backend", "openai", "--models", "openai:gpt-4o-mini"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "providers" in captured.err   # points the user at the discovery command


def test_chat_with_no_models_argument_is_rejected_by_argparse():
    with pytest.raises(SystemExit) as exc_info:
        main(["chat", "hi", "--backend", "fake"])   # --models is required
    assert exc_info.value.code == 2


def test_pinned_model_failure_reports_skipped_and_attempts(capsys):
    # An unknown provider prefix in --models is reachable purely through CLI
    # args (no need to script a fake adapter failure) and exercises
    # _print_failure's SkippedCandidate reporting path.
    exit_code = main(["chat", "hi", "--backend", "fake", "--models", "ghostprovider:x"])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "skipped" in captured.err.lower()
    assert "ghostprovider" in captured.err


# ── L1/L3 admin commands — the CLI is the admin surface until a real one exists ──

def _sqlite_env(monkeypatch, tmp_path):
    """Persistence across separate main() calls (each a fresh process in
    reality) only happens with the sqlite tier — the memory tier is
    per-invocation-ephemeral by design (Law 1). Tests that chain multiple
    commands need this; single-command tests don't."""
    monkeypatch.setenv("MODELROUTER_STORAGE", "sqlite")
    monkeypatch.setenv("MODELROUTER_SQLITE_PATH", str(tmp_path / "test.db"))


def test_tenants_create_prints_a_new_tenant_id(capsys):
    exit_code = main(["tenants", "create", "acme-corp"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "acme-corp" in captured.out
    assert "tn_" in captured.out


def test_tenants_list_reports_empty_when_ephemeral(capsys):
    exit_code = main(["tenants", "list"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "No tenants" in captured.out


def test_keys_create_for_unknown_tenant_fails_gracefully(capsys):
    exit_code = main(["keys", "create", "tn_does_not_exist"])
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "tn_does_not_exist" in captured.err


def test_memory_tier_warns_about_ephemerality(capsys, monkeypatch):
    monkeypatch.delenv("MODELROUTER_STORAGE", raising=False)
    main(["tenants", "create", "acme-corp"])
    captured = capsys.readouterr()
    assert "NOT persist" in captured.err


def test_full_admin_flow_persists_across_commands_with_sqlite_tier(capsys, tmp_path, monkeypatch):
    """The real end-to-end story: create a tenant, mint a key for it, fund
    it, and read the balance back — as four SEPARATE main() invocations,
    proving the sqlite tier is what makes CLI-based provisioning actually
    useful across process boundaries (unlike the memory tier)."""
    _sqlite_env(monkeypatch, tmp_path)

    main(["tenants", "create", "acme-corp"])
    tenant_id = capsys.readouterr().out.split("'")[1]
    assert tenant_id.startswith("tn_")

    exit_code = main(["keys", "create", tenant_id, "--name", "prod key"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "shown once" in captured.out
    plaintext_key = captured.out.strip().splitlines()[-1].rsplit(": ", 1)[-1]
    assert plaintext_key.startswith("mr_")

    exit_code = main(["credits", "add", tenant_id, "10.50"])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "10.50" in captured.out

    exit_code = main(["usage", tenant_id])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "10.500000" in captured.out

    exit_code = main(["keys", "list", tenant_id])
    captured = capsys.readouterr()
    assert exit_code == 0
    assert "prod key" in captured.out
    assert plaintext_key not in captured.out   # never shown again
