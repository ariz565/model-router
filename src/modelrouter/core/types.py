"""Canonical types — every provider's request/response normalizes to these.

Deliberately OpenAI-`choices[]`-shaped internally, regardless of which provider
actually serves the request (OpenRouter's own documented decision, adopted here
because it's a proven pattern, not cargo-culted: Anthropic's native response has
no choices[], so normalizing to one shape is a real translation, not a freebie).

ChatRequest.messages stays a bare `list[dict]` rather than a typed content-block
class — this is deliberate, not an oversight. Both OpenAI's and Anthropic's real
wire formats already accept `content` as EITHER a plain string OR a list of
typed blocks (`{"type": "text", "text": ...}`, `{"type": "image_url", ...}` /
`{"type": "image", "source": {...}}`). A caller builds whichever shape their
target model expects; adapters.py's job is translating between the two
providers' block shapes, not this module's. See MULTIMODAL_CONTENT_EXAMPLES in
this file for the two real shapes side by side.

tools/streaming/response_format on ChatRequest are added only when a real
pipeline stage consumes them (agents.md #2 — no speculative config):
`tools` feeds extensions.ToolRegistry.openai_tools_schema(); response_format
gates pipeline.healing's JSON repair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class ChatRequest:
    messages: list[dict]              # OpenAI-style [{"role": ..., "content": ...}] — content is str OR content-block list
    model: str                        # bare model name, e.g. "gpt-4o-mini" (adapter already knows its own provider)
    temperature: float = 0.7
    max_tokens: int | None = None
    response_format: str | None = None   # "json_object" | "json_schema" | None — gates response healing
    tools: list[dict] | None = None      # OpenAI tools[] shape — see extensions.ToolRegistry.openai_tools_schema()
    # Part 6.4's "cost attribution below the tenant" — caller-supplied,
    # opaque pass-through, never inspected or validated by this pipeline
    # (server.py populates this from `x-mr-feature`/`x-mr-end-user`/
    # `x-mr-session` request headers; router.py just carries it through to
    # `AccountingService.settle(tags=...)` unchanged). `None`/`{}` means "no
    # tags supplied" — settle() already defaults to `{}` either way.
    tags: dict[str, str] | None = None
    # L7 Contracts — the upgrade from healing.py's syntax-only repair to real
    # semantic enforcement: a caller-supplied JSON Schema the response must
    # actually satisfy, not just parse. `None` means "no contract" — the
    # existing response_format-gated JSON repair is unaffected either way.
    # Only consulted when response_format is "json_object"/"json_schema"
    # (see pipeline/contracts.py + router.py's own contract-enforcement step).
    json_schema: dict | None = None
    # "fail" (default): a violation is reported in the trace and never
    # silently hidden, but chat() still returns the (unrepaired) response —
    # consistent with this codebase's "never disrupt the return contract of
    # an otherwise-successful call" rule (see errors.raise_if_contract_
    # violated() for the opt-in-raise path). "retry": ONE additional direct
    # call to the same served adapter with a corrective message describing
    # the violations, before falling back to "fail"'s behavior.
    contract_policy: str = "fail"
    # Part 6.8 — prompt & policy versioning as first-class objects. Same
    # "opaque, caller-supplied, never inspected" convention as `tags` above
    # (server.py populates these from `x-mr-prompt-version`/`x-mr-policy-
    # version` headers) — the whole point is answering "did the output
    # change because we changed the prompt/policy, or because the model
    # drifted," which is impossible if a version isn't recorded alongside
    # the outcome it produced. Threaded through to BOTH `SpendSettled`
    # (accounting/) and `TraceRecorded` (observability/) — the two places
    # anything L9/6.1 would ever read a historical outcome back from.
    prompt_version: str | None = None
    policy_version: str | None = None
    # Opt-in payload capture for replay (`observability/replay.py`). Default
    # False, and that default is load-bearing rather than conservative
    # boilerplate: `Trace` records metadata only and `EvidenceBundle` stores a
    # prompt HASH, both deliberately, so nothing in this system retains prompt
    # text unless a caller explicitly asks for it here. Setting it stores the
    # messages encrypted, under a short TTL, readable only by the owning tenant —
    # which is what makes "replay this exact prompt against 4 models" possible
    # at all. Never inspected by the routing pipeline; `server.py` acts on it
    # after a call completes.
    capture_for_replay: bool = False


# The two real multimodal message shapes a caller builds `messages` content
# blocks in, kept here as documentation/reference — NOT enforced by a type,
# since enforcing one shape would make the OTHER provider's native callers
# translate for no reason. adapters.py's OpenAIAdapter/AnthropicAdapter pass
# OpenAI-shaped blocks through as-is (native) and translate them to Anthropic
# blocks only when routing to Anthropic (see adapters.translate_content_blocks).
MULTIMODAL_CONTENT_EXAMPLES = {
    "openai_image_url": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/cat.png"}},
    ],
    "openai_image_base64": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,<...>"}},
    ],
    "anthropic_image_base64": [
        {"type": "text", "text": "What's in this image?"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "<...>"}},
    ],
}


@dataclass(frozen=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


@dataclass(frozen=True)
class Choice:
    index: int
    # {"role": "assistant", "content": "..."}, plus an OPTIONAL "tool_calls"
    # key — OpenAI's own shape (`[{"id", "type": "function", "function":
    # {"name", "arguments"}}]`) — present only when the model actually
    # requested a (caller-executed) tool call; absent, not `[]`, otherwise.
    # AnthropicAdapter.chat() translates `tool_use` content blocks into this
    # same shape, same "canonical OpenAI shape internally" convention this
    # module's docstring already established.
    message: dict
    finish_reason: str                # normalized: "stop" | "length" | "error" | "tool_calls"


@dataclass(frozen=True)
class ChatResponse:
    id: str
    model: str                        # the model that actually served it
    provider: str                     # the provider that actually served it
    choices: list[Choice]
    usage: Usage


@dataclass(frozen=True)
class ChatStreamDelta:
    """One incremental piece of a streaming chat response
    (`StreamingProviderPort.stream_chat()` / `ModelRouter.stream_chat()`,
    ARCHITECTURE-PLAN.md's Phase 3). Deliberately NOT `ChatResponse`-shaped
    with a `choices[]` list — a delta is a single text fragment, not a full
    turn; forcing it into the completion shape would just mean every
    consumer unwraps `choices[0]` for no reason.

    `usage`/`finish_reason` are `None` on every delta except the LAST one —
    mirrors OpenAI's own `stream_options={"include_usage": True}` convention
    (a final, content-less chunk carrying usage) rather than inventing a
    different one. `ModelRouter.stream_chat()`'s caller can tell "this is the
    final delta" by `finish_reason is not None`.

    `tool_calls`, when set, is a list of OpenAI's own raw per-chunk tool-call
    delta shape — confirmed against openai-python's real streaming chunks:
    `[{"index": int, "id"?: str, "type"?: "function", "function": {"name"?:
    str, "arguments"?: str}}]`, where `id`/`type`/`name` appear only on the
    FIRST delta for a given `index` and every subsequent delta for that same
    index carries just an `arguments` fragment to append. This is forwarded
    as-is by the OpenAI-compat streaming surface (it already IS that wire
    shape) and translated by the native/Anthropic-compat surfaces — same
    "canonical OpenAI shape internally" convention this module's docstring
    already established for `Choice.message`. `AnthropicAdapter.stream_chat()`
    synthesizes this same shape from Anthropic's own `content_block_start`/
    `input_json` streaming events (see that adapter for the translation)."""

    content: str = ""
    finish_reason: str | None = None
    usage: Usage | None = None
    tool_calls: list[dict] | None = None


# ── Generation request/response types (image / audio / video) ──────────────
#
# Per the architecture doc's API Surface Map (§7), these are NOT chat
# completions — they're their own endpoints (/audio/speech,
# /audio/transcriptions, /videos, image generation via server tool OR
# dedicated image models) with their own request/response shape. Kept as
# separate dataclasses rather than overloading ChatRequest/ChatResponse: a
# TTS request has no `messages`, a video job is async (submit -> poll), and
# an image generation response returns image data, not a choices[] of text.
# ModelRouter.generate_image / .speech / .transcribe (router.py) route these
# through the SAME provider-fallback/retry/health/billing machinery as chat()
# — the doc's own framing is that chat-style routing controls are shared
# across every endpoint except the unified multimodal one, which these are
# deliberately NOT (each stays close to its provider's native shape instead).


@dataclass(frozen=True)
class ImageGenerationRequest:
    prompt: str
    model: str                        # bare model name, e.g. "gpt-image-1", "dall-e-3"
    n: int = 1                        # how many images to generate
    size: str = "1024x1024"
    response_format: Literal["url", "b64_json"] = "url"


@dataclass(frozen=True)
class ImageArtifact:
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


@dataclass(frozen=True)
class ImageGenerationResponse:
    id: str
    model: str
    provider: str
    images: list[ImageArtifact]


@dataclass(frozen=True)
class SpeechRequest:
    text: str
    model: str                        # e.g. "tts-1"
    voice: str = "alloy"
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = "mp3"


@dataclass(frozen=True)
class SpeechResponse:
    id: str
    model: str
    provider: str
    audio_bytes: bytes
    content_type: str                 # e.g. "audio/mpeg"


@dataclass(frozen=True)
class TranscriptionRequest:
    audio_bytes: bytes
    model: str                        # e.g. "whisper-1"
    filename: str = "audio.mp3"       # some SDKs need a name to infer content-type
    language: str | None = None
    response_format: Literal["json", "text", "verbose_json"] = "json"


@dataclass(frozen=True)
class TranscriptionResponse:
    id: str
    model: str
    provider: str
    text: str
    language: str | None = None
    duration_s: float | None = None


@dataclass(frozen=True)
class VideoGenerationRequest:
    """Per the doc: video generation is async — submit a job, then poll (or
    receive a webhook). VideoGenerationResponse below is the SUBMIT response
    (a job handle); VideoJobStatus is what polling returns."""
    prompt: str
    model: str
    mode: Literal["text-to-video", "image-to-video", "reference-to-video"] = "text-to-video"
    reference_image_url: str | None = None   # required for image-to-video / reference-to-video
    duration_s: float | None = None


@dataclass(frozen=True)
class VideoGenerationResponse:
    """Returned immediately on submit — job_id is what you poll with."""
    job_id: str
    model: str
    provider: str
    status: Literal["queued", "running"] = "queued"


@dataclass(frozen=True)
class VideoJobStatus:
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    video_url: str | None = None      # set once status == "completed"
    error: str | None = None          # set if status == "failed"


@dataclass(frozen=True)
class AttemptRecord:
    """One row in RouterMetadata.attempts — one call attempt, success or failure."""
    attempt: int                      # 0-indexed within this model's provider-retry loop
    provider: str
    model: str
    outcome: str                      # "success" | "error"
    error_type: str | None = None
    error_message: str | None = None  # human-readable — see core.errors.format_error_message
    retryable: bool | None = None
    delay_before_s: float | None = None


@dataclass(frozen=True)
class SkippedCandidate:
    """One candidate ("provider:model") that was never even attempted — no
    AttemptRecord exists for it because it never reached the adapter's
    chat()/generate_image()/etc. call at all. Distinct from a real failure:
    this answers "why didn't the router even try this one," which an empty
    RouterMetadata.attempts list otherwise can't distinguish from "everything
    was tried and failed." reason is one of:
      "unknown_provider"    — the "provider:" prefix matches no registered adapter
      "unsupported_model"   — the adapter exists but supports_model() said no
      "unsupported_capability" — the adapter exists but doesn't implement the
                                  capability Protocol this call needs (e.g. no
                                  ImageGenerationPort for an image request)
    """
    spec: str
    reason: str


@dataclass(frozen=True)
class RouterMetadata:
    """Assembled unconditionally, not opt-in-flag-gated like OpenRouter's
    X-OpenRouter-Metadata header — v0 has no HTTP layer to gate it behind, and
    withholding it by default would just make the router harder to debug for
    the one audience it currently has: direct callers.

    attempt: 0 means blocked pre-flight (a guardrail fired, no provider ever
    contacted) or the models list was empty. attempt: N > 0 means N attempts
    were recorded across every model tried, whether or not any of them
    ultimately succeeded — the distinction that matters is served_by is None
    (nothing succeeded) vs. a real "provider:model" string.
    """
    requested_model: str
    served_by: str | None = None      # "provider:model" that actually succeeded, None if all failed
    attempt: int = 0
    pipeline: list[dict] = field(default_factory=list)      # stage-by-stage trace
    attempts: list[AttemptRecord] = field(default_factory=list)
    model_fallback_index: int | None = None    # which entry in the requested models[] array succeeded
    skipped: list[SkippedCandidate] = field(default_factory=list)   # candidates never even attempted, and why
    # Part 3.3's model half of ceiling-minimization: the served endpoint's own
    # max_output_tokens when it further clamped request.max_tokens, else None
    # (no known ceiling, or it never tightened anything). The key/tenant
    # halves are already folded into the max_tokens the router was CALLED
    # with (see server.py's _effective_max_tokens) — this field reports only
    # the additional model-level clamp, which can differ per fallback
    # candidate.
    model_max_tokens_applied: int | None = None
    # L8 Observability — the SAME request_id (and its real billed cost, if
    # any) TraceService.record() writes to the durable trace log, when
    # tracing is configured. `None`/`0.0` on every pre-flight-blocked exit
    # (guardrail block, empty routing result, insufficient budget) — those
    # exits never reach the point a request_id or a real cost exists, same
    # v1 scope boundary router.py's own tracing code documents.
    request_id: str | None = None
    billed_usd: float = 0.0
