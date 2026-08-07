"""Core domain types and the one provider seam — the vocabulary every other
subpackage speaks in. Nothing here imports from any other modelrouter
subpackage, so `core` is the base of the dependency graph (routing/pipeline/
providers all import core, never the reverse)."""

from modelrouter.core.errors import (
    AllCandidatesExhaustedError,
    ConfigError,
    ErrorInfo,
    ModelRouterError,
    NoAdapterError,
    classify_error,
    format_error_message,
    raise_if_failed,
)
from modelrouter.core.ports import ImageGenerationPort, ProviderPort, SpeechPort, TranscriptionPort
from modelrouter.core.types import (
    MULTIMODAL_CONTENT_EXAMPLES,
    AttemptRecord,
    ChatRequest,
    ChatResponse,
    Choice,
    ImageArtifact,
    ImageGenerationRequest,
    ImageGenerationResponse,
    RouterMetadata,
    SkippedCandidate,
    SpeechRequest,
    SpeechResponse,
    TranscriptionRequest,
    TranscriptionResponse,
    Usage,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoJobStatus,
)

__all__ = [
    "ModelRouterError",
    "ConfigError",
    "NoAdapterError",
    "AllCandidatesExhaustedError",
    "raise_if_failed",
    "classify_error",
    "format_error_message",
    "ErrorInfo",
    "ProviderPort",
    "ImageGenerationPort",
    "SpeechPort",
    "TranscriptionPort",
    "ChatRequest",
    "ChatResponse",
    "Choice",
    "Usage",
    "AttemptRecord",
    "RouterMetadata",
    "SkippedCandidate",
    "MULTIMODAL_CONTENT_EXAMPLES",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageArtifact",
    "SpeechRequest",
    "SpeechResponse",
    "TranscriptionRequest",
    "TranscriptionResponse",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoJobStatus",
]
