"""Lazy adapters for the official Laya and independent Laya-MLX runtimes."""

from __future__ import annotations

import importlib
import inspect
import platform
import threading
from dataclasses import dataclass, field
from enum import Enum
from importlib import metadata
from typing import Any, Mapping, Protocol, runtime_checkable

from laya_decision.errors import LayaConfigurationError, LayaDependencyError, LayaInferenceError
from laya_decision.models import JsonValue


class BackendKind(str, Enum):
    TRANSFORMERS = "transformers"
    MLX = "mlx"


DEFAULT_TRANSFORMERS_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

DEFAULT_MLX_MODELS = {
    "english": "aac6fef/laya-mlx",
    "multilingual": "aac6fef/laya-multilingual-mlx",
    "typed-decisions": "aac6fef/laya-typed-decisions-mlx",
}


@dataclass(frozen=True)
class BackendConfig:
    kind: BackendKind = BackendKind.TRANSFORMERS
    device: str | None = None
    dtype: str = "float16"
    default: str = "english"
    max_loaded: int = 1
    preload: bool | tuple[str, ...] = False
    auto_task_detection: bool = False
    token: str | None = None
    revision: str | None = None
    batch_size: int = 16
    compile: bool = False
    pad_to_multiple: int | None = None
    cache_prompts: bool = False
    models: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.default not in {"english", "multilingual", "typed-decisions"}:
            raise LayaConfigurationError("unknown default Laya checkpoint")
        if self.max_loaded < 1 or self.batch_size < 1:
            raise LayaConfigurationError("max_loaded and batch_size must be positive")
        if self.dtype not in {"float16", "float32", "bfloat16"}:
            raise LayaConfigurationError("dtype must be float16, float32, or bfloat16")


@dataclass(frozen=True)
class BackendHealth:
    backend: str
    available: bool
    loaded: tuple[str, ...]
    package_version: str | None
    detail: str = "ok"


@runtime_checkable
class DecisionBackend(Protocol):
    name: str

    def predict(self, state: JsonValue, questions: Mapping[str, dict], *, model: str | None = None,
                task: str | None = None, lang: str | None = None) -> Mapping[str, Any]: ...

    def route(self, state: JsonValue, questions: Mapping[str, dict], *, model: str | None = None,
              task: str | None = None, lang: str | None = None) -> Mapping[str, Any]: ...

    def preload(self, names: list[str] | None = None) -> None: ...
    def unload(self, name: str | None = None) -> None: ...
    def health(self) -> BackendHealth: ...


class LayaBackend:
    """Upgrade-safe wrapper around either runtime's public ``Router`` API."""

    def __init__(self, config: BackendConfig | None = None, *, runtime: Any | None = None):
        self.config = config or BackendConfig()
        self.name = self.config.kind.value
        self._runtime = runtime
        self._router = None
        self._lock = threading.RLock()

    def _module(self):
        if self._runtime is not None:
            return self._runtime
        module_name = "laya_mlx" if self.config.kind == BackendKind.MLX else "laya"
        if self.config.kind == BackendKind.MLX and (
            platform.system() != "Darwin" or platform.machine().lower() not in {"arm64", "aarch64"}
        ):
            raise LayaDependencyError("Laya-MLX requires Apple Silicon and macOS")
        try:
            self._runtime = importlib.import_module(module_name)
        except ImportError as exc:
            extra = "laya-mlx" if self.config.kind == BackendKind.MLX else "laya"
            raise LayaDependencyError(f"install the optional '{extra}' runtime") from exc
        return self._runtime

    def _build_router(self):
        module = self._module()
        defaults = DEFAULT_MLX_MODELS if self.config.kind == BackendKind.MLX else DEFAULT_TRANSFORMERS_MODELS
        models = dict(defaults)
        models.update(self.config.models)
        kwargs = {
            "models": models, "device": self.config.device, "token": self.config.token,
            "max_loaded": self.config.max_loaded, "default": self.config.default,
            "auto_task_detection": self.config.auto_task_detection,
            "preload": False, "dtype": self.config.dtype, "batch_size": self.config.batch_size,
            "revision": self.config.revision, "compile": self.config.compile,
            "pad_to_multiple": self.config.pad_to_multiple, "cache_prompts": self.config.cache_prompts,
            "standalone_repos": True,
        }
        constructor = module.Router
        supported = _supported_kwargs(constructor, kwargs)
        router = constructor(**supported)
        if self.config.preload:
            names = None if self.config.preload is True else list(self.config.preload)
            router.preload(names)
        return router

    @property
    def router(self):
        with self._lock:
            if self._router is None:
                self._router = self._build_router()
            return self._router

    def predict(self, state: JsonValue, questions: Mapping[str, dict], *, model: str | None = None,
                task: str | None = None, lang: str | None = None) -> Mapping[str, Any]:
        try:
            result = self.router.predict(state, dict(questions), model=model, task=task, lang=lang)
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise LayaInferenceError(f"{self.name} inference failed: {type(exc).__name__}") from exc
        if not isinstance(result, Mapping):
            raise LayaInferenceError("Laya runtime returned a non-object result")
        return result

    def route(self, state: JsonValue, questions: Mapping[str, dict], *, model: str | None = None,
              task: str | None = None, lang: str | None = None) -> Mapping[str, Any]:
        try:
            return dict(self.router.route(state, dict(questions), model=model, task=task, lang=lang))
        except Exception as exc:
            raise LayaInferenceError(f"{self.name} routing failed: {type(exc).__name__}") from exc

    def attach(self, name: str, agent: Any) -> None:
        self.router.attach(name, agent)

    def preload(self, names: list[str] | None = None) -> None:
        self.router.preload(names)

    def unload(self, name: str | None = None) -> None:
        with self._lock:
            if self._router is not None:
                self._router.unload(name)

    def health(self) -> BackendHealth:
        try:
            module = self._module()
            loaded = tuple(getattr(self._router, "loaded", ())) if self._router is not None else ()
            version = getattr(module, "__version__", None)
            if version is None and self._runtime is None:
                package = "laya-mlx" if self.config.kind == BackendKind.MLX else "laya"
                version = metadata.version(package)
            return BackendHealth(self.name, True, loaded, version)
        except Exception as exc:
            return BackendHealth(self.name, False, (), None, type(exc).__name__)


def _supported_kwargs(callable_: Any, values: Mapping[str, Any]) -> dict[str, Any]:
    """Pass only arguments supported by the installed runtime version."""
    try:
        signature = inspect.signature(callable_)
    except (TypeError, ValueError):
        return {key: value for key, value in values.items() if value is not None}
    accepts_all = any(parameter.kind == inspect.Parameter.VAR_KEYWORD
                      for parameter in signature.parameters.values())
    return {
        key: value for key, value in values.items()
        if value is not None and (accepts_all or key in signature.parameters)
    }
