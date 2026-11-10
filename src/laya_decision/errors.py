"""Errors exposed by the vendor-neutral Laya integration."""


class LayaDecisionError(Exception):
    pass


class LayaConfigurationError(LayaDecisionError):
    pass


class LayaDependencyError(LayaConfigurationError):
    pass


class LayaValidationError(LayaDecisionError, ValueError):
    pass


class LayaInferenceError(LayaDecisionError):
    pass


class LayaUnavailableError(LayaInferenceError):
    pass


class LayaAbstainedError(LayaInferenceError):
    """Every backend answered, but none met the configured acceptance policy."""

    def __init__(self, message: str, *, attempts=()):
        self.attempts = tuple(attempts)
        super().__init__(message)
