"""Typed errors exposed by the standalone Jev client."""

class JevError(Exception):
    pass


class JevConfigurationError(JevError):
    pass


class JevValidationError(JevError, ValueError):
    pass


class JevConnectionError(JevError):
    pass


class JevAPIError(JevError):
    def __init__(self, status_code: int, message: str, *, request_id: str | None = None):
        self.status_code = status_code
        self.request_id = request_id
        super().__init__(message)


class JevAuthenticationError(JevAPIError):
    pass


class JevRateLimitError(JevAPIError):
    pass
