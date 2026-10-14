"""Provider error type and a bounded retry helper (exponential backoff, never infinite)."""
from __future__ import annotations

import time
from typing import Callable, TypeVar

T = TypeVar("T")


class ProviderError(Exception):
    """A readable AI-provider failure. `kind` is one of: setup, auth, model, rate_limit,
    invalid_request, timeout, unavailable, empty, unknown."""

    def __init__(self, message: str, kind: str = "unknown", retryable: bool = False):
        super().__init__(message)
        self.kind = kind
        self.retryable = retryable


def call_with_retries(fn: Callable[[], T], max_attempts: int = 3, base_delay: float = 1.5,
                      sleep: Callable[[float], None] = time.sleep) -> T:
    max_attempts = max(1, max_attempts)
    attempt = 0
    while True:
        attempt += 1
        try:
            return fn()
        except ProviderError as exc:
            if not exc.retryable or attempt >= max_attempts:
                raise
            sleep(base_delay * (2 ** (attempt - 1)))