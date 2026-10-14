"""Groq provider: conversational chat."""
from __future__ import annotations

# pyrefly: ignore [missing-import]
from config import RuntimeContext
# pyrefly: ignore [missing-import]
from services.errors import ProviderError, call_with_retries


def _to_provider_error(groq, exc: Exception, model: str) -> ProviderError:
    if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
        return ProviderError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.", "auth")
    if isinstance(exc, groq.NotFoundError):
        return ProviderError(
            f"Groq model '{model}' was not found for your account. Set GROQ_MODEL in .env to a valid model ID.", "model")
    if isinstance(exc, groq.RateLimitError):
        return ProviderError("Groq rate limit reached. Wait a moment and try again.", "rate_limit", retryable=True)
    if isinstance(exc, groq.APITimeoutError):
        return ProviderError("Groq timed out.", "timeout", retryable=True)
    if isinstance(exc, (groq.APIConnectionError, groq.InternalServerError)):
        return ProviderError("Groq is unreachable or temporarily unavailable.", "unavailable", retryable=True)
    if isinstance(exc, groq.BadRequestError):
        return ProviderError(
            "Groq rejected the request (the model ID may be retired or the conversation too long).", "invalid_request")

    # Fallback checking based on status code or message strings
    code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    low = str(exc).lower()
    if code in (401, 403) or "api key" in low:
        return ProviderError("Groq rejected the API key. Check GROQ_API_KEY in your .env file.", "auth")
    if code == 429 or "rate limit" in low:
        return ProviderError("Groq rate limit reached. Wait a moment and try again.", "rate_limit", retryable=True)
    if (isinstance(code, int) and code >= 500) or any(w in low for w in ("timeout", "connect", "unavailable")):
        return ProviderError("Groq is unreachable or temporarily unavailable.", "unavailable", retryable=True)

    return ProviderError(f"Groq request failed ({type(exc).__name__}).", "unknown")


def chat(ctx: RuntimeContext, messages: list[dict], temperature: float = 0.3) -> str:
    """Send chat messages ([{'role','content'}, ...]) to Groq and return the reply text."""
    s = ctx.settings
    if not s.groq_api_key:
        raise ProviderError("GROQ_API_KEY is not set. Add it to your .env file.", "auth")
    if not s.groq_model:
        raise ProviderError("GROQ_MODEL is not set. Add a model ID that your account supports to .env.", "model")
    try:
        import groq
    except ImportError as exc:
        raise ProviderError("The 'groq' package is missing. Run: pip install groq", "setup") from exc

    def _call() -> str:
        ctx.usage["groq"] += 1
        try:
            client = groq.Groq(api_key=s.groq_api_key, timeout=s.request_timeout_s, max_retries=0)
            response = client.chat.completions.create(
                model=s.groq_model, messages=messages, temperature=temperature)
            if not response.choices:
                raise ProviderError("Groq returned no choices in response.", "empty")
            text = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise _to_provider_error(groq, exc, s.groq_model) from exc
        if not text:
            raise ProviderError("Groq returned an empty response.", "empty")
        return text

    return call_with_retries(_call, s.max_retries)