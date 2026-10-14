"""Google Gemini provider: document understanding, extraction and report drafting."""
from __future__ import annotations

import hashlib

# pyrefly: ignore [missing-import]
from config import RuntimeContext
# pyrefly: ignore [missing-import]
from services.errors import ProviderError, call_with_retries

_CACHE_LIMIT = 50


def _cache_key(*parts) -> str:
    return hashlib.sha256("||".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def _to_provider_error(exc: Exception, model: str) -> ProviderError:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    low = str(exc).lower()
    if "safety" in low or "blocked" in low:
        return ProviderError("Gemini blocked the response due to content safety filters.", "safety")
    if code in (401, 403) or "api key" in low or "permission_denied" in low:
        return ProviderError("Gemini rejected the API key. Check GEMINI_API_KEY in your .env file.", "auth")
    if code == 404 or ("model" in low and "not found" in low):
        return ProviderError(
            f"Gemini model '{model}' was not found or is not supported for your key. "
            "Set GEMINI_MODEL in .env to a model ID your account supports.", "model")
    if code == 429 or "resource_exhausted" in low or "quota" in low:
        return ProviderError(
            "Gemini rate limit or free-tier quota reached. Wait a minute and try again; "
            "if it keeps happening, your daily quota may be used up.", "rate_limit", retryable=True)
    if code == 400:
        return ProviderError(
            "Gemini rejected the request as invalid (the input may be too large, or the model "
            "may not support this feature).", "invalid_request")
    if (isinstance(code, int) and code >= 500) or any(w in low for w in ("timeout", "deadline", "connect")):
        return ProviderError("Gemini is temporarily unavailable or timed out.", "unavailable", retryable=True)
    return ProviderError(f"Gemini request failed ({type(exc).__name__}).", "unknown")


def generate(ctx: RuntimeContext, prompt: str, system: str | None = None,
             json_mode: bool = False, use_cache: bool = True) -> str:
    """Send one prompt to Gemini and return text. Raises ProviderError on any failure."""
    s = ctx.settings
    if not s.gemini_api_key:
        raise ProviderError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set. Add it to your .env file.", "auth")
    if not s.gemini_model:
        raise ProviderError("GEMINI_MODEL is not set. Add a model ID that your account supports to .env.", "model")

    key = _cache_key("gemini", s.gemini_model, system, json_mode, prompt)
    if use_cache and key in ctx.cache:
        return ctx.cache[key]

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProviderError("The 'google-genai' package is missing. Run: pip install google-genai", "setup") from exc

    def _call() -> str:
        ctx.usage["gemini"] += 1
        try:
            client = genai.Client(
                api_key=s.gemini_api_key,
                http_options=types.HttpOptions(timeout=s.request_timeout_s * 1000),
            )
            config = types.GenerateContentConfig(
                system_instruction=system,
                temperature=0.2,
                response_mime_type="application/json" if json_mode else None,
            )
            response = client.models.generate_content(model=s.gemini_model, contents=prompt, config=config)
            text = (response.text or "").strip()
        except Exception as exc:  
            raise _to_provider_error(exc, s.gemini_model) from exc
        if not text:
            raise ProviderError("Gemini returned an empty response (it may have been blocked by safety filters).", "empty")
        return text

    result = call_with_retries(_call, s.max_retries)
    if use_cache:
        if key not in ctx.cache and len(ctx.cache) >= _CACHE_LIMIT:
            ctx.cache.pop(next(iter(ctx.cache)))
        ctx.cache[key] = result
    return result