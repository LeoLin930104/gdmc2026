from __future__ import annotations
import os
import requests

# --- Backend selection -------------------------------------------------------
# Flip to "api" to route every chat() call through the hosted endpoint instead
# of local LM Studio.
BACKEND = "local"  # "local" | "api"

# Local LM Studio (used when BACKEND == "local")
LOCAL_BASE_URL = "http://localhost:1234/v1"
LOCAL_MODEL = "local-model"

# Hosted OpenAI-compatible endpoint (used when BACKEND == "api").
# The API key comes from the environment so the secret is never hardcoded here.
API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1")
API_MODEL = os.environ.get("LLM_API_MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("LLM_API_KEY", "")  # set in your shell; never commit it

# Backwards-compatible aliases (referenced by docs / older callers).
DEFAULT_BASE_URL = LOCAL_BASE_URL
DEFAULT_TIMEOUT = 30


class LLMUnavailable(RuntimeError):
    """Raised when the LLM backend cannot be reached at all.

    Distinct from a generic transport/HTTP error so callers can tell "the server
    isn't there" (LM Studio not running, port closed, request timed out) from
    "the server answered but something else went wrong". The narrative generators
    catch THIS to switch to offline fallback content (see fallback_content.py),
    while still letting genuine HTTP/parse errors surface as before.
    """


def chat(
    user_message: str,
    system_message: str = "You are a creative writer for a Minecraft settlement.",
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 200,
    timeout: int = DEFAULT_TIMEOUT,
) -> str:
    """Send a chat completion request and return the assistant's text.

    Backend (local LM Studio vs. hosted API) is chosen by the module-level
    BACKEND constant. Explicitly passing base_url/model overrides that choice.

    Raises requests.HTTPError / requests.ConnectionError on transport failure,
    or RuntimeError if BACKEND == "api" but no API key is configured.
    """
    headers: dict[str, str] = {}
    if BACKEND == "api":
        if not API_KEY:
            raise RuntimeError(
                "BACKEND='api' but no API key found. Set the LLM_API_KEY "
                "environment variable (and optionally LLM_API_BASE_URL / "
                "LLM_API_MODEL), or set BACKEND='local' to use LM Studio."
            )
        resolved_base = base_url or API_BASE_URL
        resolved_model = model or API_MODEL
        headers["Authorization"] = f"Bearer {API_KEY}"
    else:
        resolved_base = base_url or LOCAL_BASE_URL
        resolved_model = model or LOCAL_MODEL

    payload = {
        "model": resolved_model,
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    try:
        r = requests.post(
            f"{resolved_base}/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout,
        )
    except (requests.ConnectionError, requests.Timeout) as exc:
        # "Not connected": server unreachable or hung. Re-raise as the typed
        # exception so generators can switch to offline fallback content.
        raise LLMUnavailable(
            f"LLM backend unreachable at {resolved_base}: {exc}"
        ) from exc
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
