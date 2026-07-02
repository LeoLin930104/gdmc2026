from __future__ import annotations
import os
from pathlib import Path

import requests

# --- .env loading ------------------------------------------------------------
# The narrative layer talks to a hosted, OpenAI-compatible LLM endpoint (there is
# no local backend). Judges paste their key into the repo-root .env (tracked, NOT
# gitignored) and it is loaded here, so no shell setup is required. Real
# environment variables always win over the file, so CI or a judge's shell can
# still override the .env values.


def _parse_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from `path` into os.environ (never overwriting).

    Minimal parser (supports `#` comments, blank lines, and optional single/
    double quotes around the value) so we don't need the python-dotenv
    dependency. Existing environment variables take precedence over the file.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_dotenv() -> None:
    """Find and load the first `.env` at or above this file (repo root, etc.)."""
    here = Path(__file__).resolve()
    for base in (here.parent, *here.parents):
        candidate = base / ".env"
        if candidate.is_file():
            _parse_dotenv(candidate)
            return


_load_dotenv()

# --- API configuration -------------------------------------------------------
# Hosted OpenAI-compatible endpoint. The key comes from the environment / .env so
# the secret is never hardcoded here. Any OpenAI-compatible provider works by
# overriding LLM_API_BASE_URL / LLM_API_MODEL.
API_BASE_URL = os.environ.get("LLM_API_BASE_URL", "https://api.openai.com/v1")
API_MODEL = os.environ.get("LLM_API_MODEL", "gpt-4o-mini")
API_KEY = os.environ.get("LLM_API_KEY", "")  # set in .env or your shell

# Backwards-compatible aliases (referenced by docs / older callers).
DEFAULT_BASE_URL = API_BASE_URL
DEFAULT_TIMEOUT = 30


class LLMUnavailable(RuntimeError):
    """Raised when the LLM endpoint cannot be used.

    Covers both "no API key configured" and "endpoint unreachable / timed out".
    The narrative generators catch THIS to switch to offline fallback content
    (see fallback_content.py), so a missing key or a network blip degrades to
    authored content instead of crashing the pipeline. Genuine HTTP errors
    (a 4xx/5xx from a reachable endpoint) still surface as requests.HTTPError.
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
    """Send a chat completion request to the hosted API and return the text.

    The API key/base/model are read from the environment (populated from the repo
    .env at import). Explicitly passing base_url/model overrides the configured
    values. Raises LLMUnavailable when no key is set or the endpoint can't be
    reached; raises requests.HTTPError on a non-2xx response from a reachable
    endpoint.
    """
    if not API_KEY:
        raise LLMUnavailable(
            "No LLM API key configured. Add LLM_API_KEY to the repo-root .env "
            "(or export it in your shell). See .env for the template."
        )

    resolved_base = base_url or API_BASE_URL
    resolved_model = model or API_MODEL
    headers = {"Authorization": f"Bearer {API_KEY}"}

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
        # "Not connected": endpoint unreachable or hung. Re-raise as the typed
        # exception so generators can switch to offline fallback content.
        raise LLMUnavailable(
            f"LLM endpoint unreachable at {resolved_base}: {exc}"
        ) from exc
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
