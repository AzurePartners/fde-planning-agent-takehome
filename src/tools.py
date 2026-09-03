"""
The agent's hands -- three tools it can call.

Each tool is a plain Python function. The model learns they exist through
`TOOL_SCHEMAS` (the menu it sees) and your code runs them through
`TOOL_DISPATCH` (name -> function). The two must stay in sync: add a tool,
add it to both.

Every tool returns a JSON-encoded string, because the result is fed straight
back into the model as message content.

Modes (TOOLS_MODE environment variable):
  replay  (default)  Network calls are served from fixtures/tools.json, so the
                     app and the tests run offline and deterministically.
  live               Real DuckDuckGo search and real HTTP fetches.

Argument validation runs in BOTH modes -- replay only short-circuits the
network, it does not skip the guards.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import urllib.parse
from typing import Any

_FIXTURES_PATH = pathlib.Path(
    os.environ.get("FIXTURES_DIR", pathlib.Path(__file__).resolve().parent.parent / "fixtures")
) / "tools.json"

_fixtures: dict[str, Any] | None = None


def _tools_mode() -> str:
    return os.environ.get("TOOLS_MODE", "replay").strip().lower()


def _load_fixtures() -> dict[str, Any]:
    global _fixtures
    if _fixtures is None:
        try:
            _fixtures = json.loads(_FIXTURES_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _fixtures = {}
    return _fixtures


def _fixture_for(tool: str, key: str) -> Any:
    node = _load_fixtures().get(tool, {})
    exact = node.get("exact", {})
    if key in exact:
        return exact[key]
    return node.get("default")


# ---------------------------------------------------------------------------
# Tool 1: web_search
# ---------------------------------------------------------------------------


def web_search(query: str, k: int = 5) -> str:
    """
    Search the web.

    Returns a JSON list of {title, url, snippet}, or a JSON object with an
    "error" key. Never raises -- the model has to be able to read the failure.
    """
    if not isinstance(query, str) or not query.strip():
        return json.dumps({"error": "query must be a non-empty string"})
    k = max(1, min(int(k or 5), 8))
    query = query.strip()

    if _tools_mode() == "replay":
        canned = _fixture_for("web_search", query.lower()) or []
        return json.dumps(canned[:k], ensure_ascii=False)

    try:
        from ddgs import DDGS
        raw = list(DDGS().text(query, max_results=k, region="wt-wt"))
    except Exception as e:  # network errors, rate limits, import errors
        return json.dumps({"error": f"search failed: {type(e).__name__}: {e}"})

    results = [
        {
            "title": (r.get("title") or "").strip(),
            "url": (r.get("href") or "").strip(),
            "snippet": (r.get("body") or "").strip()[:280],
        }
        for r in raw
    ]
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool 2: generate_image
# ---------------------------------------------------------------------------

# Pollinations.ai renders on demand from the URL itself, so this tool is pure
# string construction -- it makes no request. The image is only fetched later,
# when the UI renders the URL.
_POLLINATIONS_BASE = "https://image.pollinations.ai/prompt/"


def generate_image(prompt: str, width: int = 768, height: int = 512, seed: int = 42) -> str:
    """Build an image-generation URL from a prompt. Returns JSON with image_url."""
    if not isinstance(prompt, str) or not prompt.strip():
        return json.dumps({"error": "prompt must be a non-empty string"})
    width = max(64, min(int(width or 768), 1536))
    height = max(64, min(int(height or 512), 1536))
    seed = int(seed or 42)

    encoded = urllib.parse.quote(prompt.strip(), safe="")
    url = (
        f"{_POLLINATIONS_BASE}{encoded}"
        f"?width={width}&height={height}&seed={seed}&nologo=true&model=flux"
    )
    return json.dumps({"image_url": url, "prompt": prompt.strip()})


# ---------------------------------------------------------------------------
# Tool 3: fetch_url
# ---------------------------------------------------------------------------

_HTML_SCRIPT_RE = re.compile(r"<(script|style|noscript).*?</\1>", re.S | re.I)
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def fetch_url(url: str, max_chars: int = 2500) -> str:
    """
    GET a URL and return its visible text with HTML stripped.

    Use when a search snippet looks promising but is too short to answer the
    question. Truncated to `max_chars` to keep the context bill sane.
    """
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return json.dumps({"error": "url must start with http:// or https://"})
    max_chars = max(200, min(int(max_chars or 2500), 8000))

    if _tools_mode() == "replay":
        canned = _fixture_for("fetch_url", url) or {}
        text = str(canned.get("text", ""))
        return json.dumps({"url": url, "text": text[:max_chars]}, ensure_ascii=False)

    try:
        import requests
        resp = requests.get(
            url, timeout=10, headers={"User-Agent": "planning-agent/1.0"}
        )
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        return json.dumps({"error": f"fetch failed: {type(e).__name__}: {e}"})

    text = _HTML_SCRIPT_RE.sub(" ", html)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return json.dumps({"url": url, "text": text[:max_chars]}, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Schemas -- the menu the model sees
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the web. Use this for ANY question about recent events, "
                "prices, opening hours, availability, or anything you cannot "
                "answer from memory alone. Returns a JSON list of results with "
                "title, url and snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query, 3-10 words. Be specific.",
                    },
                    "k": {
                        "type": "integer",
                        "description": "How many results to return (1-8). Default 5.",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": (
                "Generate ONE cover image illustrating the final answer. Call "
                "this at most once, after you have searched. The prompt should "
                "be visual and concrete. Returns a URL that displays a JPG."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "A visual description of the image to create.",
                    }
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": (
                "Fetch the full visible text of a webpage. Use only when a "
                "web_search snippet is too short to answer the question. Pass "
                "the exact 'url' from a previous search result. Returns up to "
                "2500 characters of plain text."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The full http(s) URL to fetch.",
                    }
                },
                "required": ["url"],
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Dispatch -- how your code calls a tool by name
# ---------------------------------------------------------------------------

TOOL_DISPATCH: dict[str, Any] = {
    "web_search": web_search,
    "generate_image": generate_image,
    "fetch_url": fetch_url,
}
