"""
LLM client -- an OpenAI-compatible chat wrapper with a deterministic replay mode.

Three modes, selected by the LLM_MODE environment variable:

  replay   (default)  No network, no API key. Every call is served from
                      fixtures/llm.json. The app and the test suite both run
                      offline and deterministically.
  live                Real calls against LLM_BASE_URL using LLM_API_KEY.
  record              Real calls, and every request/response pair is appended
                      to the cassette so it can be replayed later.

Why replay exists: an agent is a non-deterministic system, and you cannot
iterate on a system you cannot re-run. Replay makes a run reproducible --
same input, same trace, every time, on any machine, at zero cost.

Public API:
    chat(messages, ...)             -> str
    chat_with_tools(messages, ...)  -> assistant message object
    generate_text(prompt, ...)      -> str
"""

from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re
import uuid
from typing import Any

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LLM_MODE = os.environ.get("LLM_MODE", "replay").strip().lower()

# Any OpenAI-compatible endpoint works: OpenRouter, OpenAI, Together, a local
# vLLM/Ollama server. Only used when LLM_MODE is "live" or "record".
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")
LLM_API_KEY_ENV = "LLM_API_KEY"

LLM_TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT_S", "60"))

_FIXTURES_DIR = pathlib.Path(
    os.environ.get("FIXTURES_DIR", pathlib.Path(__file__).resolve().parent.parent / "fixtures")
)
_CASSETTE_PATH = _FIXTURES_DIR / "llm.json"


def _mode() -> str:
    """Read the mode fresh each call so tests can flip it with monkeypatch."""
    return os.environ.get("LLM_MODE", LLM_MODE).strip().lower()


# ---------------------------------------------------------------------------
# Live client (lazy -- never constructed in replay mode, so no key needed)
# ---------------------------------------------------------------------------

_client: Any = None


def _get_client() -> Any:
    global _client
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "The `openai` package isn't installed. Run:  pip install -r requirements.txt"
        ) from e

    api_key = os.environ.get(LLM_API_KEY_ENV)
    if not api_key:
        raise RuntimeError(
            f"LLM_MODE={_mode()} needs a real API key, but {LLM_API_KEY_ENV} is not set.\n"
            "  1) cp .env.example .env\n"
            f"  2) put your key in {LLM_API_KEY_ENV}=...\n"
            "  3) or run with LLM_MODE=replay (the default) to work offline."
        )
    _client = OpenAI(api_key=api_key, base_url=LLM_BASE_URL, timeout=LLM_TIMEOUT_S)
    return _client


# ---------------------------------------------------------------------------
# Cassette
# ---------------------------------------------------------------------------

_cassette: dict[str, Any] | None = None


def _load_cassette() -> dict[str, Any]:
    global _cassette
    if _cassette is None:
        try:
            _cassette = json.loads(_CASSETTE_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            _cassette = {"exact": {}, "defaults": {}}
    return _cassette


def _request_key(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    """Stable hash of a request. Exact-match layer of the replay lookup."""
    payload = {
        "messages": messages,
        "tools": sorted(t["function"]["name"] for t in (tools or [])),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _record(key: str, value: Any) -> None:
    cassette = _load_cassette()
    cassette.setdefault("exact", {})[key] = value
    _FIXTURES_DIR.mkdir(parents=True, exist_ok=True)
    _CASSETTE_PATH.write_text(
        json.dumps(cassette, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Replay: intent routing
# ---------------------------------------------------------------------------
#
# An exact cassette hit is preferred, but a miss must never break the run --
# a candidate who formats a user message slightly differently should still get
# a working app. So on a miss we fall back to routing by INTENT, detected from
# the system prompt, and serve a shape-correct canned response.

_INTENT_MARKERS: list[tuple[str, str]] = [
    ("You are the Planner for", "planner"),
    ("You are the Plan Reviser", "reviser"),
    ("You write ONE-LINE observations", "observer"),
    ("You are the Executor for one step", "executor"),
]


def _detect_intent(messages: list[dict[str, Any]]) -> str:
    system = ""
    for m in messages:
        if m.get("role") == "system":
            system = str(m.get("content") or "")
            break
    for marker, intent in _INTENT_MARKERS:
        if marker in system:
            return intent
    return "executor"


_GOAL_RE = re.compile(r"(?:User's goal|Original goal|overall goal is):\s*\n\s*(.+)")
_HINT_RE = re.compile(r"tool_hint:\s*(\w+)")
# Anchored on the tool_hint line that follows it, so a prior-steps summary
# earlier in the same prompt cannot be mistaken for the current step.
_STEP_GOAL_RE = re.compile(r"step\s+\d+:\s*(.+)\n\s*tool_hint:")


def _joined(messages: list[dict[str, Any]]) -> str:
    return "\n".join(str(m.get("content") or "") for m in messages)


def _extract_goal(messages: list[dict[str, Any]]) -> str:
    m = _GOAL_RE.search(_joined(messages))
    return m.group(1).strip() if m else "the user's goal"


def _extract_tool_hint(messages: list[dict[str, Any]]) -> str:
    m = _HINT_RE.search(_joined(messages))
    return m.group(1).strip() if m else "none"


def _extract_step_goal(messages: list[dict[str, Any]]) -> str:
    hits = _STEP_GOAL_RE.findall(_joined(messages))
    return hits[-1].strip() if hits else "this step"


def _fill(template: str, messages: list[dict[str, Any]]) -> str:
    """Substitute {{goal}} / {{step_goal}} tokens. Deliberately NOT str.format
    -- the planner fixtures are JSON and are full of literal braces."""
    return (
        template.replace("{{goal}}", _extract_goal(messages))
        .replace("{{step_goal}}", _extract_step_goal(messages))
    )


def _apply_rules(node: Any, messages: list[dict[str, Any]]) -> str:
    """A fixture node is either a plain string, or {"rules": [...], "default": ...}
    where each rule fires when its "contains" string appears in the request."""
    if isinstance(node, str):
        return _fill(node, messages)
    if isinstance(node, dict):
        haystack = _joined(messages)
        for rule in node.get("rules", []):
            if str(rule.get("contains", "")) in haystack:
                return _fill(str(rule.get("reply", "")), messages)
        return _fill(str(node.get("default", "")), messages)
    return ""


def _replay_text(messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None) -> str:
    cassette = _load_cassette()
    hit = cassette.get("exact", {}).get(_request_key(messages, tools))
    if isinstance(hit, str):
        return hit
    if isinstance(hit, dict) and isinstance(hit.get("content"), str):
        return hit["content"]

    defaults = cassette.get("defaults", {})
    intent = _detect_intent(messages)
    if intent == "executor":
        by_hint = defaults.get("executor", {})
        node = by_hint.get(_extract_tool_hint(messages), by_hint.get("none", ""))
        return _apply_rules(node, messages)
    return _apply_rules(defaults.get(intent, ""), messages)


# ---------------------------------------------------------------------------
# Replay: fake assistant message objects (mirror the OpenAI SDK shape)
# ---------------------------------------------------------------------------


class _ReplayFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class _ReplayToolCall:
    def __init__(self, name: str, arguments: dict[str, Any]):
        self.id = f"replay_{uuid.uuid4().hex[:8]}"
        self.type = "function"
        self.function = _ReplayFunction(name, json.dumps(arguments, ensure_ascii=False))


class _ReplayMessage:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"


def _replay_tool_call(messages: list[dict[str, Any]], allowed: set[str]) -> _ReplayMessage | None:
    """Decide whether the replayed model wants a tool this turn.

    Rule: if the conversation already contains a tool result, the tool has run
    and the model should now write its text. Otherwise ask for one tool call.
    This is stateless and deterministic -- it does not depend on how many times
    the candidate's loop has called us.
    """
    if any(m.get("role") == "tool" for m in messages):
        return None

    hint = _extract_tool_hint(messages)
    step_goal = _extract_step_goal(messages)
    if hint == "web_search" and "web_search" in allowed:
        return _ReplayMessage(tool_calls=[_ReplayToolCall("web_search", {"query": step_goal[:80], "k": 5})])
    if hint == "fetch_url" and "fetch_url" in allowed:
        return _ReplayMessage(tool_calls=[_ReplayToolCall("fetch_url", {"url": "https://example.org/city-guide"})])
    if hint == "generate_image" and "generate_image" in allowed:
        prompt = f"travel poster illustrating {_extract_goal(messages)}"
        return _ReplayMessage(tool_calls=[_ReplayToolCall("generate_image", {"prompt": prompt})])
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def chat(
    messages: list[dict[str, Any]],
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    """Single-turn or multi-turn chat. Returns the assistant's text."""
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of role/content dicts")

    mode = _mode()
    if mode == "replay":
        return _replay_text(messages, None)

    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content or ""
    if mode == "record":
        _record(_request_key(messages, None), text)
    return text


def chat_with_tools(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    temperature: float = 0.2,
    max_tokens: int = 1024,
):
    """Chat with tool-calling enabled.

    Returns the FULL assistant message object, not just the text -- inspect
    `.content` for the reply and `.tool_calls` for the tools the model wants
    executed. Feed each tool's output back as a {"role": "tool", ...} message.
    """
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list of role/content dicts")
    if not isinstance(tools, list) or not tools:
        raise ValueError("tools must be a non-empty list of OpenAI tool schemas")

    mode = _mode()
    if mode == "replay":
        allowed = {t["function"]["name"] for t in tools}
        call = _replay_tool_call(messages, allowed)
        if call is not None:
            return call
        return _ReplayMessage(content=_replay_text(messages, tools))

    response = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    msg = response.choices[0].message
    if mode == "record" and not getattr(msg, "tool_calls", None):
        _record(_request_key(messages, tools), msg.content or "")
    return msg


def generate_text(
    prompt: str,
    system_instruction: str = "",
    temperature: float = 0.7,
    max_tokens: int = 1024,
) -> str:
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})
    return chat(messages, temperature=temperature, max_tokens=max_tokens)


# ----- Self-check -----------------------------------------------------------

if __name__ == "__main__":
    print(f"mode      : {_mode()}")
    print(f"cassette  : {_CASSETTE_PATH}  ({'found' if _CASSETTE_PATH.exists() else 'MISSING'})")
    if _mode() == "replay":
        print(f"entries   : {len(_load_cassette().get('exact', {}))} exact + intent defaults")
    else:
        print(f"base url  : {LLM_BASE_URL}")
        print(f"model     : {LLM_MODEL}")
        print(f"api key   : {'set' if os.environ.get(LLM_API_KEY_ENV) else 'NOT SET'}")
    print(f"reply     : {generate_text('Reply with exactly the word: ready')!r}")
