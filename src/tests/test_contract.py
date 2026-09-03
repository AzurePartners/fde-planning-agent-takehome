"""
Contract tests -- your fast feedback loop for Part A.

These run offline: no API key, no network. Where the code would call a model
we patch it, so a full run takes about a second and costs nothing.

    cd src && pytest -q

This file is deliberately NOT the full grading set. It checks that you have
understood each contract, not that you have covered every edge of it. Passing
everything here is the floor, not the ceiling -- read the docstrings in
planner.py and agent.py for the behaviour you are actually held to.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any
from unittest.mock import patch

import pytest

SRC = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SRC))


# ===========================================================================
# Smoke
# ===========================================================================


def test_modules_import():
    """Every module must import cleanly, with no API key present."""
    import planner  # noqa: F401
    import agent  # noqa: F401
    import tools  # noqa: F401
    import itinerary_app  # noqa: F401


def test_replay_mode_needs_no_key(monkeypatch):
    """The default runtime is offline. If this fails, nothing else can run."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("LLM_MODE", "replay")
    from llm_client import chat
    from planner import PLANNER_SYSTEM

    reply = chat([{"role": "system", "content": PLANNER_SYSTEM},
                  {"role": "user", "content": "User's goal:\nPlan a day in Kyoto\n\nWrite the plan now."}])
    assert json.loads(reply)["steps"], "replay must serve a usable plan offline"


def test_tools_replay_is_offline(monkeypatch):
    monkeypatch.setenv("TOOLS_MODE", "replay")
    import tools

    results = json.loads(tools.web_search("anything at all"))
    assert isinstance(results, list) and results and "url" in results[0]


# ===========================================================================
# write_plan -- provided; these should pass before you start
# ===========================================================================


def test_write_plan_blank_goal_returns_single_clarifying_step():
    from planner import write_plan

    plan = write_plan("   ")
    assert isinstance(plan, list) and len(plan) == 1
    assert plan[0].tool_hint == "none"


def test_write_plan_parses_a_clean_reply():
    from planner import write_plan

    reply = json.dumps({"steps": [
        {"n": 1, "goal": "Search Kyoto attractions", "tool_hint": "web_search"},
        {"n": 2, "goal": "Generate cover image", "tool_hint": "generate_image"},
        {"n": 3, "goal": "Write the itinerary", "tool_hint": "none"},
    ]})
    with patch("planner.chat", return_value=reply) as mock_chat:
        plan = write_plan("Plan Kyoto")
    assert mock_chat.called, "write_plan must call the model"
    assert [s.tool_hint for s in plan] == ["web_search", "generate_image", "none"]


def test_write_plan_uses_planner_system_verbatim():
    from planner import write_plan, PLANNER_SYSTEM

    reply = json.dumps({"steps": [{"n": 1, "goal": "go", "tool_hint": "none"}]})
    with patch("planner.chat", return_value=reply) as mock_chat:
        write_plan("plan a trip")
    args, kwargs = mock_chat.call_args
    messages = args[0] if args else kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == PLANNER_SYSTEM


def test_write_plan_survives_a_garbage_reply():
    from planner import write_plan

    with patch("planner.chat", return_value="sorry, just chatting today"):
        plan = write_plan("look up cats")
    assert plan, "a malformed reply must still leave the agent something to run"


# ===========================================================================
# A1 -- revise_plan
# ===========================================================================


def test_revise_plan_with_nothing_left_returns_empty():
    from planner import revise_plan

    assert revise_plan(goal="x", done=[], remaining=[], observation="ok") == []


def test_revise_plan_keeps_approved_steps_on_garbage_reply():
    from planner import revise_plan, Step

    remaining = [Step(n=2, goal="keep me", tool_hint="none")]
    with patch("planner.chat", return_value="<<not json>>"):
        out = revise_plan(goal="x", done=[], remaining=remaining, observation="surprise: hmm")
    assert out == remaining


def test_revise_plan_tells_the_model_what_happened():
    from planner import revise_plan, REVISER_SYSTEM, Step

    captured: dict[str, Any] = {}

    def capture(messages, temperature=0.7, max_tokens=1024):
        captured["messages"] = messages
        return json.dumps({"steps": []})

    with patch("planner.chat", side_effect=capture):
        revise_plan(
            goal="Plan my Kyoto day",
            done=[(Step(n=1, goal="search done", tool_hint="web_search"), "ok: found 5 things")],
            remaining=[Step(n=2, goal="write itinerary", tool_hint="none")],
            observation="surprise: it is raining all day",
        )
    assert captured["messages"][0]["content"] == REVISER_SYSTEM
    user_msg = captured["messages"][1]["content"]
    for needle in ("Plan my Kyoto day", "surprise: it is raining all day", "step 1", "step 2"):
        assert needle in user_msg, f"the reviser was not told about: {needle}"


# ===========================================================================
# execute_step -- provided
# ===========================================================================


class _FakeFn:
    def __init__(self, name: str, arguments: str):
        self.name, self.arguments = name, arguments


class _FakeToolCall:
    def __init__(self, id: str, name: str, arguments: str):
        self.id, self.type, self.function = id, "function", _FakeFn(name, arguments)


class _FakeMsg:
    def __init__(self, content: str | None = None, tool_calls: list[Any] | None = None):
        self.content, self.tool_calls = content, tool_calls


def test_execute_step_text_hint_never_touches_tools():
    from planner import Step
    from agent import execute_step

    with patch("agent.chat", return_value="text answer") as mc, \
         patch("agent.chat_with_tools") as mct, \
         patch("agent._observe", return_value="ok: looks fine"):
        r = execute_step(Step(n=1, goal="summarize", tool_hint="none"), goal="x")
    assert r.status == "done" and r.text == "text answer"
    assert mc.called and not mct.called
    assert r.tool_calls == []


def test_execute_step_search_runs_the_tool_and_keeps_the_source():
    from planner import Step
    from agent import execute_step

    turns = [
        _FakeMsg(tool_calls=[_FakeToolCall("c1", "web_search", '{"query": "kyoto", "k": 2}')]),
        _FakeMsg(content="found stuff [https://example.org/city-guide]"),
    ]
    with patch("agent.chat_with_tools", side_effect=turns), \
         patch("agent._observe", return_value="ok: searched"):
        r = execute_step(Step(n=1, goal="search kyoto", tool_hint="web_search"), goal="plan kyoto")
    assert r.status == "done"
    assert [c.name for c in r.tool_calls] == ["web_search"]
    assert "https://example.org/city-guide" in r.sources


# ===========================================================================
# A2 -- run_planning_agent
# ===========================================================================


def _step(n: int, hint: str = "none", goal: str = "do thing"):
    from planner import Step
    return Step(n=n, goal=goal, tool_hint=hint)


def _arg(args, kwargs, name: str, index: int):
    """Read an argument whether you passed it positionally or by name."""
    if name in kwargs:
        return kwargs[name]
    return args[index] if len(args) > index else None


def test_blank_goal_costs_nothing():
    from agent import run_planning_agent

    with patch("agent.write_plan") as mp:
        r = run_planning_agent("   ")
    assert r.stopped_reason == "error"
    assert r.final_answer == "Please type a goal first."
    assert not mp.called, "a blank goal must not reach the planner"


def test_refusing_the_gate_runs_nothing():
    from agent import run_planning_agent

    with patch("agent.write_plan", return_value=[_step(1)]), \
         patch("agent.execute_step") as me:
        r = run_planning_agent("g", approve_plan=lambda p: False)
    assert r.stopped_reason == "cancelled"
    assert r.final_answer == "Cancelled before any tool ran."
    assert not me.called, "the approve gate must be able to stop every tool call"


def test_steps_execute_in_order_and_the_last_narrative_step_is_the_answer():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "none", "step one"), _step(2, "none", "step two")]
    results = [
        StepResult(step=plan[0], status="done", text="r1 text", observation="ok: r1"),
        StepResult(step=plan[1], status="done", text="r2 text final", observation="ok: r2"),
    ]
    with patch("agent.write_plan", return_value=plan), \
         patch("agent.execute_step", side_effect=results) as me:
        r = run_planning_agent("plan it", max_revisions=0)
    assert me.call_count == 2
    assert r.stopped_reason == "done"
    assert r.final_answer == "r2 text final"
    assert [s.n for s in r.final_plan] == [1, 2]


def test_a_surprise_can_rewrite_the_rest_of_the_plan():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "web_search", "search"), _step(2, "none", "old final")]
    revised = [_step(2, "none", "new final after surprise")]
    results = [
        StepResult(step=plan[0], status="done", text="!!", observation="surprise: rain all day"),
        StepResult(step=revised[0], status="done", text="revised answer", observation="ok"),
    ]
    with patch("agent.write_plan", return_value=plan), \
         patch("agent.revise_plan", return_value=revised) as mr, \
         patch("agent.execute_step", side_effect=results):
        r = run_planning_agent("g", max_revisions=1)
    assert mr.called
    assert len(r.revisions) == 1 and r.revisions[0]["after_step"] == 1
    assert r.final_answer == "revised answer"


def test_empty_plan_is_an_error():
    from agent import run_planning_agent

    with patch("agent.write_plan", return_value=[]), \
         patch("agent.execute_step") as me:
        r = run_planning_agent("g")
    assert r.stopped_reason == "error"
    assert "couldn't draft a plan" in r.final_answer
    assert not me.called


def test_later_steps_see_what_earlier_steps_observed():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "web_search", "look it up"), _step(2, "none", "write it up")]
    results = [
        StepResult(step=plan[0], status="done", text="a", observation="ok: found hours"),
        StepResult(step=plan[1], status="done", text="final", observation="ok"),
    ]
    seen: list[str] = []

    def fake_execute(*args, **kwargs):
        seen.append(_arg(args, kwargs, "prior_summary", 2) or "")
        return results[len(seen) - 1]

    with patch("agent.write_plan", return_value=plan), \
         patch("agent.execute_step", side_effect=fake_execute):
        run_planning_agent("g", max_revisions=0)
    assert seen[0] == "", "the first step has nothing to look back on"
    assert "found hours" in seen[1], \
        "step 2 has to see what step 1 observed, or the steps are just a list"


def test_tool_budget_is_passed_down():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "web_search")]
    with patch("agent.write_plan", return_value=plan), \
         patch("agent.execute_step", return_value=StepResult(
             step=plan[0], status="done", text="t", observation="ok")) as me:
        run_planning_agent("g", max_revisions=0, per_step_tool_calls=3)
    args, kwargs = me.call_args
    assert _arg(args, kwargs, "max_tool_calls", 3) == 3


def test_no_replan_when_the_queue_is_already_empty():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "none")]
    with patch("agent.write_plan", return_value=plan), \
         patch("agent.revise_plan") as mr, \
         patch("agent.execute_step", side_effect=[StepResult(
             step=plan[0], status="done", text="t",
             observation="surprise: late news")]):
        run_planning_agent("g", max_revisions=2)
    assert not mr.called, "revising an empty queue costs a call and can only do harm"


def test_the_progress_callback_sees_every_step():
    from agent import run_planning_agent, StepResult

    plan = [_step(1, "none"), _step(2, "none")]
    results = [
        StepResult(step=plan[0], status="done", text="a", observation="ok"),
        StepResult(step=plan[1], status="done", text="b", observation="ok"),
    ]
    seen: list[int] = []
    with patch("agent.write_plan", return_value=plan), \
         patch("agent.execute_step", side_effect=results):
        r = run_planning_agent(
            "g", on_step_done=lambda res: seen.append(res.step.n), max_revisions=0
        )
    assert seen == [1, 2], "hand on_step_done to the recorder, or a UI never updates"
    assert r.stopped_reason == "done"


def test_the_run_recorder_does_the_bookkeeping_for_you():
    """Not a test of your code -- a worked example of the provided recorder.

    This one is green before you start. Read it: it is shorter than the
    docstring and it shows the three calls A2 needs.
    """
    from agent import _RunRecorder, StepResult

    plan = [_step(1, "web_search"), _step(2, "none")]
    run = _RunRecorder(goal="g", initial_plan=plan)
    run.record_step(StepResult(step=plan[0], status="done", text="a",
                               observation="surprise: rain", sources=["https://a"]))
    run.record_revision(after_step=1, trigger="surprise: rain",
                        before=[_step(2, "generate_image")], after=[_step(2, "none")])
    run.record_step(StepResult(step=plan[1], status="done", text="the answer",
                               observation="ok", sources=["https://a", "https://b"]))
    result = run.finish(stopped_reason="done")

    assert result.final_answer == "the answer"
    assert result.sources == ["https://a", "https://b"]
    assert run.revision_count == 1
    assert [s.n for s in result.final_plan] == [1, 2]


# ===========================================================================
# Provided parser -- these should pass before you write a line
# ===========================================================================


def test_parser_coerces_an_invalid_tool_hint():
    from planner import _parse_plan

    steps = _parse_plan(json.dumps({"steps": [{"n": 1, "goal": "do thing", "tool_hint": "telepathy"}]}))
    assert steps and steps[0].tool_hint == "none"


def test_parser_strips_markdown_fences():
    from planner import _parse_plan

    raw = '```json\n{"steps": [{"n": 1, "goal": "x", "tool_hint": "none"}]}\n```'
    assert len(_parse_plan(raw)) == 1
