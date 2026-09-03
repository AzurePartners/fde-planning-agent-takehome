"""
The planner: turns a goal into an ordered plan, and revises that plan mid-run.

The agent works in two halves. This file WRITES plans; agent.py EXECUTES them.
Splitting them buys two things a reactive tool-calling loop cannot give you:

  1. The user can see and approve the plan before any tool fires.
  2. When a step surprises us, we can rewrite the steps that have not run yet.

The data shapes and the prompts are fixed. What is missing is the two model
calls and the wiring around them.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

from llm_client import chat


# ---------------------------------------------------------------------------
# Data shape -- do not change; the UI and the tests depend on it
# ---------------------------------------------------------------------------


@dataclass
class Step:
    """One line of the plan. Steps execute in order."""

    n: int            # 1-based step number
    goal: str         # one-sentence goal for this step
    tool_hint: str    # one of: "web_search", "fetch_url", "generate_image", "none"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_VALID_HINTS = {"web_search", "fetch_url", "generate_image", "none"}


# ---------------------------------------------------------------------------
# Prompts -- use verbatim; do not edit for Part A
# ---------------------------------------------------------------------------


PLANNER_SYSTEM = """\
You are the Planner for an Itinerary Architect agent. The user gives you
a goal. You write a SHORT, ORDERED plan -- 3 to 5 steps, never more --
that another agent will execute.

Each step is one of these kinds, declared by `tool_hint`:
- "web_search"     : look up facts on the live web
- "fetch_url"      : read a specific webpage fully (only after a search)
- "generate_image" : make one cover image (use this at most ONCE, near the end)
- "none"           : a pure thinking step (summarize, schedule, decide)

Hard rules:
- 3 to 5 steps total. Never more.
- At most 2 "web_search" steps.
- At most 1 "generate_image" step (skip it for non-visual goals).
- The LAST step is always "none" -- it writes the final answer/itinerary.
- Steps must be ordered. Search BEFORE you fetch. Decide BEFORE you draw.

Reply ONLY with a JSON object in this exact shape, nothing else:

{
  "steps": [
    {"n": 1, "goal": "...", "tool_hint": "web_search"},
    {"n": 2, "goal": "...", "tool_hint": "generate_image"},
    {"n": 3, "goal": "Write the final itinerary.", "tool_hint": "none"}
  ]
}

Do NOT wrap the JSON in markdown fences. Do NOT add commentary.
"""


REVISER_SYSTEM = """\
You are the Plan Reviser. You get:
- the ORIGINAL goal
- the steps that have ALREADY been done, with one-line observations
- the steps still REMAINING (not yet started)
- the most recent OBSERVATION that triggered the revision

Decide whether to:
(a) keep the remaining steps as-is, or
(b) rewrite them to react to what we just learned.

If you keep them, return the remaining list unchanged.
If you rewrite, return a NEW list of remaining steps. You may shorten it,
re-order it, or change tool_hints, but the total run (already-done +
remaining) must still be 5 steps or fewer.

Same JSON shape as the planner. Same rules apply. Renumber so the next
step number continues from where the done-steps left off.

Reply ONLY with the JSON object, no fences, no commentary.
"""


# ===========================================================================
# write_plan -- provided, and a worked example of the house style
# ===========================================================================
#
# Read this before you write A1. It is the same shape of problem: build a user
# message, make one model call with a fixed system prompt, parse defensively,
# and guarantee the caller gets something runnable no matter what came back.


def write_plan(goal: str, max_steps: int = 5) -> list[Step]:
    """
    Turn a user goal into an ordered, executable plan.

    Never raises and never returns an empty list: everything downstream treats
    an empty plan as a hard failure. A blank goal costs no model call, and a
    malformed reply degrades into a minimal plan rather than into nothing.
    """
    if not isinstance(goal, str) or not goal.strip():
        return [Step(n=1, goal="Ask the user for a clearer goal.", tool_hint="none")]

    goal = goal.strip()
    user_msg = f"User's goal:\n{goal}\n\nWrite the plan now."

    try:
        raw = chat(
            [
                {"role": "system", "content": PLANNER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.3,
            max_tokens=500,
        )
    except Exception:
        # An upstream failure must degrade into the fallback below, not take
        # the whole run down with it.
        raw = ""

    steps = _parse_plan(raw, max_steps=max_steps)
    if not steps:
        steps = [
            Step(n=1, goal=f"Search the web for: {goal}", tool_hint="web_search"),
            Step(n=2, goal="Write a short answer using the search results.", tool_hint="none"),
        ]
    return steps


# ===========================================================================
# TODO A1 -- revise_plan
# ===========================================================================


def revise_plan(
    goal: str,
    done: list[tuple[Step, str]],
    remaining: list[Step],
    observation: str,
    max_steps: int = 5,
) -> list[Step]:
    """
    Rewrite the not-yet-executed steps in light of something we just learned.

    Args:
        goal: the original user goal.
        done: (step, observation) pairs for steps that have already run.
        remaining: steps not yet started. Only these may be rewritten.
        observation: the one-line observation that triggered this revision.
        max_steps: cap on the TOTAL run length, done steps included.

    Contract:
      - Uses REVISER_SYSTEM verbatim as the system message.
      - The user message must give the model everything it needs to decide:
        the goal, what has already run and how it went, what is still queued,
        and the triggering observation.
      - Returns [] when there is nothing left to revise.
      - Returned steps are renumbered to continue after the last done step, and
        the total run still respects `max_steps`.
      - NEVER raises. If the model returns garbage, the caller must not silently
        lose the steps the user already approved.

    `_parse_plan` below takes an `expect_start_at=` argument for exactly the
    renumbering case; `write_plan` above does not need it, so it does not show
    it. Read `write_plan` first -- it is the same shape of problem.
    """
    raise NotImplementedError("TODO A1: implement revise_plan")


# ===========================================================================
# Provided helpers -- you may read them, you do not need to change them
# ===========================================================================


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _parse_plan(
    raw: str,
    max_steps: int = 5,
    expect_start_at: int | None = None,
) -> list[Step]:
    """Best-effort: extract a plan from a possibly chatty model reply."""
    if not raw:
        return []
    text = raw.strip()
    fence = _JSON_FENCE_RE.search(text)
    if fence:
        text = fence.group(1).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first : last + 1]

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []

    if not isinstance(obj, dict) or not isinstance(obj.get("steps"), list):
        return []

    steps: list[Step] = []
    next_expected = expect_start_at if expect_start_at is not None else 1
    for raw_step in obj["steps"][:max_steps]:
        if not isinstance(raw_step, dict):
            continue
        goal = str(raw_step.get("goal", "")).strip()
        hint = str(raw_step.get("tool_hint", "none")).strip().lower()
        if not goal:
            continue
        if hint not in _VALID_HINTS:
            hint = "none"
        try:
            n = int(raw_step.get("n", next_expected))
        except (TypeError, ValueError):
            n = next_expected
        steps.append(Step(n=n, goal=goal, tool_hint=hint))
        next_expected = n + 1

    if steps and steps[-1].tool_hint != "none":
        # Make room for the closing step BEFORE appending it. Appending first
        # and truncating after would drop the very step we just added.
        if len(steps) >= max_steps > 1:
            steps = steps[: max_steps - 1]
        steps.append(
            Step(
                n=steps[-1].n + 1,
                goal="Write the final answer using everything gathered so far.",
                tool_hint="none",
            )
        )
        steps = steps[:max_steps]

    return steps
