"""
The executor: runs an approved plan, one step at a time.

planner.py writes the plan. This file walks it:

    PLAN     ask the planner for an ordered plan
    APPROVE  show it to a human and wait -- nothing runs before they say yes
    EXECUTE  run one step, with the earlier steps' observations as context
    OBSERVE  reduce each step's output to one line: ok / thin / surprise
    REPLAN   a "surprise" may rewrite the steps that have not run yet

`run_planning_agent` is that whole loop. `execute_step` is one turn of it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from llm_client import chat, chat_with_tools
from planner import Step, revise_plan, write_plan
from tools import TOOL_DISPATCH, TOOL_SCHEMAS


# ---------------------------------------------------------------------------
# Result shapes -- the UI, the run log and the tests all depend on these
# ---------------------------------------------------------------------------


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: str


@dataclass
class StepResult:
    step: Step
    status: str  # "done" | "skipped" | "error"
    text: str = ""
    observation: str = ""
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    image_url: str | None = None
    sources: list[str] = field(default_factory=list)


@dataclass
class PlanRun:
    goal: str
    initial_plan: list[Step]
    final_plan: list[Step]
    revisions: list[dict[str, Any]] = field(default_factory=list)
    step_results: list[StepResult] = field(default_factory=list)
    final_answer: str = ""
    image_url: str | None = None
    sources: list[str] = field(default_factory=list)
    stopped_reason: str = "done"


# ---------------------------------------------------------------------------
# Prompts -- use verbatim; do not edit for Part A
# ---------------------------------------------------------------------------


def _executor_system(step: Step, goal: str, prior_summary: str) -> str:
    return f"""\
You are the Executor for one step of an Itinerary Architect plan.

The user's overall goal is:
  {goal}

Already done before this step:
  {prior_summary or '(nothing yet -- this is the first step)'}

YOUR step right now:
  step {step.n}: {step.goal}
  tool_hint: {step.tool_hint}

Rules:
- Do ONLY this step. Don't try to finish the whole plan in one go.
- If tool_hint is "web_search", "fetch_url", or "generate_image", call
  THAT tool first (1-2 calls max), then write a SHORT result (2-4 sentences).
- If tool_hint is "none", write the answer directly, no tools.
- Cite any factual claim with the source URL in brackets, like [https://...].
- This is the LAST step? Then write a polished, well-structured final
  answer for the user. Otherwise keep it brief -- the planner needs your
  output, not a finished essay.
"""


_OBSERVER_SYSTEM = """\
You write ONE-LINE observations about what just happened in an agent step.

You will be given the step's goal and the text the executor produced.
Return a single sentence (max 25 words) that says either:
- "ok: <what we learned>"  (the step worked)
- "thin: <what's missing>" (the step ran but the result is weak)
- "surprise: <new fact>"   (the step revealed something the plan didn't expect)

Reply with just that one sentence. No markdown. No quotes.
"""


# ---------------------------------------------------------------------------
# Run bookkeeping -- provided. Use these from A2; you should not need to edit
# them. They own the SHAPE of the result; you own the control flow.
# ---------------------------------------------------------------------------


def _stopped(goal: str, plan: list[Step], reason: str, answer: str) -> PlanRun:
    """A run that ended before any step executed.

    Nothing ran, so there is nothing to summarise: the caller gets the plan as
    it stood, a truthful `reason`, and the message to show the user.
    """
    return PlanRun(
        goal=goal,
        initial_plan=list(plan),
        final_plan=list(plan),
        final_answer=answer,
        stopped_reason=reason,
    )


class _RunRecorder:
    """Keeps the books for one run, so `run_planning_agent` can stay a loop.

    You do not have to read the body. What it does for you:

      record_step(result)     remembers the step in order, keeps the FIRST
                              image any step produced, merges that step's
                              sources into the run's list in encounter order
                              without duplicates, and calls on_step_done --
                              swallowing anything the callback raises, because
                              a broken UI must not take the run down.
      record_revision(...)    appends one audit entry to .revisions in the
                              shape the UI and the tests expect. You pass the
                              queue as it looked BEFORE the rewrite and as it
                              looks after; taking that snapshot at the right
                              moment is your job, not this method's.
      revision_count          how many revisions have been recorded so far.
      finish(stopped_reason)  builds the PlanRun: the steps that actually ran,
                              the final answer, the image, the sources.
    """

    def __init__(
        self,
        goal: str,
        initial_plan: list[Step],
        on_step_done: Callable[[StepResult], None] | None = None,
    ) -> None:
        self.goal = goal
        self.initial_plan = list(initial_plan)
        self.step_results: list[StepResult] = []
        self.revisions: list[dict[str, Any]] = []
        self.image_url: str | None = None
        self.sources: list[str] = []
        self._on_step_done = on_step_done

    @property
    def revision_count(self) -> int:
        return len(self.revisions)

    def record_step(self, result: StepResult) -> None:
        self.step_results.append(result)
        if result.image_url and self.image_url is None:
            self.image_url = result.image_url
        for url in result.sources:
            if url and url not in self.sources:
                self.sources.append(url)
        if self._on_step_done is not None:
            try:
                self._on_step_done(result)
            except Exception:
                pass  # a caller's broken callback must not take the run down

    def record_revision(
        self,
        after_step: int,
        trigger: str,
        before: list[Step],
        after: list[Step],
    ) -> None:
        self.revisions.append(
            {
                "after_step": after_step,
                "trigger": trigger,
                "before": [s.to_dict() for s in before],
                "after": [s.to_dict() for s in after],
            }
        )

    def finish(self, stopped_reason: str = "done") -> PlanRun:
        final_answer = ""
        for r in reversed(self.step_results):
            if r.step.tool_hint == "none" and r.text:
                final_answer = r.text
                break
        if not final_answer and self.step_results:
            final_answer = self.step_results[-1].text
        if not final_answer:
            final_answer = "(no final answer produced)"
        return PlanRun(
            goal=self.goal,
            initial_plan=self.initial_plan,
            final_plan=[r.step for r in self.step_results] or list(self.initial_plan),
            revisions=self.revisions,
            step_results=self.step_results,
            final_answer=final_answer,
            image_url=self.image_url,
            sources=self.sources,
            stopped_reason=stopped_reason,
        )

# ===========================================================================
# TODO A2 -- run_planning_agent
# ===========================================================================


def run_planning_agent(
    goal: str,
    plan: list[Step] | None = None,
    approve_plan: Callable[[list[Step]], bool] | None = None,
    on_step_done: Callable[[StepResult], None] | None = None,
    max_steps: int = 5,
    max_revisions: int = 1,
    per_step_tool_calls: int = 2,
) -> PlanRun:
    """
    Run the whole Plan -> Approve -> Act -> Observe -> Replan loop.

    Args:
        goal: what the user asked for.
        plan: a plan the caller already drafted and had approved. When given,
            execute exactly that plan instead of drafting a new one -- an
            approval gate is worthless if the plan can change after it.
        approve_plan: human-in-the-loop gate. Called once with the plan BEFORE
            anything executes. Returning False must abort the run with zero
            tool calls. None means "no gate".
        on_step_done: progress callback so a UI can stream results. Hand it to
            the recorder; the recorder makes it best-effort.
        max_steps: cap on total plan length, revisions included.
        max_revisions: how many times the plan may be rewritten mid-run.
        per_step_tool_calls: tool-call budget inside a single step.

    The shape, in order:

        1. a blank goal          stop, without calling the planner
        2. the plan              use the one passed in, or write_plan(...);
                                 stop if it comes back empty
        3. approve_plan(...)     stop if the human says no
        4. one step at a time    execute_step(...), then recorder.record_step()
        5. a "surprise" may      revise_plan(...) over the steps still queued,
           rewrite what is left  then recorder.record_revision()
        6. recorder.finish()     hand back the PlanRun

    Contract:
      - Blank goal: no model call at all. stopped_reason "error", final_answer
        exactly "Please type a goal first."
      - No plan to run: stopped_reason "error" and a final_answer that tells
        the user to rephrase (must contain "couldn't draft a plan").
      - Gate refused: stopped_reason "cancelled", final_answer exactly
        "Cancelled before any tool ran.", and execute_step never called.
      - Otherwise steps execute in order, each one seeing a summary of what the
        earlier steps observed, and stopped_reason is "done".
      - An observation beginning with "surprise" may trigger revise_plan on the
        steps that have NOT run yet -- at most max_revisions times, and never
        when the queue is already empty.
      - A revision has to record the queue as it stood BEFORE the rewrite, so
        take that snapshot before you replace the queue, not after.

    `_stopped` and `_RunRecorder` above are provided. They own the result
    shapes: the audit-trail keys, the image and source merging, the callback
    guard, the final-answer fallback. You own the control flow.

    Returns:
        A PlanRun. This function does not raise for ordinary failures -- it
        reports them through stopped_reason and final_answer.
    """
    raise NotImplementedError("TODO A2: implement the plan/act/observe/replan loop")


# ===========================================================================
# execute_step -- provided
# ===========================================================================


def execute_step(
    step: Step,
    goal: str,
    prior_summary: str = "",
    max_tool_calls: int = 2,
) -> StepResult:
    """
    Run exactly one step of the plan and report what happened.

    A step whose tool_hint is "none" is answered from the model directly.
    Anything else goes through the tool-calling path, capped at
    `max_tool_calls` calls and limited to the tools that step is entitled to.
    """
    if step.tool_hint == "none":
        return _execute_text_step(step, goal, prior_summary)
    return _execute_tool_step(step, goal, prior_summary, max_tool_calls)


# ---------------------------------------------------------------------------
# Provided helpers -- read them; you should not need to change them
# ---------------------------------------------------------------------------


def _execute_text_step(step: Step, goal: str, prior_summary: str) -> StepResult:
    text = chat(
        [
            {"role": "system", "content": _executor_system(step, goal, prior_summary)},
            {"role": "user", "content": "Do your step now."},
        ],
        temperature=0.4,
        max_tokens=900,
    ).strip()
    observation = _observe(step, text)
    return StepResult(step=step, status="done", text=text, observation=observation)


def _execute_tool_step(
    step: Step,
    goal: str,
    prior_summary: str,
    max_tool_calls: int,
) -> StepResult:
    allowed = _allowed_schemas_for(step.tool_hint)
    if not allowed:
        return _execute_text_step(step, goal, prior_summary)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": _executor_system(step, goal, prior_summary)},
        {"role": "user", "content": "Do your step now. Call the tool first if useful."},
    ]
    tool_calls_used = 0
    captured_calls: list[ToolCallRecord] = []
    image_url: str | None = None
    sources: list[str] = []

    while True:
        active = allowed if tool_calls_used < max_tool_calls else []
        if active:
            msg = chat_with_tools(messages, tools=active, temperature=0.2, max_tokens=700)
        else:
            messages.append(
                {
                    "role": "user",
                    "content": "Tool budget for this step is used up. Write your short result now.",
                }
            )
            text = chat(messages, temperature=0.3, max_tokens=700).strip()
            observation = _observe(step, text)
            return StepResult(
                step=step,
                status="done",
                text=text,
                observation=observation,
                tool_calls=captured_calls,
                image_url=image_url,
                sources=sources,
            )

        tcs = getattr(msg, "tool_calls", None) or []
        if not tcs:
            text = (msg.content or "").strip()
            observation = _observe(step, text)
            return StepResult(
                step=step,
                status="done",
                text=text,
                observation=observation,
                tool_calls=captured_calls,
                image_url=image_url,
                sources=sources,
            )

        messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in tcs
                ],
            }
        )

        for tc in tcs:
            name = tc.function.name
            try:
                args = json.loads(tc.function.arguments or "{}")
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}
            fn = TOOL_DISPATCH.get(name)
            if fn is None:
                result_str = json.dumps({"error": f"unknown tool: {name}"})
            else:
                try:
                    result_str = fn(**args)
                except Exception as e:
                    result_str = json.dumps({"error": f"{type(e).__name__}: {e}"})
            captured_calls.append(ToolCallRecord(name=name, arguments=args, result=result_str))
            tool_calls_used += 1

            if name == "generate_image" and image_url is None:
                try:
                    obj = json.loads(result_str)
                    if isinstance(obj, dict) and obj.get("image_url"):
                        image_url = obj["image_url"]
                except json.JSONDecodeError:
                    pass
            if name == "web_search":
                try:
                    obj = json.loads(result_str)
                    if isinstance(obj, list):
                        for item in obj:
                            if isinstance(item, dict) and item.get("url"):
                                sources.append(item["url"])
                except json.JSONDecodeError:
                    pass

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                }
            )


def _allowed_schemas_for(tool_hint: str) -> list[dict[str, Any]]:
    if tool_hint == "web_search":
        keep = {"web_search", "fetch_url"}
    elif tool_hint == "fetch_url":
        keep = {"fetch_url"}
    elif tool_hint == "generate_image":
        keep = {"generate_image"}
    else:
        return []
    return [t for t in TOOL_SCHEMAS if t["function"]["name"] in keep]


def _observe(step: Step, text: str) -> str:
    if not text:
        return "thin: step produced no text"
    try:
        raw = chat(
            [
                {"role": "system", "content": _OBSERVER_SYSTEM},
                {
                    "role": "user",
                    "content": f"Step goal: {step.goal}\n\nExecutor wrote:\n{text[:1200]}",
                },
            ],
            temperature=0.1,
            max_tokens=80,
        ).strip()
    except Exception:
        return "ok: (observer unavailable)"
    if not raw:
        return "ok"
    return raw.splitlines()[0][:200]
