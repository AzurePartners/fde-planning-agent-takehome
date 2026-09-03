"""
Run one goal end-to-end and print the trace. No UI, no key needed.

    python run_once.py "Plan a two-day team offsite in Lisbon for eight people."
    LLM_MODE=live python run_once.py "..."      # with your own key

Useful as a fast loop while you work, and as the thing a reviewer runs first.
"""

from __future__ import annotations

import sys

from dotenv import load_dotenv

load_dotenv()

from agent import run_planning_agent  # noqa: E402


def main() -> int:
    goal = " ".join(sys.argv[1:]).strip() or "Plan a one-day layover in Tokyo on a 5000 yen budget."
    print(f"goal: {goal}\n")

    def approve(plan) -> bool:
        print("proposed plan:")
        for s in plan:
            print(f"  {s.n}. [{s.tool_hint}] {s.goal}")
        print()
        return True

    def on_step(result) -> None:
        print(f"  step {result.step.n} [{result.step.tool_hint}] {result.status}")
        for call in result.tool_calls:
            print(f"      tool  {call.name}({call.arguments})")
        print(f"      obs   {result.observation}")

    run = run_planning_agent(goal, approve_plan=approve, on_step_done=on_step)

    print(f"\nstopped_reason : {run.stopped_reason}")
    for rev in run.revisions:
        print(f"replan after step {rev['after_step']}: {rev['trigger']}")
    if run.image_url:
        print(f"image          : {run.image_url}")
    if run.sources:
        print("sources        :")
        for u in run.sources:
            print(f"  - {u}")
    print("\n--- final answer ---")
    print(run.final_answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
