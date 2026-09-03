"""
Streamlit UI for the planning agent.

Run from this directory:
    streamlit run itinerary_app.py

Three things this surface has to get right:

1. **Plan card** -- the drafted plan is shown before anything runs.
2. **Approve gate** -- no tool fires until a human clicks Approve.
3. **Step trace** -- each step renders as its own card as it completes, and a
   mid-run replan is surfaced rather than hidden.
"""

from __future__ import annotations

import json
import os
import time
from html import escape as _html_escape

import streamlit as st
from dotenv import load_dotenv

from agent import PlanRun, StepResult, execute_step, run_planning_agent
from llm_client import LLM_API_KEY_ENV, LLM_BASE_URL, LLM_MODEL, _mode
from planner import Step, write_plan

load_dotenv()

st.set_page_config(
    page_title="Itinerary Architect",
    page_icon="I",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ----- Styling --------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], .stMarkdown, .stChatMessage {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
.main .block-container { max-width: 900px; padding-top: 1.4rem; padding-bottom: 6rem; }
[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer { visibility: hidden; }

.app-header { padding: 0.4rem 0 1.1rem 0; margin-bottom: 0.4rem; border-bottom: 1px solid #ececf1; }
.app-header h2 { margin: 0; font-size: 1.15rem; font-weight: 600; line-height: 1.2; color: #202123; }
.app-header p { margin: 0.25rem 0 0 0; font-size: 0.85rem; font-weight: 400; color: #6e6e80; }

.empty-state { text-align: center; padding: 2.4rem 1rem 0.6rem 1rem; }
.empty-state h3 { font-weight: 500; color: #353740; margin: 0 0 0.6rem 0; }
.empty-state p { color: #6e6e80; margin: 0; font-size: 0.92rem; }

.answer-card {
    background: #ffffff; border: 1px solid #ececf1; border-radius: 10px;
    padding: 1.1rem 1.3rem; margin: 0.6rem 0; line-height: 1.55; color: #2d2d2d;
}
.answer-card h4 { margin: 0 0 0.5rem 0; font-weight: 600; font-size: 0.95rem; color: #202123; }
.sources { font-size: 0.82rem; color: #6e6e80; margin-top: 0.6rem; }
.sources a { color: #475569; text-decoration: none; border-bottom: 1px dotted #c7c7d1; }
.sources a:hover { color: #1c1917; }

.plan-card {
    background: #fafaf9; border: 1px solid #ececf1; border-radius: 10px;
    padding: 0.9rem 1.1rem; margin: 0.6rem 0;
}
.plan-card h4 {
    margin: 0 0 0.45rem 0; font-size: 0.85rem; font-weight: 600;
    color: #202123; text-transform: uppercase; letter-spacing: 0.05em;
}
.plan-card ol { margin: 0; padding-left: 1.3rem; }
.plan-card li {
    margin: 0.25rem 0; font-size: 0.88rem; color: #353740; line-height: 1.45;
}
.plan-card .hint {
    display: inline-block; font-size: 0.7rem; color: #6e6e80;
    background: #f4f4f5; padding: 0.05rem 0.4rem; border-radius: 3px;
    margin-left: 0.4rem; font-variant-numeric: tabular-nums;
}

.step-card {
    background: #ffffff; border: 1px solid #ececf1; border-radius: 8px;
    padding: 0.7rem 0.95rem; margin: 0.4rem 0;
}
.step-card .head {
    display: flex; align-items: baseline; gap: 0.5rem;
    font-size: 0.82rem; color: #6e6e80; margin-bottom: 0.3rem;
}
.step-card .head .num { font-weight: 700; color: #353740; }
.step-card .head .status { font-variant-numeric: tabular-nums; color: #57534e; }
.step-card .head .hint {
    font-size: 0.7rem; background: #f4f4f5; color: #6e6e80;
    padding: 0.05rem 0.4rem; border-radius: 3px;
}
.step-card .goal {
    font-size: 0.92rem; color: #2d2d2d; margin-bottom: 0.35rem; line-height: 1.45;
}
.step-card .obs {
    font-size: 0.78rem; color: #6e6e80; font-style: italic;
}

.tool-card {
    background: #fafaf9; border: 1px solid #ececf1; border-left: 3px solid #6e6e80;
    border-radius: 6px; padding: 0.5rem 0.75rem; margin: 0.3rem 0 0.3rem 0.8rem;
    font-size: 0.8rem; color: #353740;
}
.tool-card .meta {
    display: flex; justify-content: space-between; color: #6e6e80;
    font-size: 0.7rem; margin-bottom: 0.25rem;
}
.tool-card .name { font-weight: 600; color: #202123; }
.tool-card pre {
    margin: 0.25rem 0 0 0; background: #f4f4f5; padding: 0.45rem 0.6rem;
    border-radius: 4px; font-size: 0.74rem; color: #353740;
    max-height: 200px; overflow: auto; white-space: pre-wrap; word-break: break-word;
}

.revision-card {
    background: #fdf6e3; border: 1px solid #e0d9b8; border-radius: 6px;
    padding: 0.55rem 0.8rem; margin: 0.4rem 0; font-size: 0.82rem; color: #57534e;
}
.revision-card strong { color: #4a3f1a; }

.sidebar-card {
    background: #fafaf9; border: 1px solid #ececf1; border-radius: 8px;
    padding: 0.65rem 0.85rem; margin: 0.4rem 0;
    font-size: 0.82rem; color: #353740;
}
.sidebar-card h4 { margin: 0 0 0.35rem 0; font-size: 0.78rem; font-weight: 600; color: #202123; text-transform: uppercase; letter-spacing: 0.05em; }
.sidebar-card .row { display: flex; justify-content: space-between; padding: 0.15rem 0; font-size: 0.78rem; color: #6e6e80; }
.sidebar-card .row .v { color: #353740; font-variant-numeric: tabular-nums; }
.sidebar-card code { font-size: 0.74rem; background: #f4f4f5; padding: 0.05rem 0.3rem; border-radius: 3px; }

.disclaimer { text-align: center; color: #8e8ea0; font-size: 0.78rem; margin: 1.4rem 0 0 0; }
</style>
""",
    unsafe_allow_html=True,
)




# ----- Session state --------------------------------------------------------

if "history" not in st.session_state:
    st.session_state["history"] = []   # list[(goal, PlanRun)]
if "pending_plan" not in st.session_state:
    st.session_state["pending_plan"] = None    # {"goal": str, "plan": list[Step]}
if "queued_goal" not in st.session_state:
    st.session_state["queued_goal"] = ""


# ----- Sidebar --------------------------------------------------------------

with st.sidebar:
    st.markdown("### Itinerary Architect")
    st.caption("A planning agent. Drafts a plan, asks you to approve, then runs it step by step.")

    st.markdown("#### Settings")
    max_steps = st.slider("Max plan steps", 3, 6, 5,
                          help="Cap on plan length, including any revisions.")
    max_revisions = st.slider("Max revisions", 0, 3, 1,
                              help="How many times the planner can revise the remaining steps "
                                   "after seeing a surprise observation.")
    per_step_calls = st.slider("Tool calls per step", 1, 3, 2,
                               help="Hard cap on tools inside ONE step's executor.")

    mode = _mode()
    api_key_set = bool(os.environ.get(LLM_API_KEY_ENV))
    st.markdown(
        f"""
<div class="sidebar-card">
  <h4>Runtime</h4>
  <div class="row"><span>LLM_MODE</span><span class="v"><code>{_html_escape(mode)}</code></span></div>
  <div class="row"><span>TOOLS_MODE</span><span class="v"><code>{_html_escape(os.environ.get("TOOLS_MODE", "replay"))}</code></span></div>
  <div class="row"><span>model</span><span class="v"><code>{_html_escape(LLM_MODEL if mode != "replay" else "fixtures/llm.json")}</code></span></div>
  <div class="row"><span>endpoint</span><span class="v"><code>{_html_escape(LLM_BASE_URL if mode != "replay" else "offline")}</code></span></div>
  <div class="row"><span>{LLM_API_KEY_ENV}</span><span class="v">{"set" if api_key_set else "not set"}</span></div>
  <div class="row"><span>max steps</span><span class="v">{max_steps}</span></div>
  <div class="row"><span>max revisions</span><span class="v">{max_revisions}</span></div>
</div>
"""
,
        unsafe_allow_html=True,
    )

    if st.session_state["history"]:
        st.markdown("#### Recent")
        for i, (g, _r) in enumerate(reversed(st.session_state["history"][-6:])):
            label = (g[:40] + "...") if len(g) > 40 else g
            if st.button(label, key=f"recent_{i}", use_container_width=True):
                st.session_state["queued_goal"] = g
                st.rerun()


# ----- Header ---------------------------------------------------------------

st.markdown(
    """
<div class="app-header">
  <h2>Itinerary Architect</h2>
  <p>Give me a goal. I'll write a plan, ask you to approve it, then run it step by step.</p>
</div>
""",
    unsafe_allow_html=True,
)


# ----- Starter chips --------------------------------------------------------

STARTERS = [
    "Plan a one-day layover in Tokyo on a 5000 yen budget.",
    "Plan a two-day team offsite in Lisbon for eight people.",
    "Plan a week of evening study for a database systems exam.",
    "Plan a rainy-day itinerary in Kyoto that avoids queues.",
]


def _queue(g: str) -> None:
    st.session_state["queued_goal"] = g
    st.session_state["pending_plan"] = None


if not st.session_state["history"] and st.session_state["pending_plan"] is None:
    st.markdown(
        """
<div class="empty-state">
  <h3>What should we plan?</h3>
  <p>Pick a starter or type your own goal below.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, s in enumerate(STARTERS):
        with cols[i % 2]:
            if st.button(s, key=f"starter_{i}", use_container_width=True):
                _queue(s)
                st.rerun()


# ----- Render helpers -------------------------------------------------------


def _plan_html(plan: list[Step]) -> str:
    items = "".join(
        f'<li>{_html_escape(s.goal)}'
        f'<span class="hint">{_html_escape(s.tool_hint)}</span></li>'
        for s in plan
    )
    return f"<div class='plan-card'><h4>Proposed plan</h4><ol>{items}</ol></div>"


def _step_card_html(result: StepResult) -> str:
    status_glyph = {"done": "+", "skipped": "-", "error": "x"}.get(result.status, "+")
    head = (
        f"<div class='head'>"
        f"<span class='num'>{status_glyph} step {result.step.n}</span>"
        f"<span class='hint'>{_html_escape(result.step.tool_hint)}</span>"
        f"<span class='status'>{_html_escape(result.status)}</span>"
        f"</div>"
    )
    goal = f"<div class='goal'>{_html_escape(result.step.goal)}</div>"
    obs = (
        f"<div class='obs'>observation: {_html_escape(result.observation)}</div>"
        if result.observation else ""
    )
    return f"<div class='step-card'>{head}{goal}{obs}</div>"


def _tool_cards_html(result: StepResult) -> str:
    if not result.tool_calls:
        return ""
    cards = []
    for i, rec in enumerate(result.tool_calls, start=1):
        args = json.dumps(rec.arguments, ensure_ascii=False)
        if len(args) > 200:
            args = args[:200] + " ..."
        try:
            parsed = json.loads(rec.result)
            pretty = json.dumps(parsed, ensure_ascii=False, indent=2)
        except Exception:
            pretty = rec.result
        if len(pretty) > 900:
            pretty = pretty[:900] + "\n... [truncated]"
        cards.append(
            f"<div class='tool-card'>"
            f"<div class='meta'><span class='name'>{i}. {_html_escape(rec.name)}</span>"
            f"<span>args: {_html_escape(args)}</span></div>"
            f"<pre>{_html_escape(pretty)}</pre>"
            f"</div>"
        )
    return "".join(cards)


def _render_run(goal: str, run: PlanRun) -> None:
    # User goal pill
    st.markdown(
        f"""
<div style="display:flex; justify-content:flex-end; margin:1.2rem 0 0.5rem 0;">
  <div style="background:#ececf1; color:#1c1917; padding:0.55rem 0.95rem;
              border-radius:18px 18px 4px 18px; max-width:78%;
              font-size:0.92rem; line-height:1.45;">{_html_escape(goal)}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Plan (initial, before any revisions)
    if run.initial_plan:
        st.markdown(_plan_html(run.initial_plan), unsafe_allow_html=True)

    # ===================================================================
    # TODO A3 -- render the step trace and any mid-run replans
    # ===================================================================
    #
    # A reviewer looking at this page must be able to answer, without opening
    # a terminal: what did the agent do, in what order, what did each step
    # conclude, which tools ran with which arguments, and -- if the plan
    # changed mid-run -- when it changed and what triggered it.
    #
    # `run.step_results` holds one StepResult per executed step.
    # `run.revisions` holds one dict per replan, with keys "after_step",
    # "trigger", "before" and "after".
    #
    # _step_card_html() and _tool_cards_html() are written for you just above
    # and return HTML strings. Render HTML with
    # st.markdown(..., unsafe_allow_html=True), and put per-step detail behind
    # st.expander(...) so the page stays scannable.
    #
    # Escape anything that came from the model or the web before it reaches
    # the page. Delete the caption below when you are done.
    st.caption("TODO A3: render the step trace and revision panel here.")
    # END TODO A3
    # ===================================================================

    # Cover image
    if run.image_url:
        st.image(run.image_url, use_container_width=True)

    # Final answer
    st.markdown(
        f"""
<div class="answer-card">
  <h4>Final answer</h4>
  <div>{_html_escape(run.final_answer).replace(chr(10), '<br/>')}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    # Sources
    if run.sources:
        items = "".join(
            f'<li><a href="{_html_escape(u)}" target="_blank" rel="noopener">{_html_escape(u)}</a></li>'
            for u in run.sources
        )
        st.markdown(
            f"<div class='sources'><strong>Sources</strong><ul>{items}</ul></div>",
            unsafe_allow_html=True,
        )


# ----- Past run history ----------------------------------------------------

for g, r in st.session_state["history"]:
    _render_run(g, r)


# ----- Pending plan (waiting for user approval) ----------------------------

if st.session_state["pending_plan"] is not None:
    pending = st.session_state["pending_plan"]
    st.markdown(
        f"""
<div style="display:flex; justify-content:flex-end; margin:1.2rem 0 0.5rem 0;">
  <div style="background:#ececf1; color:#1c1917; padding:0.55rem 0.95rem;
              border-radius:18px 18px 4px 18px; max-width:78%;
              font-size:0.92rem; line-height:1.45;">{_html_escape(pending['goal'])}</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(_plan_html(pending["plan"]), unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    if c1.button("Approve and run", type="primary", use_container_width=True):
        goal = pending["goal"]
        with st.status("Running approved plan...", expanded=False) as status:
            t0 = time.time()
            # Run the plan the user actually approved -- not a fresh draft.
            # The gate auto-approves because the approval already happened here.
            run = run_planning_agent(
                goal,
                plan=pending["plan"],
                approve_plan=lambda p: True,
                max_steps=max_steps,
                max_revisions=max_revisions,
                per_step_tool_calls=per_step_calls,
            )
            elapsed = time.time() - t0
            status.update(
                label=f"Done in {elapsed:.1f}s, {len(run.step_results)} steps, "
                      f"{len(run.revisions)} revision(s).",
                state="complete",
            )
        st.session_state["history"].append((goal, run))
        st.session_state["pending_plan"] = None
        st.rerun()
    if c2.button("Cancel", use_container_width=True):
        st.session_state["pending_plan"] = None
        st.rerun()


# ----- Input ---------------------------------------------------------------

typed = st.chat_input("What would you like me to plan?")
goal = typed or st.session_state["queued_goal"]
if st.session_state["queued_goal"]:
    st.session_state["queued_goal"] = ""

if goal and st.session_state["pending_plan"] is None:
    with st.status("Drafting plan...", expanded=False):
        plan = write_plan(goal, max_steps=max_steps)
    st.session_state["pending_plan"] = {"goal": goal, "plan": plan}
    st.rerun()


st.markdown(
    '<p class="disclaimer">Plans are auto-generated and may be off. Approve before any tool runs. Sources can be wrong; verify.</p>',
    unsafe_allow_html=True,
)
