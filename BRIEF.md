# Take-home: the planning agent

**Role:** Forward Deployed Engineer
**Time:** aim for about three hours. Please don't spend more than four. We
would rather read a focused submission with honest gaps than a padded one.
**Deadline:** 7 days from the day you receive this.

---

## The situation

A travel operations team runs a research desk. Someone asks "plan a two-day
offsite in Lisbon for eight people", and a coordinator spends forty minutes
searching, reading pages, and writing it up.

We built them a prototype agent in a two-day workshop. It drafts a short plan,
shows the plan to the coordinator, waits for approval, then executes the plan
step by step: searching, reading, and rewriting the remaining steps when
something unexpected turns up.

The prototype demos well, and parts of it were never finished. **You are the
FDE on this account.** Finish it, and tell us how you would know it works.

## What you have

```
src/planner.py       writes and revises plans        <- one thing to implement
src/agent.py         executes an approved plan       <- one thing to implement
src/itinerary_app.py the Streamlit surface           <- one thing to implement
src/tools.py         web search, page fetch, image   <- provided
src/llm_client.py    model access + replay mode      <- provided
src/run_once.py      terminal runner                 <- provided
src/tests/           contract tests, offline         <- your feedback loop
fixtures/            recorded model and tool traffic <- provided
```

**Everything runs offline.** `LLM_MODE=replay` and `TOOLS_MODE=replay` are the
defaults: model calls are served from `fixtures/llm.json`, web calls from
`fixtures/tools.json`. No API key, no network, no cost, and the same input
produces the same trace every time. If you have your own key you can set
`LLM_MODE=live`, but nothing here requires it and we grade in replay mode.

```bash
make setup && make test      # offline, a second or two
make demo                    # one goal, end to end, in the terminal
make app                     # the Streamlit UI
```

`make test` starts red. Twelve of the twenty-four tests fail until you write
the three things below; the other twelve cover code we shipped and should be
green from the first run.

---

## Part A: make it work · 70 pts

Three things are unimplemented, marked `TODO A1`-`TODO A3`. The docstrings
state the contract each one has to satisfy.

| | where | what | pts |
|---|---|---|---:|
| A1 | `planner.revise_plan` | rewrite the steps that have not run yet | 21 |
| A2 | `agent.run_planning_agent` | the plan → approve → act → observe → replan loop | 30 |
| A3 | `itinerary_app` render block | show the step trace and any mid-run replan | 19 |

Three things are already written, and you should read them before you start:

- **`planner.write_plan`**: the same shape of problem as A1, and it shows the
  house style. Read it first.
- **`agent.execute_step`**: runs one step. A2 calls it; you don't write it.
- **`agent._RunRecorder`**: the run's bookkeeping, the audit-trail keys, the
  image and source merging, the progress-callback guard, the final-answer
  fallback. **A2 is a control-flow problem. The recorder owns the result
  shapes, you own the order things happen in.** There is a worked example of
  it at the bottom of the A2 tests.

The contract docstrings are the specification. The visible tests are a floor,
not a ceiling: we grade against a larger set of the same contracts.

## Part B: make the case · 30 pts

Fill in `DECISIONS.md`. **Two questions, in prose, three to five sentences
each.** This is not a formality. It is worth nearly a third of the grade, and
we read it before we read your diff.

1. **What did you deliberately not do, and why?** (12): what you found,
   judged, and left, and what would change your mind. This includes anything
   you noticed in the code we shipped and decided not to touch.
2. **How would you know this works?** (18): concretely, on a system that is
   non-deterministic by construction: what do you measure, how do you get a
   repeatable read on it, and what number is bad?

Plus two one-liners: how you used AI assistance, and roughly how long you spent.

---

## Ground rules

**AI tools are allowed and expected.** Use whatever you use at work. Note in
`DECISIONS.md` roughly how you used them and where you did or did not trust the
output. We are not testing whether you can type Python from memory. We are
testing judgment, and judgment is what survives the tooling.

**We will read your code with you.** Shortlisted candidates get a 45-minute
session: walk us through what you built, then extend it live with us. Submit
work you can defend.

**Ask questions.** If something is genuinely ambiguous, email
usman.asif@azurepartners.ai. That is the job. If you would rather make a call
and move on, do that and write it down. Both are fine; silently guessing and
not saying so is not.

## Submitting

Email usman.asif@azurepartners.ai a zip or a link to a private repo containing:

- your working tree, `.env` excluded
- `DECISIONS.md`, filled in
- anything you added: tests, scripts, notes

Make sure `make setup && make test && make demo` works from a clean checkout on
a machine with no API key. That is the first thing we run.
