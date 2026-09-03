# Planning agent

A goal in, a plan out, executed step by step with a human approval gate in the
middle.

```
PLAN     draft an ordered plan for the user's goal
APPROVE  show it to a human: nothing runs until they say yes
EXECUTE  run one step at a time, each seeing what the earlier steps found
OBSERVE  reduce each step to one line: ok / thin / surprise
REPLAN   a surprise may rewrite the steps that have not run yet
```

**If you are a candidate: read [BRIEF.md](BRIEF.md) first.** This file is just
how to run things.

## Quick start

```bash
make setup     # venv + dependencies
make test      # offline contract tests
make demo      # one goal, end to end, printed as a trace
make app       # the Streamlit UI at localhost:8501
```

No API key is required. Nothing here touches the network by default.

## Runtime modes

Both default to `replay`, which is what makes the repo reproducible.

| variable | values | meaning |
|---|---|---|
| `LLM_MODE` | `replay` \| `live` \| `record` | where model responses come from |
| `TOOLS_MODE` | `replay` \| `live` | where search and page fetches come from |

- **replay**: served from `fixtures/`. Offline, free, deterministic: the same
  goal produces the same trace on every machine.
- **live**: real calls. Needs `LLM_API_KEY` and any OpenAI-compatible
  `LLM_BASE_URL`. Copy `.env.example` to `.env` first.
- **record**: real calls, and each response is appended to `fixtures/llm.json`
  so it can be replayed later.

Replay looks up an exact hash of the request first; on a miss it falls back to
a shape-correct canned response chosen by intent, so a run never breaks just
because you phrased a prompt differently.

One caveat: `generate_image` returns a URL that a remote service renders on
demand. The tool itself makes no request, but the UI will fetch that URL when
it displays the image.

## Layout

```
BRIEF.md              the assignment
DECISIONS.md          what you decided and why: you fill this in
fixtures/
  llm.json            recorded model traffic + intent fallbacks
  tools.json          recorded search results and page text
src/
  planner.py          writes plans, revises them mid-run
  agent.py            executes an approved plan
  tools.py            web_search, fetch_url, generate_image
  llm_client.py       model access, replay/live/record
  itinerary_app.py    Streamlit UI
  run_once.py         terminal runner
  tests/              offline contract tests
```
