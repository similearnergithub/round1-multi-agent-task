# Multi-Agent AI Brief Generator

This project is a compact **LangChain + LangGraph** multi-agent system. Give it a
question or topic and it returns a practical brief after several agents share
typed results and pass through a quality gate.

## Why Gemini Flash

I chose Google's Gemini Flash-class model (`gemini-3.8-flash`) because it is fast,
cost-effective for an interview-sized workflow, and supports structured JSON
outputs reliably. The model can be changed with `GEMINI_MODEL` without changing
the graph.

## Architecture

```text
START
  -> Researcher -> Analyst -> Critic -> Finalizer -> END
```

- **Researcher** produces factual notes, assumptions, and claims to verify.
- **Analyst** consumes those notes and creates a summary, recommendations, and risks.
- **Critic** consumes both outputs and validates usefulness and internal consistency.
- **Finalizer** combines the shared state into the final Markdown response.
- **Tool:** `calculate_reading_time` is used by the finalizer to report the output size.

The `BriefState` typed dictionary is the communication channel. Pydantic schemas
validate every agent's structured output. If an agent or the model call fails,
the workflow stops safely and prints an explicit error instead of returning a
misleading success.

## Setup

Requires Python 3.10+.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# edit .env and set GOOGLE_API_KEY
```

Get a Gemini key from [Google AI Studio](https://aistudio.google.com/apikey).

## Run

```powershell
python app.py "Should a small company adopt an AI coding assistant?"
```

You can also run `python app.py` without arguments and enter the topic when
prompted.

The program prints the final answer. Intermediate agent outputs are represented
in the shared graph state and are validated before the finalizer runs.
Progress for each agent is printed to stderr, and each model call has a
45-second timeout so a network or quota problem produces an error instead of
leaving the terminal apparently stuck.

## Design and reliability notes

- Low temperature favors repeatable, grounded recommendations.
- Structured output schemas reject malformed agent responses.
- The critic is deliberately separate from the analyst so the workflow has an
  independent quality check rather than blindly formatting one model response.
- No web search tool is included by default, so the system does not pretend that
  model-generated claims are live citations. `sources_to_verify` makes that
  limitation visible and is the extension point for a search API.
