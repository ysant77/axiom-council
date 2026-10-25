# Axiom Council

A policy-gated, evaluated council of **open-weight models** that routes each task to the
smallest model that is measurably good enough, on whichever machine can run it, and asks a
human before anything risky.

> Status: **R0 — control-plane foundation.** Contracts, policy, audit, config store,
> capability ledger, cost-aware router, director inbox and an OpenAI-compatible provider.
> Multi-role planning, review and arbitration land in later releases (see
> [docs/ROADMAP.md](docs/ROADMAP.md)).

## Why

Large hosted models are expensive to call thousands of times. Most sub-tasks do not need
them. Axiom Council treats "which model should do this?" as a measured engineering decision:

1. **Rules decide what is allowed.** Workspace and role ceilings live in code and config;
   no model can grant itself a capability.
2. **Evidence decides who is capable.** A capability ledger records verified outcomes per
   model and task class. Priors on model cards are only a starting belief.
3. **Cost decides among the capable.** Candidates are ranked by expected time and energy
   per *verified* success, so failures count against a model.
4. **Humans decide what matters.** Risky actions and unresolved questions go to the
   director's inbox with a hash-bound approval token.

All runtimes speak the OpenAI-compatible API: Ollama and llama.cpp for development, vLLM and
SGLang for serving.

## Quick start

```bash
uv sync
ollama pull qwen3:4b && ollama pull gemma3:4b && ollama pull gpt-oss:20b
uv run council config seed      # load config/*.yaml into the versioned config store
uv run council nodes            # which machines and models are up
uv run council route "Plan the PageTrace adapter" --task-class plan
uv run council ask "Summarise the transformer architecture in three bullets"
uv run council inbox list       # approvals and clarifications waiting on you
uv run council audit            # verify the tamper-evident audit chain
uv run council sandbox selftest pagetrace   # needs Docker: snapshot, isolate, baseline checks
```

Runtime state lives in `.council/` (override with `AXIOM_COUNCIL_HOME`).

## Layout

| Module | Responsibility |
|---|---|
| `contracts.py` | Versioned, strict Pydantic contracts (TaskSpec, ApprovalToken, ...) |
| `policy.py` | Deterministic, fail-closed workspace and role policy |
| `configstore.py` | Versioned SQLite store for models, nodes, roles, workspaces, prompts |
| `registry.py` | Typed views over the config store; prompt templates |
| `ledger.py` | Verified outcomes per model and task class |
| `router.py` | Eligibility, skill gate and cost ranking over (model, node) placements |
| `inbox.py` | Director approvals and clarifications |
| `audit.py` | Append-only, hash-chained audit log |
| `providers/` | OpenAI-compatible inference client |
| `bus/` | Envelopes, broker, route policy and interceptors (injection, secrets, canaries) |
| `sandbox/` | Pinned repo snapshots, path-policed edits, network-denied Docker checks |
| `council.py` | The run loop tying it together |
| `cli.py` | The `council` command |

## Development

```bash
git config core.hooksPath .githooks   # once per clone: blocks pushes to main, bad branch names
uv run ruff check . && uv run ruff format --check . && uv run mypy && uv run pytest
```

All changes reach `main` through a `<type>/<kebab-case>` branch and a pull request with green CI.

CI enforces the same checks and a minimum of 85% coverage. See
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) and
[docs/DISTRIBUTED.md](docs/DISTRIBUTED.md).
