# Architecture

## Principles

1. **Rules decide what is allowed.** `policy.py` checks every task and every action against
   workspace ceilings and role grants. It fails closed and no model output can widen it.
2. **Evidence decides who is capable.** `ledger.py` stores outcomes verified by someone other
   than the model (a deterministic check or the director). Model-card priors are a starting
   belief worth ten pseudo-observations, so real results take over quickly.
3. **Cost decides among the capable.** `router.py` ranks eligible placements by
   memory-seconds per verified success. On a shared machine, memory held over time is the
   scarce resource, so a 14 GB model has to be proportionally better to win a task that a
   3 GB model can do.
4. **Humans decide what matters.** High-risk tasks, missing placements and clarifying
   questions go to the director's inbox. Approval tokens are bound to task, action and
   artifact hash and expire.
5. **Everything is auditable.** `audit.py` is an append-only, hash-chained log; config used
   by any run is recoverable by version and hash from `configstore.py`.

## Run loop (R0)

```
director ──TaskSpec──▶ policy.check_task ──deny──▶ audit + reasons
                              │allow
                              ▼
                   policy.check_action(role, llm_inference)
                              │
                              ▼
               prompt = registry.prompt_for(role, task_class)
                              │
                              ▼
     router.route ── hard checks (modality, context, memory, data trust, budget)
                  ── skill gate (posterior lower bound ≥ min_success)
                  ── rank by GB·s per verified success
                              │
          none eligible ──────┼──── needs approval ──▶ director inbox
                              ▼
          provider.chat (OpenAI-compatible: Ollama / llama.cpp / vLLM / SGLang)
                              │
                              ▼
                     audit: inference_completed
```

## Authority chain

`worker → dispatcher → director`. Workers answer `CLARIFY:` instead of guessing; the
dispatcher (gpt-oss:20b) answers what it can and escalates the rest. The dispatcher may only
choose among placements the router marked eligible, and must justify the choice.

## State

| Store | File (under `AXIOM_COUNCIL_HOME`) | Purpose |
|---|---|---|
| Config store | `config.sqlite3` | Versioned models, nodes, roles, workspaces, prompts |
| Capability ledger | `ledger.sqlite3` | Verified outcomes per model and task class |
| Audit log | `audit.jsonl` | Hash-chained event history |
| Inbox | `inbox/*.json` | Approvals and clarifications for the director |

Config and runtime state are separate so either can move to Postgres independently.

## Extending

- **New model:** add a card to the config store (or `config/models.yaml` and reseed) and list
  it under a node's `serves`. Set `reasoning_overhead_tokens` and `request_defaults` for
  thinking models.
- **New machine:** add a node with its OpenAI-compatible URL and memory budget.
- **New role or workspace:** add it to the store; policy picks it up without code changes.
- **New runtime:** implement the same `chat` / `list_models` surface as
  `OpenAICompatProvider` and inject it through `Council(provider_factory=...)`.
