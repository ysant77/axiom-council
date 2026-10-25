# Roadmap

Each release ships behind its own acceptance gate. Claims are made only from measured runs.

| Release | Scope | Acceptance gate |
|---|---|---|
| **R0** — foundation | Contracts, fail-closed policy, hash-chained audit, versioned config store, capability ledger, cost-aware router, director inbox (CLI), OpenAI-compatible provider, CI with 85% coverage gate | Invalid scopes denied; audit tamper detected; same input routes reproducibly; live call on the Mac node |
| **R1** — the council refactors PageTrace | See [RFC 0001](rfc/0001-council-refactors-pagetrace.md). R1.1 message bus and interceptors; R1.2 network-denied sandbox and check runner; R1.3 context pack, plan/edit schemas, API-diff gate; R1.4 end-to-end refactor of `pagetrace/evaluation.py` with dispatcher, builder, reviewer and director approval; R1.5 grounded answers over PageTrace evidence | PageTrace's unchanged test suite passes on the council's refactor; every message routed, inspected and audited; no model ever executes code |
| **R2** — distributed council | Windows RTX 4060 node over Tailscale; parallel builders/reviewers across machines; node heartbeats and load-aware placement; Telegram as a second director channel | Same task set split across two machines; node failure reroutes without losing audit events |
| **R3** — evaluated council | A/B/C against deterministic and single-agent baselines on 30+ fixed engineering tasks; learned routing from the capability ledger | Council beats single agent on verified success per total effort, or extra roles are switched off |
| **R4** — edge and mobile | Mobile director client (approval inbox, local small model), edge node profile | Offline approval flow works; measured on named devices |
| Later | Content workflows (profile analysis → draft → humanise → critique) as a council template; vLLM/SGLang serving; learned routing | Each justified by benchmark and attack tests |

Atmos stays paused until the R1 pilot works. Data gathering for it can run in parallel
through approved, read-only council tasks.
