# Roadmap

Each release ships behind its own acceptance gate. Claims are made only from measured runs.

| Release | Scope | Acceptance gate |
|---|---|---|
| **R0** — foundation | Contracts, fail-closed policy, hash-chained audit, versioned config store, capability ledger, cost-aware router, director inbox (CLI), OpenAI-compatible provider, CI with 85% coverage gate | Invalid scopes denied; audit tamper detected; same input routes reproducibly; live call on the Mac node |
| **R1** — PageTrace ↔ Lumen | Read-only PageTrace adapter; grounded Q&A over a paper chosen in Lumen; deterministic citation verifier feeding the ledger; dispatcher (gpt-oss:20b) choosing among eligible workers; 30+ fixture questions from the AI papers corpus | Every claim maps to a source span or is marked unsupported; routing beats always-largest-model on GB·s per verified answer with no quality loss |
| **R2** — distributed council | Windows RTX 4060 node over Tailscale; node heartbeats and load-aware placement; Telegram as a second director channel | Same task set split across two machines; node failure reroutes without losing audit events |
| **R3** — plan · build · review | Network-denied sandbox builder, reviewer role, evidence-based arbitration; A/B/C against deterministic and single-agent baselines | Council beats single agent on verified success per total effort, or extra roles are switched off |
| **R4** — edge and mobile | Mobile director client (approval inbox, local small model), edge node profile | Offline approval flow works; measured on named devices |
| Later | Content workflows (profile analysis → draft → humanise → critique) as a council template; vLLM/SGLang serving; learned routing | Each justified by benchmark and attack tests |

Atmos stays paused until the R1 pilot works. Data gathering for it can run in parallel
through approved, read-only council tasks.
