# Threat model

Scope: the council control plane, its agents, the message broker, adapters and the sandbox
that applies model-written code. Reviewed at every release.

## Frameworks we map to

- **OWASP Top 10 for LLM Applications (2025)** — LLM01–LLM10.
- **OWASP Agentic Security Initiative — Agentic AI Threats and Mitigations** — agent-specific
  threats such as memory poisoning, tool misuse, privilege compromise, cascading
  hallucinations, inter-agent communication poisoning and rogue agents.
- **NIST AI RMF 1.0** and its **Generative AI Profile (NIST AI 600-1)** — govern, map,
  measure, manage.
- **MITRE ATLAS** — adversary tactics against ML systems, used to write attack fixtures.
- **NIST SSDF (SP 800-218)** — secure development of the council itself; supply-chain checks.

## Trust boundaries

1. Director ↔ council (CLI now; Telegram, mobile later).
2. Council code ↔ model output (every model reply is untrusted data).
3. Council ↔ repository and document content (untrusted: READMEs, comments, PDFs).
4. Council ↔ sandbox (model-written code executes only here).
5. Council ↔ other machines (inference nodes over the LAN/Tailscale).

## Threats and controls

| Threat | Framework | Control in the council | Status |
|---|---|---|---|
| Direct or indirect prompt injection (instructions hidden in repo files, PDFs, tool output) | LLM01, ATLAS | Untrusted content fenced with random boundary markers; models never trigger actions — they emit schema-checked data; injection heuristics interceptor flags and audits; trust label on every envelope | R1.1 |
| Inter-agent communication poisoning (one agent manipulating another) | ASI | All traffic through the broker; route allowlist; payload schemas per kind; untrusted label propagates to anything derived from untrusted input | R1.1 |
| Excessive agency / privilege escalation | LLM06, ASI | Role and workspace ceilings in code; no model gets a shell or credentials; deterministic gatekeeper performs every action | R0 ✔ |
| Improper output handling (model output executed or trusted) | LLM05 | Edits applied only in a network-denied sandbox; path policy; tests and CI files frozen for refactors | R1.2 |
| Data and model poisoning (bad model weights, poisoned ledger) | LLM04, ASI | Pinned model IDs and hashes in the config store; ledger only accepts outcomes verified by someone other than the model | R0 ✔ / R2 hashes |
| Supply chain (dependencies, actions, model downloads) | LLM03, SSDF | `uv --locked`, pip-audit, gitleaks, pinned action majors; model pulls from known registries only | R0 ✔ |
| Sensitive information disclosure (secrets in prompts, logs, patches) | LLM02 | Secret-scanning interceptor on every envelope; no secrets in config; audit stores hashes and sizes, not raw private content | R1.1 |
| System prompt leakage | LLM07 | Canary token per prompt version; interceptor rejects outputs that contain it | R1.1 |
| Cascading hallucination (wrong plan accepted downstream) | LLM09, ASI | Deterministic gates outrank models; reviewer from a different model family; director approval bound to plan and patch hashes | R1.4 |
| Unbounded consumption (loops, token blow-ups) | LLM10 | Task budgets, bounded attempts, envelope size limits, per-task wall-clock limit | R0 ✔ / R1.1 |
| Approval replay or swap (approve plan A, run patch B) | — | Approval tokens bound to task, action, artifact hash and expiry | R0 ✔ |
| Audit tampering | — | Hash-chained, append-only audit log with verification | R0 ✔ |
| Compromised inference node on the LAN | ASI | Private network (Tailscale), per-node trust flag for private data, response schema validation | R2 |

## Testing

Each control gets adversarial fixtures (MITRE ATLAS-style): poisoned README instructions,
a PDF asking the model to exfiltrate secrets, an edit outside allowed paths, a forged
route, an oversized payload, a leaked canary. They run in CI like any other test.
