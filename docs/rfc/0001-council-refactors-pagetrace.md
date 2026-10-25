# RFC 0001: R1 — the council refactors PageTrace

- Status: accepted
- Date: 2026-10-25

## Goal

The council's first real task is engineering work on a sibling repository: a
**behaviour-preserving refactor of PageTrace**, proposed, built, tested and reviewed by
open-weight models, approved by the director, and delivered as a branch for a normal PR.

First target: `src/pagetrace/evaluation.py` (415 lines) → cohesive modules that meet the
council's own standards (≤ 500 lines/file, ≤ 50 lines/function), with **no behaviour change**.
A refactor is the right first task because success is objective: the existing test suite
must still pass, unchanged, and the public API must not move.

## Agents and authority

| Role | Model (initial) | May do | May not do |
|---|---|---|---|
| Director | human | approve plans and patches, answer clarifications | — |
| Dispatcher | gpt-oss:20b | plan, split work, pick workers from the router's eligible set, answer worker questions | edit files, run commands |
| Builder | router-chosen worker | propose file edits inside allowed paths | touch tests, CI, dependencies; run commands |
| Reviewer | different model family from the builder | judge the diff against the plan | edit the diff |
| Gatekeeper | deterministic code | run allowlisted checks in the sandbox, apply policy | be overruled by any model |

Models produce **data** (plans, edits, reviews) validated against schemas. Only deterministic
code acts: applying edits, running tests, creating branches. No model ever gets a shell.

## Message passing

Agents never talk to each other directly. Every message is an **envelope** on a central
**broker**:

```
sender ─▶ broker ─▶ [schema] ─▶ [route policy] ─▶ [guards] ─▶ [audit] ─▶ recipient
                         └────────── reject + audit + escalate ──────────┘
```

- **Envelope:** id, task id, parent id, sender role, recipient role, kind (`plan`, `edit`,
  `review`, `clarify`, `answer`, `verdict`, `evidence`), payload, payload hash, trust label
  (`trusted` = produced by council code or director; `untrusted` = model output or repo
  content), timestamp.
- **Route policy:** an allowlist matrix (config) of which role may send which kind to which
  role — e.g. a builder may send `edit` to the gatekeeper and `clarify` to the dispatcher,
  but nothing to the director directly.
- **Interceptors:** an ordered, pluggable chain run on every envelope: schema validation,
  route policy, size limits, injection heuristics, secret scanning, path policy for edits,
  then the audit append. Any interceptor can reject; the director can attach a tap to watch
  live traffic. This is where interception happens — one choke point, not per-agent hooks.
- **Graph:** parent ids make each task's conversation a DAG, kept in the audit log, so any
  outcome can be traced back through every message that led to it.

**Why not a GNN?** Graph neural networks learn patterns over graph-structured data. Message
passing between agents is a *protocol* problem (who may say what to whom, verified how),
which needs deterministic routing and validation, not a learned model. Once we have
hundreds of traced task DAGs, a graph model could be studied for *routing prediction* — a
research question for later, not infrastructure.

## Pipeline

1. **Snapshot.** Clone PageTrace at a pinned commit into the council's sandbox directory.
   The director's own checkout is never touched.
2. **Context.** Deterministic code builds the context pack: file tree, target file, its
   tests, size report, public API (names exported via AST). Repo content is labelled
   untrusted and fenced with random boundary markers.
3. **Plan.** Dispatcher returns a schema-checked plan: new module layout, which functions
   move where, acceptance checks. Plan goes to the director's inbox (approval token bound to
   the plan hash).
4. **Build.** Builder returns whole-file edits (`create` / `replace`) — easier for small
   models than unified diffs. Path policy: only `src/pagetrace/**`; tests, CI, packaging and
   dependency files are frozen for a refactor.
5. **Gate.** Gatekeeper applies edits in the sandbox and runs the allowlisted checks in a
   network-denied container: PageTrace's tests, ruff, the size check, and an AST diff of the
   public API. Failures go back to the builder (bounded attempts).
6. **Review.** Reviewer (other model family) gets plan + diff + gate results; returns a
   verdict with findings. It cannot pass failing gates.
7. **Present.** Diff, gate evidence, review and cost go to the director's inbox, approval
   bound to the patch hash. A changed patch needs a new approval.
8. **Deliver.** On approval, the council commits to a `refactor/…` branch in the PageTrace
   sandbox clone as the owner. Pushing and opening the PR stay explicit director actions.

Every verified outcome (gate pass/fail, review agreement) is written to the capability
ledger, so the router learns which models are good at `plan`, `code_patch` and `review`.

## Milestones (one PR each)

| PR | Delivers | Gate |
|---|---|---|
| R1.1 | Envelope, broker, route policy, interceptor chain, guards, threat model | Forbidden routes and poisoned payloads rejected and audited |
| R1.2 | Repo snapshot + network-denied Docker sandbox + allowlisted check runner | Edits outside allowed paths refused; container has no network |
| R1.3 | Context pack, plan schema, edit schema, API-diff gate | Plan and edits validated; API change detected |
| R1.4 | End-to-end pipeline with dispatcher, builder, reviewer, director inbox | `evaluation.py` refactor passes PageTrace tests unchanged |
| R1.5 | Grounded-answer workflow over PageTrace evidence (parked branch) | Citations verified, ledger updated |

## Hardware

R1 runs on the Mac alone: gpt-oss:20b as dispatcher plus one 4B worker fit in the ~22 GB
budget, and the work is sequential. The Windows RTX 4060 node arrives in **R2**, where
parallel builders and reviewers on separate machines, node failure and load-aware placement
are the point of the release.
