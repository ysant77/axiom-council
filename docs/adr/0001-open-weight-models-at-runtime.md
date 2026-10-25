# ADR 0001: Open-weight models only at runtime

- Status: accepted
- Date: 2026-10-25

## Context

The council exists to show that most sub-tasks can be handled by smaller open-weight models
at a fraction of the cost of a large hosted model, with measured quality. Hosted proprietary
models at runtime would hide that cost and make results irreproducible.

## Decision

At runtime the council calls only open-weight models served on machines we control, through
the OpenAI-compatible API (Ollama and llama.cpp for development; vLLM and SGLang for
serving). Proprietary coding assistants may help developers build and review the code, but
take no part in task execution, routing or arbitration.

## Consequences

- Every model choice is reproducible from a pinned model ID and config version.
- Savings are reported against a configured hosted-API shadow price, recalibrated before
  publication.
- Capability ceilings are those of current open-weight models; the roadmap adds a stronger
  open model or rented GPU only when a benchmark shows the need.
