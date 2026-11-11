"""`council` command-line interface: the director's first channel."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path

from axiom_council.audit import AuditTamperError, open_audit
from axiom_council.configstore import KINDS, ConfigStore
from axiom_council.contracts import Budget, Capability, TaskClass, TaskSpec
from axiom_council.council import Council, open_registry
from axiom_council.evals.bakeoff import summarise
from axiom_council.evals.session import parse_contestants, run_bakeoff
from axiom_council.providers import ProviderError, provider_for
from axiom_council.sandbox.runner import DockerSandbox
from axiom_council.sandbox.selftest import selftest


def _home() -> Path:
    return Path(os.environ.get("AXIOM_COUNCIL_HOME", ".council"))


def _config() -> Path:
    return Path(os.environ.get("AXIOM_COUNCIL_CONFIG", "config"))


def _cmd_nodes(_: argparse.Namespace) -> int:
    registry = open_registry(_home(), _config())
    for node in registry.nodes.values():
        if not node.enabled:
            print(f"{node.node_id:<16} off   {node.runtime:<10} {node.hardware}")
            continue
        try:
            provider = provider_for(node, timeout=5)
        except ProviderError as exc:
            print(f"{node.node_id:<16} error {exc}")
            continue
        try:
            live = set(provider.list_models())
            status = "up"
        except ProviderError:
            live, status = set(), "down"
        finally:
            provider.close()
        print(f"{node.node_id:<16} {status:<5} {node.runtime:<10} {node.hardware}")
        for model_id in node.serves:
            mark = "ok" if model_id in live else "missing"
            print(f"    {model_id:<28} {mark}")
    return 0


def _task_from_args(args: argparse.Namespace) -> TaskSpec:
    return TaskSpec(
        workspace_id=args.workspace,
        objective=args.objective,
        task_class=TaskClass(args.task_class),
        actor="director",
        allowed_capabilities=frozenset({Capability.LLM_INFERENCE}),
        budget=Budget(max_tokens=args.max_tokens, max_wall_seconds=600),
        acceptance_checks=("director_review",),
    )


def _cmd_route(args: argparse.Namespace) -> int:
    council = Council(open_registry(_home(), _config()), _home())
    try:
        decision = council.router.route(
            _task_from_args(args),
            prompt_tokens=args.prompt_tokens,
            output_tokens=args.output_tokens,
        )
    finally:
        council.close()
    for c in decision.ranked:
        print(
            f"ELIGIBLE  {c.model_id:<24} @{c.node_id:<14} p={c.p_success:.2f}"
            f" (lb {c.p_lower:.2f}, n={c.trials}) ~{c.expected_seconds:.0f}s"
            f" shadow ${c.shadow_api_usd:.4f}"
        )
    for r in decision.rejected:
        print(f"REJECTED  {r.model_id:<24} @{r.node_id:<14} {r.reason}")
    print(f"CHOSEN    {decision.chosen.model_id if decision.chosen else '— escalate to director'}")
    return 0


def _cmd_ask(args: argparse.Namespace) -> int:
    council = Council(open_registry(_home(), _config()), _home())
    try:
        outcome = council.run(
            _task_from_args(args), args.objective, output_tokens=args.output_tokens
        )
    finally:
        council.close()
    print(f"state: {outcome.state}")
    if outcome.reasons:
        print("reasons:", "; ".join(outcome.reasons))
    if outcome.inbox_item:
        print(f"waiting on director: council inbox show {outcome.inbox_item}")
    if outcome.result:
        r = outcome.result
        print(
            f"model: {r.model_id} @ {r.node_id}  ({r.usage.total_tokens} tok, "
            f"{r.usage.wall_seconds:.1f}s)\n"
        )
        print(r.output)
    return 0 if outcome.result or outcome.inbox_item else 1


def _cmd_inbox(args: argparse.Namespace) -> int:
    council = Council(open_registry(_home(), _config()), _home())
    try:
        inbox = council.inbox
        if args.action == "list":
            for item in inbox.pending():
                print(f"{item.item_id}  {item.kind:<13} {item.task_id}  {item.summary}")
        elif args.action == "show":
            print(inbox.get(args.item_id).model_dump_json(indent=2))
        elif args.action == "approve":
            token = inbox.approve(
                args.item_id, "director", timedelta(hours=args.ttl_hours), args.text or ""
            )
            council.audit.append(
                "approved", token.task_id, "director", token=token.token_id,
                item=args.item_id, artifact=token.artifact_hash,
            )  # fmt: skip
            print(f"approved: {token.token_id} (expires {token.expires_at.isoformat()})")
        elif args.action == "reject":
            inbox.reject(args.item_id, args.text or "")
            council.audit.append("rejected", inbox.get(args.item_id).task_id, "director")
            print("rejected")
        elif args.action == "answer":
            inbox.answer(args.item_id, args.text or "")
            print("answered")
    finally:
        council.close()
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    store = ConfigStore(_home() / "config.sqlite3")
    try:
        if args.action == "seed":
            print(json.dumps(store.seed(_config())))
        else:
            for kind in KINDS:
                for key in store.active(kind):
                    entry = store.get(kind, key)
                    assert entry is not None
                    print(f"{kind:<10} {key:<24} v{entry.version}  {entry.sha256[:12]}")
    finally:
        store.close()
    return 0


def _cmd_sandbox(args: argparse.Namespace) -> int:
    registry = open_registry(_home(), _config())
    profile = registry.sandboxes.get(args.sandbox_id)
    if profile is None:
        print(f"unknown sandbox {args.sandbox_id!r}", file=sys.stderr)
        return 2
    report = selftest(
        profile,
        DockerSandbox(profile),
        _home(),
        open_audit(_home()),
        commit=args.commit,
    )
    print(f"snapshot  {report.snapshot.path} @ {report.snapshot.short}")
    print(f"image     {report.image}")
    print(f"network   {'denied' if report.isolation.passed else 'REACHABLE'}")
    print(f"frozen    {'edit rejected' if report.frozen_edit_rejected else 'EDIT ALLOWED'}")
    for check in report.checks:
        status = "pass" if check.passed else "FAIL"
        print(f"check     {check.name:<10} {status}  {check.duration_seconds:.1f}s")
        if not check.passed:
            print(check.output_tail)
    print(f"selftest  {'PASSED' if report.passed else 'FAILED'}")
    return 0 if report.passed else 1


def _cmd_bakeoff(args: argparse.Namespace) -> int:
    registry = open_registry(_home(), _config())
    records, results = run_bakeoff(
        registry,
        _home(),
        Path(args.fixtures),
        parse_contestants(args.models),
        with_code=not args.skip_code,
        split=args.split,
        repair_attempts=args.repair,
    )
    for row in summarise(records):
        print(json.dumps(row))
    print(f"results: {results}")
    return 0


def _cmd_tasks(_: argparse.Namespace) -> int:
    council = Council(open_registry(_home(), _config()), _home())
    try:
        for row in council.tasks.summary():
            print(
                f"{row['task_id']}  {row['state']:<22} {row['updated_at'][:19]}  {row['objective']}"
            )
    finally:
        council.close()
    return 0


def _cmd_resume(args: argparse.Namespace) -> int:
    council = Council(open_registry(_home(), _config()), _home())
    try:
        outcome = council.resume(args.task_id)
    except PermissionError as exc:
        print(f"refused: {exc}", file=sys.stderr)
        return 1
    finally:
        council.close()
    print(f"state: {outcome.state}")
    if outcome.result:
        print(outcome.result.output)
    return 0


def _cmd_audit(args: argparse.Namespace) -> int:
    log = open_audit(_home())
    if args.action == "adopt":
        if not args.reason:
            print("audit adopt: --reason required", file=sys.stderr)
            return 2
        record = log.adopt("director", args.reason)
        print(json.dumps({"adopted": record["seq"], "records": log.verify()}))
        return 0
    try:
        head = log.head()
    except AuditTamperError as exc:
        print(json.dumps({"chain": "broken", "error": str(exc)}))
        return 1
    print(json.dumps({"records": head.count, "head": head.head, "chain": "intact"}))
    return 0


def _task_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("objective")
    p.add_argument("--workspace", default="sandbox")
    p.add_argument("--task-class", default="triage", choices=[t.value for t in TaskClass])
    p.add_argument("--max-tokens", type=int, default=16_000)
    p.add_argument("--output-tokens", type=int, default=1024)


def _add_task_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Commands that submit, route, list and resume tasks."""
    route = sub.add_parser("route", help="explain where a task would be routed")
    _task_args(route)
    route.add_argument("--prompt-tokens", type=int, default=500)
    route.set_defaults(fn=_cmd_route)

    ask = sub.add_parser("ask", help="run a single routed task")
    _task_args(ask)
    ask.set_defaults(fn=_cmd_ask)

    sub.add_parser("tasks", help="list recent tasks and their states").set_defaults(fn=_cmd_tasks)
    resume = sub.add_parser("resume", help="run an approved task (verifies its approval token)")
    resume.add_argument("task_id")
    resume.set_defaults(fn=_cmd_resume)


def _add_director_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Commands the director uses to decide, configure and audit."""
    inbox = sub.add_parser("inbox", help="director approvals and clarifications")
    inbox.add_argument("action", choices=["list", "show", "approve", "reject", "answer"])
    inbox.add_argument("item_id", nargs="?")
    inbox.add_argument("--text", help="reason (approve/reject) or answer text")
    inbox.add_argument("--ttl-hours", type=float, default=4)
    inbox.set_defaults(fn=_cmd_inbox)

    config = sub.add_parser("config", help="seed or list the versioned config store")
    config.add_argument("action", choices=["seed", "list"])
    config.set_defaults(fn=_cmd_config)

    audit = sub.add_parser("audit", help="verify the audit chain and anchors, or adopt a log")
    audit.add_argument("action", nargs="?", choices=["verify", "adopt"], default="verify")
    audit.add_argument("--reason", help="why the current log is trusted (adopt only)")
    audit.set_defaults(fn=_cmd_audit)


def _add_infra_commands(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Commands that probe nodes, test sandboxes and evaluate models."""
    sub.add_parser("nodes", help="probe configured nodes and models").set_defaults(fn=_cmd_nodes)

    sandbox = sub.add_parser("sandbox", help="self-test a sandbox profile (needs Docker)")
    sandbox.add_argument("action", choices=["selftest"])
    sandbox.add_argument("sandbox_id")
    sandbox.add_argument("--commit", default="HEAD")
    sandbox.set_defaults(fn=_cmd_sandbox)

    bakeoff = sub.add_parser("bakeoff", help="compare models on fixed, checkable fixtures")
    bakeoff.add_argument("--models", required=True, help="model[=think],... e.g. qwen3.8:27b=low")
    bakeoff.add_argument("--fixtures", default="evals/fixtures")
    bakeoff.add_argument("--skip-code", action="store_true", help="skip sandboxed code fixtures")
    bakeoff.add_argument(
        "--split",
        choices=["development", "calibration", "holdout"],
        default="calibration",
        help="calibration updates routing; development and holdout never do",
    )
    bakeoff.add_argument(
        "--repair", type=int, choices=range(4), help="repairs after public-check feedback"
    )
    bakeoff.set_defaults(fn=_cmd_bakeoff)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="council", description="Axiom Council control plane")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_task_commands(sub)
    _add_director_commands(sub)
    _add_infra_commands(sub)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "inbox" and args.action != "list" and not args.item_id:
        print("inbox: item_id required", file=sys.stderr)
        return 2
    try:
        code: int = args.fn(args)
    except AuditTamperError as exc:
        print(f"audit: {exc}", file=sys.stderr)
        return 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
