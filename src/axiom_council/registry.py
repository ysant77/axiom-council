"""Typed views over the config store: model cards, compute nodes, roles, workspaces, prompts.

A model card holds *claims* (context length, modalities, licence, memory). Whether a model
is actually good at a task class is measured separately by the capability ledger.
"""

from __future__ import annotations

from string import Template
from typing import Any

from pydantic import Field

from axiom_council.bus.interceptors import Route
from axiom_council.configstore import ConfigStore
from axiom_council.contracts import Strict, TaskClass
from axiom_council.policy import RolePolicy, WorkspacePolicy
from axiom_council.sandbox.profile import SandboxProfile
from axiom_council.settings import Settings


class ModelCard(Strict):
    model_id: str
    family: str
    licence: str
    params_b: float = Field(gt=0)
    active_params_b: float | None = None
    context_tokens: int = Field(gt=0)
    modalities: frozenset[str] = frozenset({"text"})
    tool_calling: bool = False
    memory_gb: float = Field(gt=0, description="resident memory at the served quantisation")
    prior_skill: dict[TaskClass, float] = Field(
        default_factory=dict,
        description="prior success probability per task class before any measurement",
    )
    shadow_usd_per_mtok: float = Field(
        default=0.0, ge=0, description="price of a comparable hosted API, for savings reports"
    )
    reasoning_overhead_tokens: int = Field(
        default=0, ge=0, description="hidden reasoning tokens to budget on top of the answer"
    )
    request_defaults: dict[str, Any] = Field(
        default_factory=dict, description="extra request fields, e.g. reasoning_effort"
    )
    think: bool | str | None = Field(
        default=None,
        description="native reasoning control (false, true, low, medium, high) where the engine"
        " supports it; None leaves the engine default",
    )


class NodeSpec(Strict):
    node_id: str
    base_url: str
    runtime: str
    hardware: str
    memory_budget_gb: float = Field(gt=0)
    watts_under_load: float = Field(gt=0)
    serves: tuple[str, ...]
    usd_per_kwh: float = Field(default=0.30, ge=0)
    trusted_for_private: bool = False
    enabled: bool = True
    energy_meter: str = Field(
        default="estimate", description="macmon, battery, nvidia-smi or estimate"
    )


class PromptTemplate(Strict):
    """A versioned prompt. `user` uses $placeholders; missing values raise, never blank out."""

    prompt_id: str
    role: str
    task_class: TaskClass | None = None
    system: str
    user: str = "$input"

    def render(self, **values: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": Template(self.user).substitute(values)},
        ]


class Registry(Strict):
    models: dict[str, ModelCard]
    nodes: dict[str, NodeSpec]
    roles: dict[str, RolePolicy]
    workspaces: dict[str, WorkspacePolicy]
    prompts: dict[str, PromptTemplate] = Field(default_factory=dict)
    routes: dict[str, Route] = Field(default_factory=dict)
    sandboxes: dict[str, SandboxProfile] = Field(default_factory=dict)
    settings: Settings = Field(default_factory=Settings)

    @classmethod
    def from_store(cls, store: ConfigStore) -> Registry:
        registry = cls(
            models={k: ModelCard(**v) for k, v in store.active("model").items()},
            nodes={k: NodeSpec(**v) for k, v in store.active("node").items()},
            roles={k: RolePolicy(**v) for k, v in store.active("role").items()},
            workspaces={k: WorkspacePolicy(**v) for k, v in store.active("workspace").items()},
            prompts={k: PromptTemplate(**v) for k, v in store.active("prompt").items()},
            routes={k: Route(**v) for k, v in store.active("route").items()},
            sandboxes={k: SandboxProfile(**v) for k, v in store.active("sandbox").items()},
            settings=Settings.from_entries(store.active("setting")),
        )
        registry.validate_links()
        return registry

    def validate_links(self) -> None:
        for node in self.nodes.values():
            unknown = set(node.serves) - self.models.keys()
            if unknown:
                raise ValueError(f"node {node.node_id} serves unknown models {sorted(unknown)}")
        for prompt in self.prompts.values():
            if prompt.role not in self.roles:
                raise ValueError(f"prompt {prompt.prompt_id} targets unknown role {prompt.role}")
        parties = self.roles.keys() | {"director"}
        for route in self.routes.values():
            unknown = {route.sender, route.recipient} - parties
            if unknown:
                raise ValueError(f"route {route.route_id} names unknown roles {sorted(unknown)}")
        for kind in self.settings.bakeoff.kinds.values():
            if kind.prompt_id not in self.prompts:
                raise ValueError(f"bake-off kind names unknown prompt {kind.prompt_id}")
        for profile in self.sandboxes.values():
            if profile.workspace_id not in self.workspaces:
                raise ValueError(f"sandbox {profile.sandbox_id} names unknown workspace")

    def prompt_for(self, role: str, task_class: TaskClass) -> PromptTemplate | None:
        """Most specific prompt for a role: task-class match first, then the role default."""
        matches = [p for p in self.prompts.values() if p.role == role]
        exact = [p for p in matches if p.task_class is task_class]
        generic = [p for p in matches if p.task_class is None]
        return next(iter(exact or generic), None)
