"""Round 5 full-system composition for Kraken-R.

Round 5 is deliberately a composition boundary.  It does not introduce a
worker, scheduler, provider, store, executor, settlement implementation, or
adaptive reducer.  It validates the seams owned by the earlier rounds, seals a
prediction before execution, and delegates grounded credit to the existing
constitutional cycle and adaptive substrate.

The public entry point is :func:`run_round_five`.  Callers provide the already
validated cognition trace and the existing bounded execution adapter; this
module owns only the immutable cross-layer trace and deterministic decisions
about information sufficiency and contribution authority.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Iterable, Mapping

from .adaptive_substrate import AdaptiveAudit, AdaptiveState, apply_grounded_adaptation
from .contracts import Authority, Objective, TaskState
from .cognition_kernel import (
    EpistemicClaim,
    ProcessingTrace,
    replay_processing_trace,
)
from .cycle import ConstitutionalCycle, CycleTrace
from .grounded_execution import (
    GroundedExecutionExecutor,
    GroundedExecutionRequest,
    GroundedExecutionVerifier,
    TrustedExecutorIdentity,
    VerifiedGroundedExecution,
    replay_grounded_execution,
)
from .outcome_learning import (
    ContributionKind,
    ContributionRecord,
    GroundedLearningEpisode,
    PredictionCommitment,
    apply_outcome_learning,
    bind_grounded_outcome,
    commit_prediction,
    evaluate_grounded_outcome,
    make_contribution,
    replay_outcome_learning,
    seal_prediction_request,
)
from .plastic_routing import RouteTopology, SettlementRouteRecord, select_candidate_route


MAX_HUMAN_INPUTS = 8
MAX_KNOWN_FACTS = 16
MAX_UNRESOLVED_PARTS = 16
MAX_ASSUMPTIONS = 8
MAX_ATLAS_ENTRIES = 32
MAX_TRACE_BYTES = 250_000


class RoundFiveError(ValueError):
    """Raised when a full-system composition crosses a hard boundary."""


class HumanInputRole(str, Enum):
    """Roles a human statement may play; none is execution truth."""

    PREFERENCE = "preference"
    OBJECTIVE = "objective"
    CONSTRAINT = "constraint"
    FACTUAL_ASSERTION = "factual_assertion"
    OBSERVATION = "observation"
    CORRECTION = "correction"
    AUTHORIZATION = "authorization"


class HumanAuthority(str, Enum):
    """Authority carried by a human statement."""

    STATED = "user_stated"
    CORRECTION = "user_correction"
    CONTINUE = "user_authorized_continue"
    NONE = "none"


class SufficiencyDecision(str, Enum):
    """The bounded next-step decision for an information state."""

    CLARIFICATION = "clarification"
    RETRIEVAL = "qualified_retrieval"
    CALCULATION = "deterministic_calculation"
    ASSUMPTION = "explicit_assumption"
    CONTINUE_WITH_UNCERTAINTY = "continue_with_uncertainty"
    BLOCKED = "blocked_non_identifiable"
    SUFFICIENT = "sufficient"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            _jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _route_dict(route: Any) -> dict[str, Any]:
    return {
        "route_id": route.route_id,
        "context_id": route.context_id,
        "source": route.source,
        "target": route.target,
        "weight": route.weight,
        "success_count": route.success_count,
        "failure_count": route.failure_count,
        "settlement_ids": list(route.settlement_ids),
        "evidence_ids": list(route.evidence_ids),
        "lineage_ids": list(route.lineage_ids),
    }


def _topology_dict(topology: RouteTopology) -> dict[str, Any]:
    return {
        "topology_id": topology.topology_id,
        "version": topology.version,
        "generation": topology.generation,
        "routes": [_route_dict(item) for item in topology.routes],
        "applied_settlement_ids": list(topology.applied_settlement_ids),
    }


def _selection_dict(selection: Any) -> dict[str, Any]:
    return {
        "topology_id": selection.topology_id,
        "topology_version": selection.topology_version,
        "topology_generation": selection.topology_generation,
        "context_id": selection.context_id,
        "route_id": selection.route_id,
        "transaction_id": selection.transaction_id,
        "objective_id": selection.objective_id,
        "task_state_id": selection.task_state_id,
        "task_state_version": selection.task_state_version,
        "candidate_scores": [list(item) for item in selection.candidate_scores],
    }


def _identifier(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(character.isspace() for character in value)
    ):
        raise RoundFiveError(f"{name} must be a non-empty identifier without whitespace")
    return value


def _texts(values: Iterable[str], name: str, maximum: int) -> tuple[str, ...]:
    try:
        result = tuple(values)
    except TypeError as exc:
        raise RoundFiveError(f"{name} must be iterable") from exc
    if len(result) > maximum or any(not isinstance(item, str) or not item.strip() for item in result):
        raise RoundFiveError(f"{name} is invalid or exceeds its bound")
    return result


@dataclass(frozen=True)
class HumanInput:
    """A typed human input that remains candidate context, never evidence."""

    input_id: str
    role: HumanInputRole
    value: Mapping[str, Any]
    authority: HumanAuthority = HumanAuthority.STATED
    relevant: bool = True

    def __post_init__(self) -> None:
        _identifier(self.input_id, "human input_id")
        object.__setattr__(self, "role", HumanInputRole(self.role))
        object.__setattr__(self, "authority", HumanAuthority(self.authority))
        if not isinstance(self.value, Mapping) or not self.value:
            raise RoundFiveError("human input value must be a non-empty mapping")
        if not isinstance(self.relevant, bool):
            raise RoundFiveError("human input relevance must be boolean")
        if self.role is HumanInputRole.AUTHORIZATION and self.authority is not HumanAuthority.CONTINUE:
            raise RoundFiveError("authorization input requires continue authority")
        if self.role is HumanInputRole.CORRECTION and self.authority is not HumanAuthority.CORRECTION:
            raise RoundFiveError("correction input requires correction authority")
        object.__setattr__(self, "value", _freeze(dict(self.value)))

    @property
    def input_hash(self) -> str:
        return _digest(self.to_dict(include_hash=False))

    def to_contribution(self) -> ContributionRecord:
        return make_contribution(
            f"{self.input_id}-contribution",
            ContributionKind.HUMAN_INPUT,
            self.value,
            source_ids=(self.input_id,),
            relevant=self.relevant,
            correction=self.role is HumanInputRole.CORRECTION,
            provenance={
                "role": self.role.value,
                "authority": self.authority.value,
                "candidate_only": True,
            },
        )

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "input_id": self.input_id,
            "role": self.role.value,
            "value": _jsonable(self.value),
            "authority": self.authority.value,
            "relevant": self.relevant,
        }
        if include_hash:
            value["input_hash"] = self.input_hash
        return value


@dataclass(frozen=True)
class InformationSufficiency:
    """A deterministic, bounded decision that preserves known uncertainty."""

    decision: SufficiencyDecision
    known_facts: tuple[str, ...] = ()
    unresolved_parts: tuple[str, ...] = ()
    explicit_assumptions: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    continue_authorized: bool = False
    decision_hash: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "decision", SufficiencyDecision(self.decision))
        object.__setattr__(self, "known_facts", _texts(self.known_facts, "known_facts", MAX_KNOWN_FACTS))
        object.__setattr__(
            self,
            "unresolved_parts",
            _texts(self.unresolved_parts, "unresolved_parts", MAX_UNRESOLVED_PARTS),
        )
        object.__setattr__(
            self,
            "explicit_assumptions",
            _texts(self.explicit_assumptions, "explicit_assumptions", MAX_ASSUMPTIONS),
        )
        object.__setattr__(self, "reasons", _texts(self.reasons, "sufficiency reasons", 8))
        if not isinstance(self.continue_authorized, bool):
            raise RoundFiveError("continue_authorized must be boolean")
        if self.decision is SufficiencyDecision.SUFFICIENT and (
            self.unresolved_parts
            or self.explicit_assumptions
            or self.continue_authorized
        ):
            raise RoundFiveError(
                "sufficient information cannot retain uncertainty or assumptions"
            )
        if self.decision is SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY and (
            not self.unresolved_parts or not self.continue_authorized
        ):
            raise RoundFiveError(
                "continued uncertainty requires unresolved scope and authorization"
            )
        if (
            self.decision is SufficiencyDecision.ASSUMPTION
            and not self.explicit_assumptions
        ):
            raise RoundFiveError("assumption decision requires explicit assumptions")
        expected = _digest(self.to_dict(include_hash=False))
        if self.decision_hash:
            if self.decision_hash != expected:
                raise RoundFiveError("information sufficiency hash is invalid")
        else:
            object.__setattr__(self, "decision_hash", expected)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "decision": self.decision.value,
            "known_facts": list(self.known_facts),
            "unresolved_parts": list(self.unresolved_parts),
            "explicit_assumptions": list(self.explicit_assumptions),
            "reasons": list(self.reasons),
            "continue_authorized": self.continue_authorized,
        }
        if include_hash:
            value["decision_hash"] = self.decision_hash
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InformationSufficiency":
        expected = {
            "decision",
            "known_facts",
            "unresolved_parts",
            "explicit_assumptions",
            "reasons",
            "continue_authorized",
            "decision_hash",
        }
        if not isinstance(value, Mapping) or set(value) != expected:
            raise RoundFiveError("information sufficiency schema is invalid")
        return cls(
            value["decision"],
            tuple(value["known_facts"]),
            tuple(value["unresolved_parts"]),
            tuple(value["explicit_assumptions"]),
            tuple(value["reasons"]),
            value["continue_authorized"],
            value["decision_hash"],
        )


def assess_information_sufficiency(
    claims: Iterable[EpistemicClaim] = (),
    *,
    known_facts: Iterable[str] = (),
    unresolved_parts: Iterable[str] = (),
    explicit_assumptions: Iterable[str] = (),
    clarification_needed: bool = False,
    qualified_retrieval_needed: bool = False,
    deterministic_calculation_needed: bool = False,
    non_identifiable: bool = False,
    continue_authorized: bool = False,
) -> InformationSufficiency:
    """Choose the maximum defensible next step without collapsing uncertainty.

    Precedence is intentionally conservative: a blocked/non-identifiable state
    dominates all action requests, then clarification, qualified retrieval,
    calculation, and explicit assumptions.  Continuation is permitted only
    when a caller explicitly supplies a continue authorization.
    """

    claim_items = tuple(claims)
    if not all(isinstance(item, EpistemicClaim) for item in claim_items):
        raise RoundFiveError("sufficiency claims must be EpistemicClaim records")
    known = _texts(known_facts, "known_facts", MAX_KNOWN_FACTS)
    unresolved = _texts(unresolved_parts, "unresolved_parts", MAX_UNRESOLVED_PARTS)
    assumptions = _texts(explicit_assumptions, "explicit_assumptions", MAX_ASSUMPTIONS)
    if any(not isinstance(item, bool) for item in (
        clarification_needed,
        qualified_retrieval_needed,
        deterministic_calculation_needed,
        non_identifiable,
        continue_authorized,
    )):
        raise RoundFiveError("sufficiency flags must be boolean")
    reasons: list[str] = []
    if non_identifiable:
        decision = SufficiencyDecision.BLOCKED
        reasons.append("work is non-identifiable or blocked")
    elif clarification_needed:
        decision = SufficiencyDecision.CLARIFICATION
        reasons.append("a user clarification changes the problem boundary")
    elif qualified_retrieval_needed:
        decision = SufficiencyDecision.RETRIEVAL
        reasons.append("a load-bearing fact requires qualified external evidence")
    elif deterministic_calculation_needed:
        decision = SufficiencyDecision.CALCULATION
        reasons.append("a deterministic calculation can resolve a bounded question")
    elif assumptions:
        decision = SufficiencyDecision.ASSUMPTION
        reasons.append("continuation depends on explicitly named assumptions")
    elif unresolved:
        decision = (
            SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY
            if continue_authorized
            else SufficiencyDecision.CLARIFICATION
        )
        reasons.append("known facts do not resolve every declared part")
    else:
        decision = SufficiencyDecision.SUFFICIENT
        reasons.append("declared information is sufficient for the bounded step")
    return InformationSufficiency(
        decision,
        known,
        unresolved,
        assumptions,
        tuple(reasons),
        continue_authorized,
    )


@dataclass(frozen=True)
class ConsumerAtlasEntry:
    """One read path for a state field, including deliberate no-op fields."""

    field: str
    writers: tuple[str, ...]
    readers: tuple[str, ...]
    decision_change: str
    computation_change: str
    consequence: str
    causal: bool = True

    def __post_init__(self) -> None:
        _identifier(self.field, "atlas field")
        for name in ("writers", "readers"):
            object.__setattr__(self, name, _texts(getattr(self, name), f"atlas {name}", 8))
        for name in ("decision_change", "computation_change", "consequence"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise RoundFiveError(f"atlas {name} is required")
        if not isinstance(self.causal, bool):
            raise RoundFiveError("atlas causal flag must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "writers": list(self.writers),
            "readers": list(self.readers),
            "decision_change": self.decision_change,
            "computation_change": self.computation_change,
            "consequence": self.consequence,
            "causal": self.causal,
        }


@dataclass(frozen=True)
class ConsumerNoOp:
    """A read/write audit result for state that cannot change the outcome."""

    field: str
    changed: bool
    consumed: bool
    reason: str
    causal: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "changed": self.changed,
            "consumed": self.consumed,
            "reason": self.reason,
            "causal": self.causal,
        }


def consumer_atlas() -> tuple[ConsumerAtlasEntry, ...]:
    """Return the static cross-layer write/read/decision/computation atlas."""

    entries = (
        ConsumerAtlasEntry(
            "objective",
            ("human_input", "objective_contract"),
            ("problem_graph", "prediction", "execution"),
            "sets the task boundary",
            "binds every downstream identity",
            "work is scoped to one objective",
        ),
        ConsumerAtlasEntry(
            "recognition_projection",
            ("recognition_memory",),
            ("cognition_projection", "semantic_context"),
            "selects candidate context",
            "changes capability availability and focused context",
            "can change the selected route without becoming evidence",
        ),
        ConsumerAtlasEntry(
            "semantic_trace",
            ("semantic_capability",),
            ("prediction_commitment", "attribution"),
            "declares a candidate explanation",
            "changes the sealed cognitive output",
            "eligible grounded success may earn competence credit",
        ),
        ConsumerAtlasEntry(
            "qualified_retrieval",
            ("external_reality",),
            ("epistemic_revision", "sufficiency"),
            "changes candidate claim status",
            "may change the next bounded operation",
            "never independently earns adaptive credit",
            False,
        ),
        ConsumerAtlasEntry(
            "human_input",
            ("human",),
            ("sufficiency", "candidate_context"),
            "may clarify, constrain, or authorize continuation",
            "does not create execution truth",
            "human contribution credit is withheld",
            False,
        ),
        ConsumerAtlasEntry(
            "deterministic_tool",
            ("tool_adapter",),
            ("candidate_context",),
            "may provide a declared calculation",
            "does not independently create competence",
            "tool contribution credit is withheld",
            False,
        ),
        ConsumerAtlasEntry(
            "grounded_outcome",
            ("grounded_execution_verifier",),
            ("comparison", "settlement", "adaptive_substrate"),
            "settles success or failure",
            "updates one existing route within its bound",
            "later cognition reads the changed topology",
        ),
        ConsumerAtlasEntry(
            "route_topology",
            ("adaptive_substrate",),
            ("cognition_projection", "route_selection"),
            "changes route preference",
            "changes later topology reads and capability selection",
            "a later bounded computation observes the learned weight",
        ),
        ConsumerAtlasEntry(
            "declared_only_model_fields",
            ("semantic_provider",),
            ("candidate_context",),
            "no constitutional decision",
            "no adaptive computation",
            "intentional no-op until independently grounded",
            False,
        ),
    )
    if len(entries) > MAX_ATLAS_ENTRIES:
        raise RoundFiveError("consumer atlas exceeds its bound")
    return entries


def audit_consumer_atlas(
    before: Mapping[str, Any] | None = None,
    after: Mapping[str, Any] | None = None,
) -> tuple[ConsumerNoOp, ...]:
    """Audit declared state against the atlas without inferring causality.

    ``before`` and ``after`` are optional snapshots.  The audit reports
    declared-but-noncausal fields explicitly and treats an absent snapshot as
    an unobserved change rather than a successful consequence.
    """

    before = before or {}
    after = after or {}
    result: list[ConsumerNoOp] = []
    for entry in consumer_atlas():
        changed = entry.field in before and entry.field in after and before[entry.field] != after[entry.field]
        result.append(
            ConsumerNoOp(
                entry.field,
                changed,
                bool(entry.readers),
                (
                    "declared state is read by a downstream consumer"
                    if entry.causal
                    else "declared candidate input is intentionally noncausal"
                ),
                entry.causal,
            )
        )
    return tuple(result)


@dataclass(frozen=True)
class RoundFiveTrace:
    """The complete immutable cross-layer trace for one bounded integration."""

    objective: Objective
    problem_id: str
    processing_trace: ProcessingTrace
    sufficiency: InformationSufficiency
    human_inputs: tuple[HumanInput, ...]
    commitment: PredictionCommitment
    outcome: Any
    learning_episode: GroundedLearningEpisode
    settlement_record: SettlementRouteRecord
    adaptive_audit: AdaptiveAudit | None
    topology_before: RouteTopology
    topology_after: RouteTopology
    atlas_audit: tuple[ConsumerNoOp, ...]
    candidate_only: bool = True
    trace_hash: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.objective, Objective):
            raise RoundFiveError("round-five objective is invalid")
        _identifier(self.problem_id, "round-five problem_id")
        if not isinstance(self.processing_trace, ProcessingTrace):
            raise RoundFiveError("round-five processing trace is invalid")
        if self.processing_trace.problem.problem_id != self.problem_id:
            raise RoundFiveError("round-five problem binding is invalid")
        if not isinstance(self.sufficiency, InformationSufficiency):
            raise RoundFiveError("round-five sufficiency is invalid")
        humans = tuple(self.human_inputs)
        if len(humans) > MAX_HUMAN_INPUTS or not all(isinstance(item, HumanInput) for item in humans):
            raise RoundFiveError("round-five human inputs are invalid")
        object.__setattr__(self, "human_inputs", humans)
        if not isinstance(self.commitment, PredictionCommitment):
            raise RoundFiveError("round-five commitment is invalid")
        if not isinstance(self.learning_episode, GroundedLearningEpisode):
            raise RoundFiveError("round-five learning episode is invalid")
        if not isinstance(self.settlement_record, SettlementRouteRecord):
            raise RoundFiveError("round-five settlement record is invalid")
        if not isinstance(self.topology_before, RouteTopology) or not isinstance(self.topology_after, RouteTopology):
            raise RoundFiveError("round-five topology records are invalid")
        commitment = self.learning_episode.commitment
        comparison = self.learning_episode.comparison
        selection = self.settlement_record.selection
        cycle = self.settlement_record.constitutional_trace
        if (
            self.commitment != commitment
            or commitment.commitment_hash != self.commitment.commitment_hash
            or self.outcome != self.learning_episode.outcome
            or selection.route_id != commitment.route_id
            or selection.transaction_id != commitment.transaction_id
            or selection.objective_id != commitment.objective_id
            or selection.task_state_id != commitment.task_state_id
            or selection.task_state_version != commitment.task_state_version
            or self.settlement_record.settlement.observed_outcome
            != comparison.observed_outcome.value
            or cycle.execution.observations.get("record_hash")
            != self.learning_episode.outcome.record_hash
        ):
            raise RoundFiveError("round-five causal bindings are inconsistent")
        if self.learning_episode.attribution.eligible:
            if self.adaptive_audit is None or not self.learning_episode.adaptive_learning:
                raise RoundFiveError("eligible round-five trace lacks delegated learning")
            if self.topology_before == self.topology_after:
                raise RoundFiveError("eligible round-five trace did not change topology")
        elif (
            self.adaptive_audit is not None
            or self.topology_before != self.topology_after
            or self.learning_episode.adaptive_learning is not None
        ):
            raise RoundFiveError("withheld round-five trace claims adaptive change")
        atlas = tuple(self.atlas_audit)
        if not all(isinstance(item, ConsumerNoOp) for item in atlas):
            raise RoundFiveError("round-five atlas audit is invalid")
        if atlas != audit_consumer_atlas(
            _snapshot(self.topology_before), _snapshot(self.topology_after)
        ):
            raise RoundFiveError("round-five consumer atlas audit is inconsistent")
        object.__setattr__(self, "atlas_audit", atlas)
        if self.candidate_only is not True:
            raise RoundFiveError("round-five trace must remain candidate-only")
        expected = _digest(self.to_dict(include_hash=False))
        if self.trace_hash:
            if self.trace_hash != expected:
                raise RoundFiveError("round-five trace hash is invalid")
        else:
            object.__setattr__(self, "trace_hash", expected)

    @property
    def changed_topology(self) -> bool:
        return self.topology_before != self.topology_after

    @property
    def later_computation(self) -> tuple[tuple[str, float], ...]:
        """The topology read that a later bounded cognition projection consumes."""

        return tuple((route.route_id, route.weight) for route in self.topology_after.routes)

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = {
            "objective": self.objective.to_dict(),
            "problem_id": self.problem_id,
            "processing_trace": self.processing_trace.to_dict(),
            "sufficiency": self.sufficiency.to_dict(),
            "human_inputs": [item.to_dict() for item in self.human_inputs],
            "commitment": self.commitment.to_dict(),
            "outcome": self.outcome.to_dict() if hasattr(self.outcome, "to_dict") else _jsonable(self.outcome),
            "learning_episode": self.learning_episode.to_dict(),
            "settlement_record": {
                "record_id": self.settlement_record.record_id,
                "selection": _selection_dict(self.settlement_record.selection),
                "constitutional_trace": self.settlement_record.constitutional_trace.to_dict(),
                "provenance": _jsonable(self.settlement_record.provenance),
            },
            "adaptive_audit": self.adaptive_audit.to_dict() if self.adaptive_audit is not None else None,
            "topology_before": _topology_dict(self.topology_before),
            "topology_after": _topology_dict(self.topology_after),
            "atlas_audit": [item.to_dict() for item in self.atlas_audit],
            "candidate_only": self.candidate_only,
        }
        if include_hash:
            value["trace_hash"] = self.trace_hash
        return value


def _validate_human_continuation(
    sufficiency: InformationSufficiency,
    human_inputs: tuple[HumanInput, ...],
) -> None:
    authorizations = tuple(
        item.role is HumanInputRole.AUTHORIZATION
        and item.authority is HumanAuthority.CONTINUE
        and item.relevant
        for item in human_inputs
    )
    if sufficiency.decision is SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY:
        if not any(authorizations):
            raise RoundFiveError(
                "continuation with unresolved information requires typed human authorization"
            )
        scoped = tuple(
            item
            for item in human_inputs
            if item.role is HumanInputRole.AUTHORIZATION
            and item.authority is HumanAuthority.CONTINUE
            and item.relevant
            and tuple(item.value.get("unresolved_parts", ()))
            == sufficiency.unresolved_parts
        )
        if not scoped:
            raise RoundFiveError(
                "continued uncertainty authorization is not bound to unresolved scope"
            )


def _snapshot(topology: RouteTopology) -> dict[str, Any]:
    return {
        "route_topology": _topology_dict(topology),
        "routes": {item.route_id: item.weight for item in topology.routes},
    }


def run_round_five(
    objective: Objective,
    problem: Any,
    adaptive_state: AdaptiveState,
    processing_trace: ProcessingTrace,
    grounded_request: GroundedExecutionRequest,
    *,
    executor: GroundedExecutionExecutor,
    authorized_state: TaskState,
    expected_outcome: str = "success",
    success_criteria: Iterable[str] = (),
    commitment_id: str | None = None,
    outcome_id: str | None = None,
    context_id: str = "candidate-work",
    sufficiency: InformationSufficiency | None = None,
    human_inputs: Iterable[HumanInput] = (),
    contributions: Iterable[ContributionRecord] = (),
) -> RoundFiveTrace:
    """Run the one canonical Round 5 path through existing authorities.

    The order is strict: replay cognition, resolve sufficiency, seal and commit
    the exact future request, execute once through the existing bounded
    executor, independently verify it, run the existing constitutional
    settlement, then delegate eligible credit to the existing adaptive
    reducer.  Human, retrieval, tool, partial, contradictory, and insufficient
    inputs remain represented but cannot create competence credit.
    """

    if not isinstance(objective, Objective):
        raise RoundFiveError("round-five requires an Objective")
    if not hasattr(problem, "problem_id") or not hasattr(problem, "original_task"):
        raise RoundFiveError("round-five requires a ProblemGraph")
    if not isinstance(adaptive_state, AdaptiveState):
        raise RoundFiveError("round-five requires AdaptiveState")
    if not isinstance(executor, GroundedExecutionExecutor):
        raise RoundFiveError("round-five requires the existing grounded executor")
    if not isinstance(authorized_state, TaskState):
        raise RoundFiveError("round-five requires an authorized TaskState")
    if (
        authorized_state.objective_id != objective.objective_id
        or authorized_state.phase != "authorized"
        or authorized_state.version != 5
        or authorized_state.authority is not Authority.KRAKEN_CANDIDATE
    ):
        raise RoundFiveError("round-five state is not the candidate authorized state")
    if problem.original_task.task_id != objective.objective_id:
        raise RoundFiveError("objective and problem task identities disagree")
    try:
        validated = replay_processing_trace(processing_trace)
    except Exception as exc:
        raise RoundFiveError("round-five requires a replayable processing trace") from exc
    if validated.problem.problem_id != problem.problem_id:
        raise RoundFiveError("processing trace is bound to another problem")
    humans = tuple(human_inputs)
    if len(humans) > MAX_HUMAN_INPUTS or not all(isinstance(item, HumanInput) for item in humans):
        raise RoundFiveError("human inputs are invalid")
    selected_sufficiency = sufficiency or assess_information_sufficiency(
        tuple(item.statement for item in validated.claims),
        unresolved_parts=tuple(
            item.statement for item in validated.claims if item.unresolved_ids
        ),
        continue_authorized=any(item.authority is HumanAuthority.CONTINUE for item in humans),
    )
    trace_unresolved = tuple(
        item.statement for item in validated.claims if item.unresolved_ids
    )
    if selected_sufficiency.decision is SufficiencyDecision.SUFFICIENT and trace_unresolved:
        raise RoundFiveError(
            "supplied sufficiency contradicts unresolved processing claims"
        )
    if selected_sufficiency.decision is SufficiencyDecision.CONTINUE_WITH_UNCERTAINTY:
        if selected_sufficiency.unresolved_parts != trace_unresolved:
            raise RoundFiveError(
                "continued uncertainty must exactly preserve unresolved processing claims"
            )
        if not trace_unresolved:
            raise RoundFiveError(
                "continued uncertainty requires matching unresolved processing claims"
            )
    if selected_sufficiency.decision in {
        SufficiencyDecision.BLOCKED,
        SufficiencyDecision.CLARIFICATION,
        SufficiencyDecision.RETRIEVAL,
        SufficiencyDecision.CALCULATION,
        SufficiencyDecision.ASSUMPTION,
    }:
        raise RoundFiveError(
            f"round-five cannot execute from sufficiency decision {selected_sufficiency.decision.value}"
        )
    _validate_human_continuation(selected_sufficiency, humans)
    if expected_outcome != "success":
        raise RoundFiveError(
            "the existing constitutional cycle can only bind a success prediction"
        )
    criteria = tuple(success_criteria) or objective.success_criteria or ("bounded criteria pass",)
    extras = tuple(contributions) + tuple(item.to_contribution() for item in humans)
    if commitment_id is None:
        commitment_id = f"{objective.objective_id}-round-five-commitment"
    if outcome_id is None:
        outcome_id = f"{commitment_id}-outcome"
    try:
        selection = select_candidate_route(
            adaptive_state.route_topology,
            context_id,
            transaction_id=grounded_request.transaction_id,
            objective_id=objective.objective_id,
            task_state_id=authorized_state.state_id,
            task_state_version=authorized_state.version,
        )
        selected_capability = next(
            (
                item
                for item in validated.capabilities
                if item.capability_id == validated.decision.selected_capability_id
            ),
            None,
        )
        if (
            selected_capability is None
            or selected_capability.route_id != selection.route_id
        ):
            raise RoundFiveError(
                "adaptive route selection does not match the cognition that produced the prediction"
            )
        sealed_request = seal_prediction_request(
            validated,
            grounded_request,
            commitment_id=commitment_id,
            expected_outcome=expected_outcome,
            success_criteria=criteria,
            contributions=extras,
            route_id=selection.route_id,
        )
        commitment = commit_prediction(
            validated,
            commitment_id=commitment_id,
            grounded_request=sealed_request,
            expected_outcome=expected_outcome,
            success_criteria=criteria,
            contributions=extras,
            authorized_state=authorized_state,
            transaction_id=sealed_request.transaction_id,
            route_id=selection.route_id,
        )
    except RoundFiveError:
        raise
    except Exception as exc:
        raise RoundFiveError("round-five prediction precommitment failed") from exc
    try:
        record = executor.execute(sealed_request, authorized_state=authorized_state)
        verifier = executor.verifier()
        verified = verifier.verify(
            record, request=sealed_request, authorized_state=authorized_state
        )
        cycle_trace = ConstitutionalCycle().run(
            objective,
            grounded_execution=verified,
            grounded_request=sealed_request,
            grounded_verifier=verifier,
            grounded_delivery_ledger=verifier.delivery_ledger,
        )
    except Exception as exc:
        raise RoundFiveError("round-five grounded execution or settlement failed") from exc
    try:
        outcome = bind_grounded_outcome(
            commitment,
            outcome_id=outcome_id,
            verified_execution=verified,
            request=sealed_request,
            authorized_state=authorized_state,
            trusted_executor=executor.trusted_executor(),
        )
        episode = evaluate_grounded_outcome(commitment, outcome)
        settlement_record = SettlementRouteRecord(
            f"{commitment_id}-settlement-route",
            selection,
            cycle_trace,
            provenance={
                "round": "five",
                "commitment_id": commitment_id,
                "transaction_id": selection.transaction_id,
                "objective_id": selection.objective_id,
                "task_state_id": selection.task_state_id,
                "task_state_version": selection.task_state_version,
                "route_id": selection.route_id,
                "settlement_id": cycle_trace.settlement.settlement_id,
                "evidence_ids": tuple(
                    item.evidence_id for item in cycle_trace.evidence
                ),
            },
            grounded_execution=verified,
            grounded_request=sealed_request,
            grounded_verifier=verifier,
        )
    except Exception as exc:
        raise RoundFiveError("round-five settlement binding failed") from exc
    topology_before = adaptive_state.route_topology
    adaptive_after = adaptive_state
    audit: AdaptiveAudit | None = None
    if episode.attribution.eligible:
        try:
            _, episode = apply_outcome_learning(
                topology_before, episode, settlement_record
            )
            adaptive_after, audit = apply_grounded_adaptation(
                adaptive_state, settlement_record
            )
        except Exception as exc:
            raise RoundFiveError("eligible grounded credit failed in existing adaptive reducer") from exc
    atlas = audit_consumer_atlas(_snapshot(topology_before), _snapshot(adaptive_after.route_topology))
    trace = RoundFiveTrace(
        objective,
        problem.problem_id,
        validated,
        selected_sufficiency,
        humans,
        commitment,
        outcome,
        episode,
        settlement_record,
        audit,
        topology_before,
        adaptive_after.route_topology,
        atlas,
    )
    if len(json.dumps(trace.to_dict(), sort_keys=True).encode("utf-8")) > MAX_TRACE_BYTES:
        raise RoundFiveError("round-five trace exceeds its byte bound")
    return trace


def replay_round_five(
    trace: RoundFiveTrace,
    *,
    adaptive_state: AdaptiveState,
) -> tuple[RoundFiveTrace, AdaptiveState]:
    """Replay the composed trace without invoking models, tools, or execution."""

    if not isinstance(trace, RoundFiveTrace):
        raise RoundFiveError("round-five replay requires a trace")
    if not isinstance(adaptive_state, AdaptiveState):
        raise RoundFiveError("round-five replay requires the immutable initial adaptive state")
    if trace.trace_hash != _digest(trace.to_dict(include_hash=False)):
        raise RoundFiveError("round-five replay trace hash is invalid")
    if adaptive_state.route_topology != trace.topology_before:
        raise RoundFiveError("replay initial topology does not match the trace")
    try:
        replayed_processing = replay_processing_trace(trace.processing_trace)
        replayed_episode, replayed_topology = replay_outcome_learning(
            trace.learning_episode,
            topology=adaptive_state.route_topology,
            settlement_record=trace.settlement_record,
        )
        verifier = GroundedExecutionVerifier(trace.outcome.trusted_executor)
        replayed_grounded = replay_grounded_execution(
            trace.outcome.verified_execution.record,
            request=trace.outcome.request,
            authorized_state=trace.outcome.authorized_state,
            verifier=verifier,
        )
    except Exception as exc:
        raise RoundFiveError("round-five offline replay rejected the trace") from exc
    if replayed_processing.structural_hash != trace.processing_trace.structural_hash:
        raise RoundFiveError("processing replay changed the composed trace")
    if replayed_episode.to_dict(include_hash=False) != trace.learning_episode.to_dict(include_hash=False):
        raise RoundFiveError("outcome-learning replay changed the composed trace")
    if replayed_grounded != trace.outcome.verified_execution:
        raise RoundFiveError("grounded execution replay changed the outcome")
    if trace.adaptive_audit is None:
        return trace, adaptive_state
    if (
        replayed_topology is None
        or replayed_episode.adaptive_learning
        != trace.learning_episode.adaptive_learning
    ):
        raise RoundFiveError("outcome-learning replay changed delegated credit")
    replayed_state, audit = apply_grounded_adaptation(
        adaptive_state, trace.settlement_record
    )
    if audit != trace.adaptive_audit or replayed_state.route_topology != trace.topology_after:
        raise RoundFiveError("adaptive replay changed the composed result")
    return trace, replayed_state


def run_round_five_integration(*args: Any, **kwargs: Any) -> RoundFiveTrace:
    """Descriptive alias for callers that name the stage explicitly."""

    return run_round_five(*args, **kwargs)


__all__ = [
    "ConsumerAtlasEntry",
    "ConsumerNoOp",
    "HumanAuthority",
    "HumanInput",
    "HumanInputRole",
    "InformationSufficiency",
    "RoundFiveError",
    "RoundFiveTrace",
    "SufficiencyDecision",
    "assess_information_sufficiency",
    "audit_consumer_atlas",
    "consumer_atlas",
    "replay_round_five",
    "run_round_five",
    "run_round_five_integration",
]