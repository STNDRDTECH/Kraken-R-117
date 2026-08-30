"""Round 3 qualification for bounded external-reality metabolism."""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import pytest

from kraken_r.adaptive_substrate import AdaptiveState
from kraken_r.cognition_kernel import (
    CognitionValidationError,
    EpistemicClaim,
    EpistemicStatus,
    FalsificationCondition,
    OperationResult,
    ProcessingCapability,
    ProcessingOperation,
    ProblemBranch,
    ProblemGraph,
    ProblemRequirement,
    ProblemSubtask,
    SatisfactionState,
    VerificationPressure,
    replay_processing_trace,
    run_connected_processing,
)
from kraken_r.external_reality import (
    ClaimAssessment,
    ClaimRelation,
    EvidenceNeed,
    ExternalRealityError,
    RetrievalObservation,
    SourceArtifact,
    SourceRole,
    derive_evidence_needs,
    legal_currentness_fixture,
    revise_candidate_claims,
    scientific_regime_fixture,
    source_attestation_payload,
    validate_retrieval_observation,
)
from kraken_r.plastic_routing import CandidateRoute, RouteTopology
from kraken_r.task_integrity import OriginalTask


_SIGNED_FIXTURES = {
    "feeac0f44c9349beb6fdfa2feb7183b157ad50309d12309a40107d4757dfde20": "9PIFIxroS4l5sbt9XaA9qqTW0G7ws7irmaykWtVlv2p5De/aTgSp3Q6z70f7fIz5uDS8XQVZ2YOU7V+6PvHCAg==",
    "1f8546ee80b6c65fd4332e4eb9e5522a662c6e069e1866d4deca287f71e8f1d5": "4Zlpwi3TJzsGAwhYvuduO/B7VMrTCThgw/iIS1kF+DPbZxNCM9L2iEMZqcOfsb5cW8DKF8oZHNmTzBMAMqDIAg==",
    "1cf4e8e78409735accd0d48b0311437f7cd48feb6434236e6f1c83c74917d956": "FoCCn3heKRECD47AYVKpYYyhEuINxnmcscRnq3gZZVoRvR6z6i2e3som/Wa8wU/x2wK7TS6U+OX6aemp8YnoAw==",
    "209819cc9624079ec81b75fe9a8254b8f4b319fbb44ce08cad27037d1b8f3e3f": "fiRdIjaHRAJd7/k+FQxzitr77klYmWHQ+2A7CaU7SU0eyt+n++WBWer7VXXQ5KcxaCpERnEci532/iWbnFY9Bg==",
    "560288fb362793c8c082db2d0f7f4a8d9d5b35105a173a4a66ddad8aeac56c5e": "O0edc/Y4zMOWylr7GlwZjR8tr/KFhI9XbWfR5ca6stBQqWOCqex0z5KYj7F8KKoxN4WnlWwoUjm7xAJw1FkwDQ==",
    "6e085a8e76022fc75e490cacb56c8f833996c25bf37ba68365829e6a464f52c2": "2MdcIoa4Ny9gAgOp15hH22kMrz/1Wn7mQqygwbKpgvCrqxQ3daQ1qXRMM8T140qP+5wxvMDEWwDg8OE6jtVuCw==",
    "ce04f6fa4045a56b2e181c19e65b6e5a6eb708b1ad2fe9ff49d7841eeccbb9d0": "/tC9ATnS7NgL7Qf8EICH3VE9ZHKMtOVUqTv7sXK6a+kUkkdN9u3vC8p2azrijhTWQjQN1AneNbjimIjZKCV8BQ==",
    "f4a450ac4d0f1265cdff8b7c77d8ead2f3e417b8a08f8abda0f156a857da7edb": "mMbi5NNOlxPcxL6Ael7omjaHv/p6Ru/OWG3g+kut+Ag6pFOSsl+Z+6/FgHjPoVbo2GOvzD93vESrmiYEZknYDg==",
    "85194f8e418aaa71bf974665de9eafa4dcac221974f81892843c34372bd1d257": "twRiaOvSzjkMSSh78fMf1NzvEDiX7hzyah5ZSf+3l2kjK+UAfNySuMuA+7ygT7kje6VWfLc7E7pVZOU3aSDCAA==",
}


def _problem() -> ProblemGraph:
    original = OriginalTask(
        "external-task",
        "Resolve a load-bearing current claim without granting retrieval authority.",
        "fixture",
        {"candidate_only": True},
    )
    return ProblemGraph(
        "external-problem",
        original,
        "Is the load-bearing claim supported?",
        (
            ProblemRequirement(
                "external-requirement",
                "Keep the requirement unresolved until constitutional evidence exists.",
                status=SatisfactionState.UNRESOLVED,
            ),
        ),
        (
            ProblemSubtask(
                "external-subtask",
                "Inspect the external claim.",
                ("external-requirement",),
                branch_id="external-branch",
            ),
        ),
        branches=(
            ProblemBranch(
                "external-branch",
                "External reality",
                "external-environment",
                ("external-subtask",),
            ),
        ),
        environments=("external-environment",),
        satisfaction={
            "external-requirement": SatisfactionState.UNRESOLVED,
            "external-subtask": SatisfactionState.UNRESOLVED,
        },
    )


def _claim(*, pressure: VerificationPressure | None = None) -> EpistemicClaim:
    return EpistemicClaim(
        "external-claim",
        "external-problem",
        "The current primary record supports candidate alpha.",
        status=EpistemicStatus.UNRESOLVED,
        use_site="branch-selection",
        verification_pressure=pressure
        or VerificationPressure(
            stakes=0.8,
            volatility=0.8,
            precision=0.9,
            contestability=0.7,
            actionability=0.8,
        ),
        falsification_conditions=(
            FalsificationCondition(
                "external-condition",
                "A current primary record supports candidate alpha.",
                "A current primary record contradicts candidate alpha.",
            ),
        ),
    )


def _need(claim: EpistemicClaim) -> EvidenceNeed:
    return EvidenceNeed(
        "external-need",
        claim.problem_id,
        claim.claim_id,
        "The claim controls branch selection and can change over time.",
        claim.verification_pressure,
        currentness_required=True,
        max_age_days=30,
        required_jurisdictions=("fictionland",),
        required_scopes=("candidate-alpha",),
        required_regimes=("current",),
        required_source_kinds=("official_record", "secondary_analysis"),
        required_methods=("primary-record", "document-analysis"),
        downstream_leverage=("external-subtask",),
        satisfying_conditions=("The current record supports alpha.",),
        contradicting_conditions=("The current record rejects alpha.",),
    )


def _source(
    artifact_id: str = "external-source",
    *,
    source_id: str | None = None,
    source_kind: str = "official_record",
    role: SourceRole = SourceRole.EVIDENTIARY,
    family: str = "external-family",
    root: str | None = None,
    parents: tuple[str, ...] = (),
    retrieved_on: str = "2026-08-20",
    jurisdiction: tuple[str, ...] = ("fictionland",),
    scopes: tuple[str, ...] = ("candidate-alpha",),
    regimes: tuple[str, ...] = ("current",),
    method: str = "primary-record",
    content: str = "The current primary record supports candidate alpha.",
    provenance=None,
) -> SourceArtifact:
    source_id = source_id or f"{artifact_id}-identity"
    root_artifact_id = root or artifact_id
    content_hash = hashlib.sha256(
        json.dumps(
            content,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    signature = None
    verifier_id = None
    if role is SourceRole.EVIDENTIARY:
        payload = source_attestation_payload(
            artifact_id=artifact_id,
            source_id=source_id,
            source_kind=source_kind,
            independence_family=family,
            root_artifact_id=root_artifact_id,
            parent_artifact_ids=parents,
            retrieved_on=retrieved_on,
            effective_from="2026-01-01",
            effective_until=None,
            jurisdiction=jurisdiction,
            scopes=scopes,
            regimes=regimes,
            method=method,
            content_hash=content_hash,
        )
        payload_hash = hashlib.sha256(payload).hexdigest()
        signature = _SIGNED_FIXTURES.get(payload_hash, "invalid-signature")
        verifier_id = "kraken-r-qualification-verifier"
    return SourceArtifact.create(
        artifact_id,
        source_id,
        source_kind,
        role,
        family,
        root_artifact_id=root_artifact_id,
        parent_artifact_ids=parents,
        retrieved_on=retrieved_on,
        effective_from="2026-01-01",
        jurisdiction=jurisdiction,
        scopes=scopes,
        regimes=regimes,
        method=method,
        content=content,
        verifier_id=verifier_id,
        attestation_signature=signature,
        provenance=provenance or {"retrieval": "fixture"},
    )


def _observation(
    request_id: str,
    claim: EpistemicClaim,
    need: EvidenceNeed,
    sources: tuple[SourceArtifact, ...],
    *,
    relations: tuple[ClaimRelation, ...] | None = None,
) -> RetrievalObservation:
    relations = relations or tuple(ClaimRelation.ENTAILS for _ in sources)
    return RetrievalObservation(
        "external-observation",
        request_id,
        claim.problem_id,
        "2026-08-30",
        (need.need_id,),
        sources,
        tuple(
            ClaimAssessment.create(
                f"assessment-{index}",
                need.need_id,
                claim,
                source.artifact_id,
                relation,
                source.content,
                "The quoted primary text is bound to this exact candidate claim.",
            )
            for index, (source, relation) in enumerate(zip(sources, relations))
        ),
    )


def _adaptive() -> AdaptiveState:
    state = AdaptiveState.fixture("external-adaptive")
    topology = RouteTopology(
        state.route_topology.topology_id,
        1,
        (
            CandidateRoute(
                "external-retrieve-route",
                "candidate-work",
                "candidate",
                "retrieve",
                0.60,
            ),
            CandidateRoute(
                "external-compute-route",
                "candidate-work",
                "candidate",
                "compute",
                0.50,
            ),
        ),
        generation=1,
    )
    return replace(state, generation=1, route_topology=topology)


def _capabilities() -> tuple[ProcessingCapability, ...]:
    return (
        ProcessingCapability(
            "external-retrieve-capability",
            ProcessingOperation.RETRIEVE,
            "external-retrieve-route",
            "external-branch",
            ("external-subtask",),
        ),
        ProcessingCapability(
            "external-compute-capability",
            ProcessingOperation.CHECK_LOGIC,
            "external-compute-route",
            "external-branch",
            ("external-subtask",),
        ),
    )


def test_only_unresolved_load_bearing_claims_create_evidence_needs():
    high = _claim()
    low = replace(
        high,
        claim_id="stable-low-stakes",
        verification_pressure=VerificationPressure(stakes=0.1, precision=0.1),
        falsification_conditions=(),
    )
    supported = replace(high, claim_id="already-supported", status=EpistemicStatus.SUPPORT)

    needs = derive_evidence_needs(high.problem_id, (low, high, supported))

    assert tuple(item.claim_id for item in needs) == (high.claim_id,)
    assert needs[0].currentness_required is True
    assert needs[0].candidate_only is True


def test_legal_currentness_and_scientific_regime_fixtures_are_distinct():
    legal_need, legal = legal_currentness_fixture()
    science_need, science = scientific_regime_fixture()

    assert legal_need.currentness_required is True
    assert legal_need.required_jurisdictions == ("fictionland",)
    assert science_need.currentness_required is False
    assert science_need.required_regimes == ("regime-a",)
    assert legal.retrieval_hash != science.retrieval_hash


def test_currentness_requirement_cannot_omit_its_age_bound():
    claim = _claim()
    with pytest.raises(ExternalRealityError, match="require max_age_days"):
        replace(_need(claim), max_age_days=None)

    serialized = _need(claim).to_dict()
    serialized["max_age_days"] = None
    with pytest.raises(ExternalRealityError, match="require max_age_days"):
        EvidenceNeed.from_dict(serialized)


@pytest.mark.parametrize(
    "source_kwargs,match",
    (
        ({"retrieved_on": "2026-01-01"}, "stale"),
        ({"jurisdiction": ("elsewhere",)}, "jurisdiction"),
        ({"regimes": ("historical",)}, "regime"),
        ({"scopes": ("candidate-beta",)}, "scope"),
    ),
)
def test_stale_or_inapplicable_sources_fail_closed(source_kwargs, match):
    claim = _claim()
    need = _need(claim)
    observation = _observation(
        "external-request",
        claim,
        need,
        (_source(**source_kwargs),),
    )

    with pytest.raises(ExternalRealityError, match=match):
        validate_retrieval_observation(observation, (claim,), (need,))


def test_discovery_results_cannot_promote_themselves_or_support_claims():
    with pytest.raises(ExternalRealityError, match="promoted"):
        _source(source_kind="search_result", role=SourceRole.EVIDENTIARY)

    claim = _claim()
    need = _need(claim)
    discovery = _source(
        source_kind="search_result",
        role=SourceRole.DISCOVERY,
        method="document-analysis",
    )
    assessment = ClaimAssessment.create(
        "discovery-assessment",
        need.need_id,
        claim,
        discovery.artifact_id,
        ClaimRelation.ENTAILS,
        discovery.content,
        "A search result is only a discovery lead.",
    )
    with pytest.raises(ExternalRealityError, match="discovery source"):
        RetrievalObservation(
            "discovery-observation",
            "external-request",
            claim.problem_id,
            "2026-08-30",
            (need.need_id,),
            (discovery,),
            (assessment,),
            (discovery.artifact_id,),
        )


def test_correlated_descendants_count_as_one_independence_family():
    claim = _claim()
    need = _need(claim)
    root = _source()
    secondary = _source(
        "secondary-source",
        source_kind="secondary_analysis",
        family=root.independence_family,
        root=root.artifact_id,
        parents=(root.artifact_id,),
        method="document-analysis",
        content="The current primary record supports candidate alpha. Secondary summary.",
    )
    def retrieve(request):
        observation = _observation(
            request.request_id, claim, need, (root, secondary)
        )
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"candidate_revision": "correlated-family"},
            retrieval_observation=observation,
        )

    trace = run_connected_processing(
        _problem(),
        _capabilities(),
        _adaptive(),
        claims=(claim,),
        evidence_needs=(need,),
        operation_callbacks={ProcessingOperation.RETRIEVE: retrieve},
    )
    revised = trace.claims
    revisions = trace.candidate_revisions

    assert revised[0].status is EpistemicStatus.SUPPORT
    assert revisions[0].supporting_artifact_ids == (
        root.artifact_id,
        secondary.artifact_id,
    )
    assert revisions[0].independent_source_count == 1
    assert revised[0].evidence_ids == ()


def test_duplicate_source_identity_cannot_inflate_independence():
    claim = _claim()
    need = _need(claim)
    first = _source("identity-copy-one", family="family-one")
    conflicting = _source(
        "identity-copy-two",
        source_id=first.source_id,
        family="family-two",
    )

    with pytest.raises(ExternalRealityError, match="source identity"):
        _observation(
            "external-request",
            claim,
            need,
            (first, conflicting),
        )


def test_assessment_cannot_revise_a_need_omitted_from_observation_scope():
    first_claim = _claim()
    first_need = _need(first_claim)
    second_claim = replace(first_claim, claim_id="second-external-claim")
    second_need = replace(
        first_need,
        need_id="second-external-need",
        claim_id=second_claim.claim_id,
    )
    source = _source()
    first_assessment = ClaimAssessment.create(
        "first-assessment",
        first_need.need_id,
        first_claim,
        source.artifact_id,
        ClaimRelation.ENTAILS,
        source.content,
        "The requested claim is assessed.",
    )
    undeclared_assessment = ClaimAssessment.create(
        "undeclared-assessment",
        second_need.need_id,
        second_claim,
        source.artifact_id,
        ClaimRelation.ENTAILS,
        source.content,
        "This need was not requested by the observation.",
    )
    observation = RetrievalObservation(
        "cross-need-observation",
        "external-request",
        first_claim.problem_id,
        "2026-08-30",
        (first_need.need_id,),
        (source,),
        (first_assessment, undeclared_assessment),
    )

    with pytest.raises(ExternalRealityError, match="observation scope"):
        validate_retrieval_observation(
            observation,
            (first_claim, second_claim),
            (first_need, second_need),
        )


def test_fabricated_evidentiary_attestation_is_rejected():
    valid = _source()
    with pytest.raises(ExternalRealityError, match="attestation is invalid"):
        replace(
            valid,
            source_id="fabricated-official-source",
        )
    with pytest.raises(ExternalRealityError, match="trusted verifier"):
        SourceArtifact.create(
            "fabricated-artifact",
            "fabricated-source",
            "official_record",
            SourceRole.EVIDENTIARY,
            "fabricated-family",
            retrieved_on="2026-08-20",
            effective_from="2026-01-01",
            jurisdiction=("fictionland",),
            scopes=("candidate-alpha",),
            regimes=("current",),
            method="primary-record",
            content="The current primary record supports candidate alpha.",
            provenance={"retrieval": "self-asserted"},
        )
    with pytest.raises(ExternalRealityError, match="attestation is invalid"):
        replace(
            valid,
            attestation_signature=(
                "3JOk4r5BTv+zyqIUCwzCbZ0gVjbcvRjypiT35Q61HHXNxSkBnZ0t"
                "y1bYHPRtWmZEAUoA2p7QzBqCSMbxpZnkCw=="
            ),
        )


def test_lineage_claim_binding_authority_injection_and_size_fail_closed():
    claim = _claim()
    need = _need(claim)
    root = _source()
    with pytest.raises(ExternalRealityError, match="root lineage|root identity"):
        RetrievalObservation(
            "bad-lineage",
            "external-request",
            claim.problem_id,
            "2026-08-30",
            (need.need_id,),
            (
                root,
                _source(
                    "bad-child",
                    source_kind="secondary_analysis",
                    root="forged-root",
                    parents=(root.artifact_id,),
                    method="document-analysis",
                ),
            ),
            (),
        )

    observation = _observation("external-request", claim, need, (root,))
    bad_hash = replace(observation.assessments[0], claim_hash="forged")
    with pytest.raises(ExternalRealityError, match="claim hash"):
        validate_retrieval_observation(
            replace(observation, assessments=(bad_hash,)),
            (claim,),
            (need,),
        )
    bad_quote = replace(observation.assessments[0], quoted_text="not in source")
    with pytest.raises(ExternalRealityError, match="quote"):
        replace(observation, assessments=(bad_quote,))
    with pytest.raises(ExternalRealityError, match="forbidden authority"):
        _source(provenance={"evidence_ids": ["forged"]})
    with pytest.raises(ExternalRealityError, match="bounded length"):
        _source(content="x" * 24_001)


def test_retrieval_revises_candidate_reasoning_changes_next_selection_and_replays():
    problem = _problem()
    claim = _claim()
    need = _need(claim)
    callbacks = {"count": 0}

    def retrieve(request):
        callbacks["count"] += 1
        observation = _observation(request.request_id, claim, need, (_source(),))
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"candidate_revision": "source-bound"},
            questions=("What constitutional execution could ground the claim?",),
            investigative_needs=("Keep retrieval separate from settlement.",),
            retrieval_observation=observation,
        )

    first = run_connected_processing(
        problem,
        _capabilities(),
        _adaptive(),
        claims=(claim,),
        evidence_needs=(need,),
        operation_callbacks={ProcessingOperation.RETRIEVE: retrieve},
    )

    assert first.decision.operation is ProcessingOperation.RETRIEVE
    assert first.claims[0].status is EpistemicStatus.SUPPORT
    assert first.context.satisfaction == problem.satisfaction
    assert first.result.evidence_ids == ()
    assert first.result.settlement_ids == ()
    assert first.result.adaptive_update_ids == ()
    assert first.candidate_revisions[0].candidate_only is True
    replayed = replay_processing_trace(first.to_dict())
    assert replayed.to_dict() == first.to_dict()
    assert callbacks["count"] == 1

    second = run_connected_processing(
        problem,
        _capabilities(),
        _adaptive(),
        prior_trace=first,
    )
    assert second.evidence_needs == ()
    assert second.decision.operation is ProcessingOperation.CHECK_LOGIC
    assert second.claims[0].status is EpistemicStatus.SUPPORT
    assert second.context.satisfaction == problem.satisfaction

    tampered = deepcopy(first.to_dict())
    tampered["retrieval_observation"]["sources"][0]["content"] = "forged content"
    with pytest.raises(CognitionValidationError):
        replay_processing_trace(tampered)


def test_retrieval_cannot_change_requirements_or_replay_a_fabricated_revision():
    problem = _problem()
    claim = _claim()
    need = _need(claim)

    def authoritative_retrieve(request):
        observation = _observation(request.request_id, claim, need, (_source(),))
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"candidate_revision": "source-bound"},
            downstream_satisfaction={
                "external-requirement": SatisfactionState.SATISFIED
            },
            retrieval_observation=observation,
        )

    with pytest.raises(CognitionValidationError, match="requirement state"):
        run_connected_processing(
            problem,
            _capabilities(),
            _adaptive(),
            claims=(claim,),
            evidence_needs=(need,),
            operation_callbacks={
                ProcessingOperation.RETRIEVE: authoritative_retrieve
            },
        )