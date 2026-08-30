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


def _canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()


_SIGNED_FIXTURES = {
    "07766756dd737b0bfdd6bc8153bca2872f0a2ec79b746bdf8bb99963004fde6d": "HOsdgp60UZpsCVtb4KB8EbIVcKvu6/WsDS5GRMMh+Cvy3pRrc5UxYmHziQmAOvKf97x4dMUKXe55wpRFUeqwAg==",
    "b3b70e93cc481e0e2083a9002c4f3b17d022381b0e1ddf2eca6e64a15388ed42": "QpvLQn7hHwwa8saKRRqqrIDuRUs6TPP3sKi18I5+G3O9g/r9dhnO0krkSGjJeaJZ8C5DFXS3oPheH7wQtBq0Bw==",
    "e034dc78acdbb501472a0ccd17a515d3358f90c0386448c3718e86e4cfddf4b8": "5LB1U2xxqnPbtQTKP7FxQtAfe1lVckvKHV+dXjbNlkeFfj5iw8dEbHq1cUZxuCfm99QBGi3nGqXqTUc7jbHRCQ==",
    "69c31223ac6612d31a7acb567674fcd0f6effa86af8baa39a15c702ddae3d399": "FkCdC1KwJjuAQVZZBy1W+GaOBHenscvBkQHfLGT7chPZoXhOniNCFKIXe+RGv20w7a3ucDKmUDe88bOTgjGlCw==",
    "5a1776a925caf77f711e62826b29864e0dd43cafa6da7551e31fb42255b3654a": "xj26YvqjuSy8+4+NLMeEHQalSuRTTAuo0Q6rHZvNZlGLI8EZDVTebStj+A2o91SI72f2DcFgkaNVRNEgTtLzBA==",
    "b2b83f1ac35df6b18b691324711b6c4f40fd725225633bc8b32220cf6444a22f": "U29u3uxhXHO8LlqGAmUItabkihlS8Xj5YBb+Q6mk3D7GWZB7p1bqwHK9aTUnWH00ulZ2xaMBnjWBUoEDm5BbCg==",
    "296ea0f6c85d596bf2f692ad1456f85787a3a25fd601be2badb4b1105c86c23b": "N81y5EDgm5CoGeNHMyLuveba/0GDkcTha22AnfoVWvIVzCQYqc/AsXZPJwFIIBydpZctnTKUApl/3uWW1HcHAA==",
    "60d7702abe405e82e5591ef41abc727a658c9d3bf5b65fc6a24ffb192defc012": "rdhq4E5chTjxLKHlQdloIx3PIhDRkeH1dkEA6XF54PFZoIg57ONcpjQmMDaduBT3637gF36yHzsx/ydBx4RoBw==",
    "2fb48e00baec378dd97a2e110379876a48498390c248276c29f1c4a933e47a63": "1uz77JfmeQEq6g991Wyn7SDOSWXzcEFmzeIAtlQoPbSww2/2KRGNSrip8oT5c39oMxnOEMfhzDzLM8C7sdqPCg==",
    "be1e8a9654cf0020c2244667f99cf387c2ad8cdccee2ee4a8f006a68eacc063e": "n1NiwaVO4jqm//01NlTyA6n6s6+fTo1VLYqYg12JKJXp3zihGKOHbQb8l+l8p6Ud371pSVNVuXCg61+K5XiCCQ==",
    "d391bad6b93a361ae2ecf79d6593a724d01064ae074b5bb62051d30c54d30547": "BhEzwal6RBtfF7UIBV5BiI4YX2mRMRov/XRsTcLHsSY40bzJWwrDZYJYX4BwJ5QTSKCxg0/b9tA7YBmDAOlaCg==",
    "43b9e693865d0867b15f426edb6c62046fbaf613ec9cdcaa731aff42a964e71b": "Naa7yfZNnD0ChM61e+kgiVVFvw819zxJ8Qj1Vpir/GEAClqBAiSXFuft23M4kGY9UUrCfcKO83TXuKwYeRCLCQ==",
    "2ddbb998acf62ba366c3e97edb898aa17d2464e4c5c682501a700f1cf6e7ad8f": "0VpzAsnZ/5NjAfGu53wGke31+WElRXlo+F1cEHS3kWmlxrHhPeUB+GXm/qLKEAASID6wvaNZABOzI++XRC4GAg==",
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
    source_provenance = provenance or {"retrieval": "fixture"}
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
            provenance=source_provenance,
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
        provenance=source_provenance,
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
                source=source,
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
        source=discovery,
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
        source=source,
    )
    undeclared_assessment = ClaimAssessment.create(
        "undeclared-assessment",
        second_need.need_id,
        second_claim,
        source.artifact_id,
        ClaimRelation.ENTAILS,
        source.content,
        "This need was not requested by the observation.",
        source=source,
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
    with pytest.raises(
        ExternalRealityError,
        match="claim hash|relation qualification",
    ):
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


def test_provenance_substitution_invalidates_source_attestation():
    source = _source()

    with pytest.raises(ExternalRealityError, match="attestation is invalid"):
        replace(
            source,
            provenance={"retrieval": "substituted-origin"},
        )


def test_unrelated_ambiguous_and_insufficient_relations_fail_closed():
    source = _source()
    unrelated = replace(
        _claim(),
        statement="A wholly unrelated proposition is true.",
    )
    unrelated_need = _need(unrelated)

    with pytest.raises(
        ExternalRealityError,
        match="does not match independent qualification",
    ):
        ClaimAssessment.create(
            "unrelated-entailment",
            unrelated_need.need_id,
            unrelated,
            source.artifact_id,
            ClaimRelation.ENTAILS,
            source.content,
            "Exact quote presence is not semantic support.",
            source=source,
        )

    insufficient = ClaimAssessment.create(
        "insufficient-relation",
        unrelated_need.need_id,
        unrelated,
        source.artifact_id,
        ClaimRelation.INSUFFICIENT,
        source.content,
        "The material has no qualified relation to the claim.",
        source=source,
    )
    observation = RetrievalObservation(
        "insufficient-observation",
        "external-request",
        unrelated.problem_id,
        "2026-08-30",
        (unrelated_need.need_id,),
        (source,),
        (insufficient,),
    )
    validate_retrieval_observation(
        observation,
        (unrelated,),
        (unrelated_need,),
    )

    base_claim = _claim()
    claim = replace(
        base_claim,
        falsification_conditions=(
            FalsificationCondition(
                "ambiguous-falsifier",
                "The same text cannot prove both dispositions.",
                base_claim.statement,
            ),
        ),
    )
    ambiguous_source = _source(
        "ambiguous-discovery",
        source_kind="search_result",
        role=SourceRole.DISCOVERY,
        method="document-analysis",
        content=claim.statement,
    )
    with pytest.raises(ExternalRealityError, match="ambiguous"):
        ClaimAssessment.create(
            "ambiguous-relation",
            _need(claim).need_id,
            claim,
            ambiguous_source.artifact_id,
            ClaimRelation.INSUFFICIENT,
            ambiguous_source.content,
            "Both claim and falsifier are present.",
            source=ambiguous_source,
        )


@pytest.mark.parametrize(
    ("artifact_id", "content", "quoted_text"),
    (
        (
            "negated-source",
            "It is false that The current primary record supports candidate alpha.",
            "The current primary record supports candidate alpha.",
        ),
        (
            "quoted-source",
            "The witness quoted: The current primary record supports candidate alpha.",
            "The current primary record supports candidate alpha.",
        ),
        (
            "modal-source",
            "The current primary record might support candidate alpha.",
            "The current primary record might support candidate alpha.",
        ),
        (
            "mixed-source",
            "The current primary record supports candidate alpha. "
            "This statement is disputed and not adopted.",
            "The current primary record supports candidate alpha.",
        ),
    ),
)
def test_contextualized_signed_material_cannot_qualify_support(
    artifact_id,
    content,
    quoted_text,
):
    claim = _claim()
    need = _need(claim)
    source = _source(artifact_id, content=content)

    with pytest.raises(
        ExternalRealityError,
        match="does not match independent qualification",
    ):
        ClaimAssessment.create(
            f"{artifact_id}-assessment",
            need.need_id,
            claim,
            source.artifact_id,
            ClaimRelation.ENTAILS,
            quoted_text,
            "Contextualized material cannot prove the bare claim.",
            source=source,
        )


def test_declared_falsifier_material_qualifies_contradiction():
    source = _source()
    claim = replace(
        _claim(),
        statement="A different proposition remains unresolved.",
        falsification_conditions=(
            FalsificationCondition(
                "source-text-falsifier",
                "The source states the opposite primary fact.",
                source.content,
            ),
        ),
    )
    need = _need(claim)
    assessment = ClaimAssessment.create(
        "qualified-contradiction",
        need.need_id,
        claim,
        source.artifact_id,
        ClaimRelation.CONTRADICTS,
        source.content,
        "The exact declared falsifier is present.",
        source=source,
    )
    observation = RetrievalObservation(
        "contradiction-observation",
        "external-request",
        claim.problem_id,
        "2026-08-30",
        (need.need_id,),
        (source,),
        (assessment,),
    )

    validate_retrieval_observation(observation, (claim,), (need,))
    assert assessment.qualification.relation is ClaimRelation.CONTRADICTS


def test_relation_tampering_and_duplicate_metadata_conflicts_fail_closed():
    claim = _claim()
    need = _need(claim)
    source = _source()
    observation = _observation(
        "external-request",
        claim,
        need,
        (source,),
    )
    tampered_assessment = replace(
        observation.assessments[0],
        relation=ClaimRelation.CONTRADICTS,
    )
    with pytest.raises(
        ExternalRealityError,
        match="relation qualification binding",
    ):
        replace(observation, assessments=(tampered_assessment,))

    first = _source(
        "duplicate-discovery-one",
        source_id="duplicate-discovery-identity",
        source_kind="search_result",
        role=SourceRole.DISCOVERY,
        method="document-analysis",
    )
    second = _source(
        "duplicate-discovery-two",
        source_id="duplicate-discovery-identity",
        source_kind="search_result",
        role=SourceRole.DISCOVERY,
        method="document-analysis",
        jurisdiction=("elsewhere",),
    )
    with pytest.raises(
        ExternalRealityError,
        match="security or qualification metadata",
    ):
        RetrievalObservation(
            "duplicate-metadata-observation",
            "external-request",
            claim.problem_id,
            "2026-08-30",
            (need.need_id,),
            (first, second),
            (),
            (first.artifact_id, second.artifact_id),
        )

    duplicated = replace(
        observation.assessments[0],
        assessment_id="duplicate-assessment",
        qualification=replace(
            observation.assessments[0].qualification,
            qualification_id="duplicate-assessment-qualification",
            assessment_id="duplicate-assessment",
        ),
    )
    with pytest.raises(ExternalRealityError, match="cannot be reused"):
        replace(
            observation,
            assessments=(observation.assessments[0], duplicated),
        )


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


def test_relation_rehash_attack_fails_deterministic_replay():
    problem = _problem()
    claim = _claim()
    need = _need(claim)

    def retrieve(request):
        observation = _observation(
            request.request_id,
            claim,
            need,
            (_source(),),
        )
        return OperationResult(
            f"{request.request_id}-result",
            request.request_id,
            output={"candidate_revision": "source-bound"},
            retrieval_observation=observation,
        )

    trace = run_connected_processing(
        problem,
        _capabilities(),
        _adaptive(),
        claims=(claim,),
        evidence_needs=(need,),
        operation_callbacks={ProcessingOperation.RETRIEVE: retrieve},
    )
    tampered = deepcopy(trace.to_dict())
    assessment = tampered["retrieval_observation"]["assessments"][0]
    assessment["relation"] = ClaimRelation.CONTRADICTS.value
    assessment["qualification"]["relation"] = ClaimRelation.CONTRADICTS.value
    assessment["qualification"][
        "basis"
    ] = "normalized-falsification-discriminator"
    observation_without_hash = {
        key: value
        for key, value in tampered["retrieval_observation"].items()
        if key != "retrieval_hash"
    }
    tampered["retrieval_observation"]["retrieval_hash"] = _canonical_hash(
        observation_without_hash
    )
    tampered["result"]["retrieval_observation"] = deepcopy(
        tampered["retrieval_observation"]
    )
    result_without_hash = {
        key: value
        for key, value in tampered["result"].items()
        if key != "output_hash"
    }
    tampered["result"]["output_hash"] = _canonical_hash(result_without_hash)
    trace_without_hash = {
        key: value
        for key, value in tampered.items()
        if key != "structural_hash"
    }
    tampered["structural_hash"] = _canonical_hash(trace_without_hash)

    with pytest.raises(
        CognitionValidationError,
        match="relation qualification|independently qualified",
    ):
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