from dataclasses import replace

import pytest

from kraken_r.cognition_kernel import SemanticJob
from kraken_r.recognition_memory import (
    CognitionEpisode,
    MemoryState,
    RecognitionCue,
    RecognitionMemoryError,
    RecognitionMemorySet,
    compress_episodes,
    reactivate,
    recognize,
    validate_recognition_context,
)


def _episode(
    episode_id: str,
    *,
    problem_id: str = "produce-warehouse",
    branch_id: str = "sorting-branch",
) -> CognitionEpisode:
    return CognitionEpisode(
        episode_id,
        problem_id,
        branch_id,
        "distributed-sorting",
        2,
        SemanticJob.MECHANISM_GENERATION,
        f"candidate-{episode_id}",
        "Many weak item cues combine into a stable routing pattern.",
        "The load-bearing distinction is relational shape, not item label.",
        {
            "observations": [f"observation-{index}" for index in range(12)],
            "distinctions": [f"distinction-{index}" for index in range(12)],
            "exception_detail": "A damaged label is not evidence of a damaged item.",
        },
        ("many", "weak", "cues", "routing"),
        ("inventory", "sorting"),
        ("distributed", "many_to_one"),
        ("damaged label",),
        ("classification",),
        (f"material-{episode_id}",),
        (f"lineage-{episode_id}",),
        {"request": f"request-{episode_id}", "candidate_only": True},
    )


def _memory_set():
    episodes = (_episode("one"), _episode("two"))
    memory = compress_episodes(
        episodes,
        concept_id="distributed-sorting",
        concept_version=2,
        summary="Many weak cues can recover the same relational routing shape.",
        discriminative_features=("weak", "cues", "routing"),
        supporting_features=("sorting", "inventory"),
        structural_features=("many_to_one", "distributed"),
        exceptions=("damaged label",),
        applicability_scope=("classification",),
        capability_ids=("capability-recall",),
        state=MemoryState.DORMANT,
    )
    return RecognitionMemorySet(episodes, (memory,)), memory


def test_compression_is_smaller_and_expands_exact_original_episodes():
    memory_set, memory = _memory_set()
    assert memory.active_context_item_count < memory.detailed_item_count
    assert memory_set.expand(memory.memory_id) == memory_set.episodes
    assert memory.source_lineage_ids == ("lineage-one", "lineage-two")


def test_partial_noisy_delayed_cue_reactivates_dormant_memory():
    memory_set, memory = _memory_set()
    cue = RecognitionCue.from_text(
        "cue-partial",
        "months later: noisy fragments mention weak routing plus irrelevant bananas",
        delayed_ticks=50_000,
    )
    updated, result = reactivate(memory_set, cue)
    assert result.selected_memory_id == memory.memory_id
    assert result.projection.reactivated_memory_ids == (memory.memory_id,)
    assert updated.memories[0].state is MemoryState.ACTIVE


def test_structural_cross_domain_cue_recognizes_without_shared_domain_label():
    memory_set, memory = _memory_set()
    cue = RecognitionCue(
        "cue-cross-domain",
        ("weak", "signals"),
        structural_features=("many_to_one",),
    )
    result = recognize(memory_set, cue)
    assert result.selected_memory_id == memory.memory_id


@pytest.mark.parametrize(
    "cue",
    [
        RecognitionCue.from_text("cue-false", "damaged label weak routing"),
        RecognitionCue.from_text(
            "cue-stale",
            "weak routing cues",
            concept_id="distributed-sorting",
            concept_version=1,
        ),
        RecognitionCue.from_text(
            "cue-scope",
            "weak routing cues",
            scope=("settlement",),
        ),
    ],
)
def test_false_stale_and_overbroad_matches_fail_closed(cue):
    memory_set, _ = _memory_set()
    result = recognize(memory_set, cue)
    assert result.selected_memory_id is None
    assert result.projection is None


def test_ambiguous_matches_are_rejected():
    memory_set, memory = _memory_set()
    other = replace(memory, memory_id="memory-other", compression_hash="")
    memory_set = RecognitionMemorySet(memory_set.episodes, (memory, other))
    result = recognize(memory_set, RecognitionCue.from_text("cue", "weak routing"))
    assert result.selected_memory_id is None


def test_destructive_cross_concept_merge_is_rejected():
    with pytest.raises(RecognitionMemoryError, match="identity/version"):
        compress_episodes(
            (_episode("one"), replace(_episode("two"), concept_id="other")),
            concept_id="distributed-sorting",
            concept_version=2,
            summary="invalid",
            discriminative_features=("weak",),
        )


def test_projection_schema_and_hashes_reject_tampering():
    memory_set, _ = _memory_set()
    result = recognize(
        memory_set, RecognitionCue.from_text("cue-valid", "weak routing")
    )
    context = result.projection.to_processing_context()
    proof = result.projection.to_replay_proof()
    assert validate_recognition_context(context, proof)["candidate_only"] is True
    context["projection_id"] = "recognition-tampered"
    with pytest.raises(RecognitionMemoryError, match="identity"):
        validate_recognition_context(context, proof)