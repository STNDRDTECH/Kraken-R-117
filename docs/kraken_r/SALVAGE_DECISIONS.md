# Round-Three Salvage Decisions

**Status:** preserved evidence and quarantine record  
**Scope:** candidate-only Kraken-R; no legacy mechanism is changed by this record.

## Deleted refinement traces

The eleven deleted `.rogal/refinement_traces/cycle_*_refinement.json` artifacts
are intentionally **not restored** to the working tree. They were runtime
artifacts, are retained in Git history, and contained repeated zero-primitive
rejections rather than unique committed patches or evaluation results. Restoring
them would violate the repository policy that runtime state is not
developer-authored source.

If a compliance review needs the raw records, retrieve the set from Git history
as one archive with its deletion context. Do not reintroduce it as live runtime
state or source input.

## Preserved nervous-system salvage candidates

The following legacy mechanisms are preserved in the architecture registry but
are not candidate authorities and are not connected to `kraken_r`.

| Mechanism | Disposition | Recorded blocker |
| --- | --- | --- |
| Root and emergence EventBus implementations | preserve / quarantine alternate | Duplicate ownership and incompatible signal, callback, queue, and replay semantics. |
| TickLoop and persistent event replay buffer | quarantine from candidate use | No proven live constitutional caller; replay loses identity/provenance and is not evidence settlement. |
| Hebbian cascade router | salvage only | No proven live route caller or default edges; source-wide credit assignment, swallowed persistence failures, and the graduated-reward reinforcement call has an argument mismatch. |
| Pathway tracker and dormant learning integration | observation only | Separate mutable learning state; prefix-count inflation, raw-success consolidation, and no constitutional settlement gate. |
| Topology tracker | observation only | Instantiated but lacks live interaction/snapshot callers; topology is not routing authority. |
| Pressure field | observation only | Live construction lacks propagation wiring; pressure signals are not evidence or authority. |
| Task router, tool router, and lattice bus | preserve / quarantine alternates | Separate routing/event authorities with no contract-compatible provenance or singular ownership. |
| HOP and homeostasis variants | preserve legacy / quarantine alternates | Operational helpers are not candidate lifecycle authority; duplicate or alternate ownership remains unresolved. |

## Candidate boundary

Round-three replay reuses only Kraken-R contracts and the existing
constitutional trace. It does not subscribe to either EventBus, invoke a
router, read a replay buffer, write persistence, use cascade/pathway/topology
state, or call homeostasis/HOP machinery. Any future adaptation must first
preserve event identity, task-state version, provenance, observed evidence,
settlement linkage, deterministic ordering, and explicit stopping without
creating a second authority.