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

## Round-four bounded nervous-system decision

Round 4 adapts only the following legacy *ideas*, not legacy code or authority:

| Existing mechanism | Exact candidate adaptation | Why the source is not reused |
| --- | --- | --- |
| Emergence signal record | Canonical Kraken-R `Signal` now carries optional task-state, provenance, TTL, and tick binding for candidate propagation. | The legacy shape relies on wall-clock creation order and has incompatible ownership. |
| Legacy tick-loop intent | A new fresh, deterministic in-memory propagation run has bounded ticks, deliveries, and fan-out. | The legacy loop does not consume or enforce its declared fan-out budget and has incompatible handler signatures. |
| Legacy topic, source/priority, and dedup intent | Static named candidate topics, explicit source/cause identities, integer priority with stable tie ordering, stable content deduplication, and explicit duplicate disposition are used. | Existing topic catalogs are retained as legacy vocabulary; their old dedup window and queue ordering are not a versioned TTL/provenance contract, and unknown candidate topics now reject before delivery. |

The following mechanisms remain preserved but **quarantined** from Kraken-R
authority:

| Mechanism | Round-four disposition |
| --- | --- |
| Both legacy dispatchers and the persistent replay buffer | Not imported, wrapped, subscribed to, or persisted by the candidate slice. |
| Hebbian cascade router and all reinforcement paths | Disabled for candidate use: mutable learned weights, randomized suppression, global outcome credit, and persisted cross-run state break deterministic causal replay. |
| Pathway tracker and learning integration | Observation-only legacy salvage candidate; no candidate promotion or route selection. |
| Topology tracker and pressure field | Observation-only legacy salvage candidates; telemetry is neither a route selector nor evidence. |
| HOP, homeostasis, alternate routers, and historical queues | Preserved as legacy/alternate mechanisms and not made candidate authorities. |

The accepted Round 4 slice is `kraken_r/nervous_system.py`: static pass,
amplify, and inhibit rules over canonical declared signals. It has no adaptive
or Hebbian routing. Signal inhibition may change a candidate trajectory to
`insufficient_evidence`, but a signal cannot claim or create evidence, truth,
settlement support, or learning.