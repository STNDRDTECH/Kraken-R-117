---
name: Kraken-R long-horizon soak test design and cross-file bound relationships
description: How the 10,000-cycle Kraken-R longevity soak is structured, and non-obvious numeric relationships between dynamical_substrate/adaptive_substrate/plastic_routing bounds discovered while building it.
---

## Soak test location and structure

`tests/test_kraken_r_longevity_soak.py`. A single `DynamicalState` lineage
cannot exceed `MAX_TICKS=256` (enforced at construction), so the bulk
10,000-cycle soak chains 40 independent fresh sessions x 250 ticks rather than
raising any production bound. Full run: ~20s (bulk soak ~2s, grounded-execution
tests ~18s for ~35 real pytest subprocess calls total). Uses a fixed seed
(`random.Random(20260826)`) for determinism — no wall-clock, no crypto
randomness.

## Cross-file numeric relationships (non-obvious, worth re-deriving carefully)

- `MAX_EVENT_COUNTER (4096) == MAX_TICKS (256) * MAX_TICK_EVENTS (16)` exactly.
  `historical_event_ids` never evicts within a lineage, so a maximally busy
  256-tick lineage fills it exactly as it exhausts its own tick budget —
  never overflows, but also never partial by design at max load.
- Adaptive substrate: `generation` (structural field, capped at
  `MAX_ADAPTIVE_GENERATIONS=8`) and `updates_applied` (per-generation budget,
  capped at `MAX_ADAPTIVE_UPDATES=8`) are independent counters. Ordinary
  credit (`apply_grounded_adaptation`/`form_grounded_connection`/etc.) is
  gated only by `updates_applied`, NOT by `generation`. Only
  `rollback_adaptive_state` and `invalidate_grounded_adaptation` explicitly
  check `generation >= MAX_ADAPTIVE_GENERATIONS`. Net effect: once a lineage
  reaches generation 8, ordinary bounded credit still works, but rollback and
  invalidation become permanently unavailable for that lineage — a one-way
  door for undo, not for edits.
- `RouteTopology.reset()` rebuilds fully fresh `CandidateRoute` objects
  (only `route_id`/`context_id`/`source`/`target` carried over), so it clears
  weight AND success_count/failure_count back to defaults, not just
  `applied_settlement_ids`. `CandidateRoute.success_count`/`failure_count`
  themselves have no declared ceiling at the constructor level (unlike weight
  or any audit tuple) — this is the already-filed "route-learning replay
  history grows without limit" gap, confirmed structurally.
- `MAX_CONNECTIONS=8` in the adaptive substrate is structurally unreachable
  with the default 2-route fixture topology: connection identity is derived
  from `(route.source, route.target)`, so the default fixture can produce at
  most 1-2 unique connection ids. Forcing this boundary would need a custom
  multi-context topology, not attempted in the first soak pass (judged out of
  proportion for a first hardening pass).

## Discovered gap: capacity-exhaustion errors crash `reduce_dynamical_tick`

`kraken_r/dynamical_substrate.py`'s `_is_current_lineage_error` classifies
staleness/duplication messages (stale binding, already applied/consumed/
invalidated, checkpoint not retained) into graceful `WithheldEvent`s during
tick reduction. It does NOT recognize the sibling "budget exhausted"/"limit is
exhausted" messages from the same adaptive/plastic-routing layer — e.g.
"topology settlement budget is exhausted", "adaptive generation update budget
is exhausted", "rollback budget is exhausted", "adaptive generation limit is
exhausted", "adaptive record identity retention is exhausted", "invalidation
budget is exhausted". When one of these fires while processing a real event
through `reduce_dynamical_tick`, a fully legitimate, correctly authorized,
context-matched grounded settlement crashes the tick reduction with an
uncaught exception instead of being withheld like every sibling boundary
condition. Confirmed via direct reproduction (pre-fill
`route_topology.applied_settlement_ids` to `MAX_AUDIT_LINKS`, then feed one
real grounded task-success settlement through `reduce_dynamical_tick`) — see
`test_topology_settlement_budget_exhaustion_crashes_the_dynamical_reducer`,
which pins this as a known, not-yet-fixed regression baseline. Filed as a
follow-up task rather than fixed inline, consistent with this project's
established pattern of filing audit findings as separate reviewable tasks.

## Operational hazard: do not touch the `orzhaal_bubble_*` git stash stack

This repo accumulates hundreds of auto-generated stash entries named
`orzhaal_bubble_orzhaal_<...>` (likely an automated checkpoint/bubble
mechanism). A bare `git stash` / `git stash pop` used for scratch diffing can
silently pop one of these unrelated system stashes and create merge conflicts
in runtime state files (e.g. `.rogal/*.json`, `data/*.json`). If you need a
clean-tree comparison, use `git stash push -- <specific paths>` or a
throwaway worktree/branch instead of a bare stash — never bare `git stash` in
this repo. Recovery: `git reset --hard HEAD` restores tracked files (does not
touch untracked files or the stash list); leave the pre-existing stash
entries untouched.
