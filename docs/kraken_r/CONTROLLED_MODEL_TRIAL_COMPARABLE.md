# Round 7 controlled provider trial

## Scope and authority boundary

This is a descriptive held-out label observation, not a performance evaluation. No model output became evidence, execution, settlement, learning credit, a goal, persistence, or runtime authority. The trial runner is isolated from the ROGAL daemon and controller.

## Injected configuration

- Provider identity: `replit_openai_compatible`
- Model identity: `gpt-4o-mini`
- Prompt template: `round-7-eval-v1`
- Output-token budget: `128`
- Temperature: `0.0`
- Held-out tasks: `4`

## Control results

- All enforced provider, configuration, prompt-length, output-budget, envelope, and structural-replay controls passed.
- Provider calls that failed closed: `2` (failure codes: `malformed_output`).
- `routing-heldout-brief` provider input-token control: `available_and_equal` (reported values: `[164, 164, 164]`)
- `routing-heldout-inventory` provider input-token control: `available_and_equal` (reported values: `[163, 163, 163]`)
- `routing-heldout-risk` provider input-token control: `available_and_equal` (reported values: `[164, 164, 164]`)
- `routing-heldout-summary` provider input-token control: `available_and_equal` (reported values: `[163, 163, 163]`)

## Observed held-out labels

| Condition | Observed route-hint label match rate |
| --- | ---: |
| `base_model` | `0.00` |
| `kraken_mediated` | `1.00` |
| `kraken_ablated` | `0.00` |

Observed label deltas (descriptive only):
- mediated minus base: `1.00`
- mediated minus reset-routing: `1.00`

## Replay archive

The paired JSON archive contains one immutable invocation envelope and one structural replay record for every provider call. Every replay is marked `generation_replayed: false` and `structure_replayed: true`; replay validates recorded structure and hashes without issuing another provider request.

## Claim status

- `not_claimed` — Model proposals are candidate declarations; this harness has no independent execution settlement.
