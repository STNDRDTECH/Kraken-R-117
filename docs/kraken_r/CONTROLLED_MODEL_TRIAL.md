# Round 7 controlled provider trial — provider failure record

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
- Provider calls that failed closed: `12` (failure code: `provider_error`). This record is retained as a failed injected-provider trial and does not support a comparison claim.

## Observed held-out labels

| Condition | Observed route-hint label match rate |
| --- | ---: |
| `base_model` | `0.00` |
| `kraken_mediated` | `0.00` |
| `kraken_ablated` | `0.00` |

Observed label deltas (descriptive only):
- mediated minus base: `0.00`
- mediated minus reset-routing: `0.00`

## Replay archive

The paired JSON archive contains one immutable invocation envelope and one structural replay record for every provider call. Every replay is marked `generation_replayed: false` and `structure_replayed: true`; replay validates recorded structure and hashes without issuing another provider request.

## Claim status

- `not_claimed` — Model proposals are candidate declarations; this harness has no independent execution settlement.
