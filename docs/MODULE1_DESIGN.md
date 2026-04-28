# Module 1 Design (v0.2)

This note summarizes the current Module 1 design after the v0.2 hardening pass.

## 1) Seven-gate screening

- **G1** mechanism_plausibility (refinable, LLM)
- **G2** scale_alignment (hard blocker, rule engine)
- **G3** data_availability (hard blocker, rule engine)
- **G4** identification_validity (refinable, rule+LLM)
- **G5** novelty (refinable, OpenAlex+LLM)
- **G6** automation_feasibility (hard blocker, rule engine)
- **G7** contribution_clarity (refinable, LLM)

Decision priority:

1. Any hard blocker fail -> `REJECTED`
2. All refinable pass -> `ACCEPTED`
3. >=3 refinable fail -> `TENTATIVE`
4. Otherwise -> refine until `max_rounds`, then `TENTATIVE`

## 2) Reflection loop state machine

Per topic:

1. Run hard blockers (G2/G3/G6)
2. Gather novelty evidence from OpenAlex
3. Score refinable gates (batch critique preferred, per-gate fallback)
4. Decide status (`ACCEPTED`, `REFINE`, `TENTATIVE`, `REJECTED`)
5. If `REFINE`, apply up to 2 refine operations and iterate

Additional guards:

- Signature oscillation window (`max_signature_repeats`) -> `TENTATIVE`
- Early-stop by small score delta after min rounds -> `TENTATIVE`
- Budget guardrails (run/topic/token caps)

## 3) Four-tuple signature

`Topic.four_tuple_signature()` hashes:

- exposure family
- outcome family
- geography
- identification method

Used for:

- oscillation detection
- novelty cache keying

## 4) TENTATIVE review path

TENTATIVE topics are persisted to `output/tentative_pool.json` with:

- failed gates
- topic snapshot
- legacy compatibility gate map

Operator options:

- **Promote**: move to rank-1 candidate
- **Kill**: move to graveyard/memory as rejected
- **Re-run**: one more reflection round

## 5) Prompt/schema alignment contract

Prompt enum values are generated from `models/topic_schema.py` via:

```bash
python3 scripts/generate_prompt_enums.py
```

Prompts consume `{enum_block}` to avoid enum drift.

## 6) Legacy compatibility

- `--legacy-ideation` and `LEGACY_IDEATION=1` are preserved.
- `legacy_six_gates` now follows the v0.2 key schema:
  - `impact`
  - `quantitative`
  - `novelty`
  - `publishability`
  - `automation`
  - `data_availability`

