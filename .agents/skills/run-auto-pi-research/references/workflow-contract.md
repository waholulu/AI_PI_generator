# Auto-PI Workflow Contract

## SearchPlan

Use this JSON shape for `search_plan`, `novelty_search_plan`, and `literature_search_plan`:

```json
{
  "purpose": "Initial field scan for built environment and health outcomes",
  "queries": [
    "built environment health outcomes census tract greenspace walkability",
    "street network walkability physical inactivity census tract"
  ],
  "per_query_limit": 10,
  "final_limit": 25,
  "notes": "Prefer public-health and urban-planning literature."
}
```

Keep queries specific enough for OpenAlex/arXiv retrieval. Do not include private data, paid APIs, or unsupported templates.

## CandidateSelection

Use one `candidate_id` from `output/topic_screening.json`:

```json
{
  "candidate_id": "example_candidate_id",
  "rationale": "Best balance of data availability, tract-level specificity, and clear identification."
}
```

## NoveltyAssessment

Use after reading `output/novelty_search_results.json`:

```json
{
  "verdict": "partially_overlapping",
  "confidence": "medium",
  "summary": "Prior studies cover walkability and physical activity, but the selected tract-level source/method combination is not directly duplicated.",
  "similar_papers": [
    {
      "title": "Example related paper",
      "citation_key": "",
      "overlap": "medium",
      "reason": "Similar exposure and outcome, different geography and method."
    }
  ],
  "recommended_action": "proceed"
}
```

Allowed verdicts are `novel`, `partially_overlapping`, `already_published`, and `unavailable`.

## NoveltyDecision

Use only when the engine asks for it:

```json
{
  "action": "proceed",
  "rationale": "Overlap is acceptable for a reproducible small-area implementation."
}
```

Allowed actions are `proceed`, `reselect`, and `abort`.

## Research Memo

The memo must be Markdown and contain these exact second-level headings:

- `## Research Question`
- `## Contribution`
- `## Data`
- `## Identification Strategy`
- `## Related Literature`
- `## Empirical Plan`
- `## Risks and Limitations`
- `## Execution Checklist`

Use citation keys from `output/references.bib` in `@citation_key` form.
