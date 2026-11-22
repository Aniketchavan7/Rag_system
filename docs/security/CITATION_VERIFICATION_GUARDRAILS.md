# Security & Reliability: Citation Grounding Guardrails

## 1. Post-Generation Verification Hook
Every generated synthesis is inspected by the verification guardrail before streaming to the client:
- All `[chunk_id]` references are resolved against the active candidate set.
- Sentences making empirical claims without citations trigger automated sentence-level self-correction.
- If ungrounded statements persist, the engine safely abstains with `NEEDS_REVIEW`.
