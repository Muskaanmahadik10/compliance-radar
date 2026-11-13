# Superseded results (commit af9dd6b, 13 Nov 2026)

These files are the evaluation outputs and the sample erasure certificate from commit `af9dd6b`. They are kept here unchanged, as a record. Do not quote them as current results.

A Codex audit on 13 Nov 2026 found that some of these numbers did not measure what their names said, and that the certificate's signature proved less than it seemed to:

| File | What was wrong |
|---|---|
| `brief_eval_env.json` (11 events, real models) | `grounded_first_try` was read from the *final* verifier findings, so a first draft that was repaired on revision counted as "first try"; citations removed from inside actions or affected items were not counted at all. `impact_exact` compared supplier *counts*, so a wrong supplier set of the right size would have passed. `critic_approved` mixed the critic's opinion with automatic approval. Per-row critic fields were missing in this older file. |
| `brief_eval_env_sample3.json` (3 events, real models) | Same metric definitions as above. The "3 events" were the first three matching events, not a random sample. |
| `brief_eval_mock.json` | Same metric definitions. The mock model also wrote the length of the capped top-8 supplier list into its summary instead of the SQL total, so a summary-versus-count contradiction could not show up. |
| `erasure_CTR-S0001.json` | Signed with the public fallback key `dev-signing-key-change-me`, which is in the repository, so anyone can re-sign an altered copy. Also produced before erasure tombstones existed: the document's text was still in `data/generated/documents.json`, and an ordinary re-index would have restored it. |

What these files do still show: in the 11-event real-model run, the supplier counts matched the hand-written truth on 11 of 11 events, the governing policy was cited on 11 of 11, every action carried a citation on 11 of 11, and the critic approved 7 of 11. Those are observations from that run under those definitions.

What changed after the audit (see the main README):

- Briefs now store per-attempt verifier records, so the first-try metric uses attempt 0, and every removed citation is counted.
- Impact is compared as supplier id sets, using the full id list now stored with each brief.
- The verifier also checks product ids, record names, policy-entry citations and explicit totals in the summary. Structural failures now block automatic approval even when the critic approves.
- Erasure writes a tombstone that every ingest and rebuild path respects, and removes the local source copies. Certificates name their signing key, and a demo-key signature is labelled as integrity-only.

Current results are in `../brief_eval_*.json` (rerun with the corrected definitions) and in the README.
