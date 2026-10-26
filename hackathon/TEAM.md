# Deadline triage — parallel work, no model load

| Priority | Owner to assign | Task | Done means |
|---|---|---|---|
| P0 | Submission lead | Confirm deadline/rules, fill entry from `MARKETING.md` | Entry saved with team names and actual commit URL. |
| P0 | Demo lead | Assemble 90-second recording from existing clips and graphs | Playable video with provenance labels and limitations. No fabricated live generation. |
| P0 | Comfy integrator | Review API graphs + required node/model/assets | Mapping checklist complete; no queue call until resource approval. |
| P0 | Package lead | Run offline check and make source archive | Verifier PASS; archive + checksum shared with team. |
| P1 | Evidence lead | Verify rights and input provenance of each chosen clip | A small clip list marking source/control/H3 and legacy geometry. |
| Later, gated | Model operator | Short Ref2VA fidelity test | New tagged result reviewed for look AND contacts; clocks checked. |

## First 15 minutes
1. Checkout `team/hackathon-handoff`; start at `hackathon/README.md`.
2. Run `python3 scripts/verify_hackathon.py` (stdlib only).
3. Assign the four P0 owners; use existing media rather than regenerate.
4. Agree on one truthful demo story: explicit motion source → H3 conditioning → measured failures and next test.
5. Copy API graphs into the team's Comfy workflow review; map assets under `input/band_aid/` without loading a model.

## Later inference handoff
Use `reference_match/test_plan.json`. Preserve the standard-hand-input constraint. Prior runtime handoff said the backend was stopped to return memory; do not assume current service state. Never start it, restart unrelated services, lower memory guards or create `suite4.go` without approval. No full-suite run before the short shot passes both objectives.
