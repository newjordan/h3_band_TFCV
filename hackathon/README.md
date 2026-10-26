# Band Aid — team handoff

**Timing-first control for MiniMax H3 music videos.** Source-only hackathon kit; no models or services start from opening this repository.

## Start here
1. [What works / what needs work](STATUS.md) — read before making claims.
2. [Install and package](PACKAGE.md) — Comfy custom nodes and source archive.
3. [Comfy workflows](../examples/workflows/hackathon/README.md) — portable API graphs, draft canvas layouts, assets and runtime gates.
4. [Submission copy, social posts and demo script](MARKETING.md).
5. [Team task board](TEAM.md) — parallel, time-boxed finish plan.
6. [Evidence](evidence/piano_recovery_summary.json) and [parked reference test](reference_match/test_plan.json).

## Safe offline verification
```sh
python3 scripts/verify_hackathon.py
```
Standard library only; bounded file reads. No torch/numpy import, tensor allocation, downloads, installs, model loads, server start or queue calls. Small normal Python memory use is unavoidable; this is not literally zero RAM.

## Scope of this branch
The handoff includes docs, graphs and packaging tooling. It intentionally does **not** adopt unrelated local dirty experiment code. The published commit is the source of truth. Historical media in the repo is not evidence that the new Ref2VA graph works.

**Ready:** collaborators can review the architecture, install source later, edit API graphs, prepare submission material, and run offline validation now.
**Not ready:** claiming exact piano execution, jointly correct skeleton appearance/contact, or a qualified Ref2VA / accelerated production workflow.
