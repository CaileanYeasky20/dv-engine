# dv-engine
Verifiable, privacy-preserving compute gateway running on ordinary CPUs. No secure enclaves, no FHE.

## What this proves
The validation suite runs end to end on CPU with the numpy backend: 1,000 layers folded with Byzantine submissions rejected, CRDT merge across 4 replicas, concurrent replay blocking, and compression with tamper detection. Results are reproducible from commit 3919c1f.

## What this does not prove
No one outside the author has reviewed this code or run it on production data. The compression ratio is measured on synthetic payloads and does not generalise to arbitrary data. The M31 matrix-multiplication relation has one unfinished sorry on the Freivalds soundness argument.

## Status
Pre-launch. Single author. Seeking technical review.
