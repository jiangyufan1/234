# CTE-HGNN Stage 1-3 source bundle (experimental)

This bundle recovers the previously generated Stage 1-3 source files into one consistent `tactile_learning/` layout.

Contents:
- Stage 1: causal trend/event decomposer, tests, design notes.
- Stage 2: point-to-interval temporal graph builder.
- Stage 3: heterogeneous model skeleton and dynamic pseudo-node router.

Important:
- This is a recovered source bundle, NOT a verified integrated training release.
- Stage 3's `temporal_graph.py` is used as the newer version because Stage 3 also revised that module. The older standalone Stage 2 copy is intentionally not duplicated.
- `cte_hgnn.py` currently expects latent RGB/Flow tensors and is an integration skeleton. It does not by itself adapt the existing repository dataset/encoder/training interfaces.
- Run tests only after copying into a compatible project and resolving imports/configuration. Do not overwrite the existing baseline.
