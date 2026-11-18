# AGENTS.md

# Project: ST-CTE-HGNN

## 1. Project Objective

Develop a spatiotemporal tactile object recognition system
based on the existing CTE-HGNN architecture.

The project focuses on:
- RGB temporal feature extraction.
- Optical-flow interval feature extraction.
- Trend-event decomposition.
- RGB-Flow heterogeneous temporal graphs.
- Bidirectional heterogeneous message passing.
- Dynamic pseudo-memory routing.
- Three-finger spatial relationship modeling.
- Spatiotemporal feature fusion.
- Online recognition and sampling-rate robustness.

## 2. Core Development Principles

1. Preserve the existing CTE-HGNN temporal branch.
2. Develop spatial modules independently before integration.
3. Do not rewrite existing modules without a demonstrated need.
4. Every new module must have unit tests.
5. Preserve reproducibility and backward compatibility.
6. Do not claim experimental improvements without measured results.

## 3. Temporal Physical Constraints

RGB represents point observations:
    RGB(t)

Optical flow represents interval observations:
    Flow([t_start, t_end])

A flow interval is available online only after t_end.

Requirements:
- Never align RGB and Flow solely by frame index.
- Preserve actual timestamps.
- Preserve different sequence lengths.
- Do not expose unfinished flow intervals.
- Prevent future information leakage.
- Padding must be controlled by explicit masks.
- Do not interpret optical-flow displacement as physical velocity
  without accounting for the interval duration.

## 4. Spatial Modeling Principles

The spatial branch must model physically meaningful relationships.

Potential inputs:
- Three-finger sensor positions.
- Relative sensor orientations.
- Local contact regions.
- Contact area distributions.
- Local deformation features.
- Spatial changes over time.

Requirements:
- Preserve individual finger identities.
- Define physical meaning for every node and edge.
- Use a consistent coordinate frame.
- Distinguish observable inputs from simulation-only ground truth.
- Do not directly feed complete object CAD meshes into the tactile
  classifier unless explicitly designing a separate CAD-assisted baseline.
- Object pose and simulation contact forces may be used for supervision
  and analysis when clearly separated from inference inputs.

## 5. Model Architecture

Existing temporal modules must remain available:

- trend_event.py
- temporal_graph.py
- dynamic_routing.py
- cte_hgnn.py
- cte_encoders.py
- cte_model.py

New spatial modules are developed independently:

- spatial_encoder.py
- spatial_graph.py
- spatiotemporal_fusion.py

The final model must preserve the ability to evaluate:
- Temporal-only baseline.
- Spatial-only baseline.
- Temporal + spatial fusion.
- Individual module ablations.

## 6. Dataset Organization

Project data root:

    data/

Expected organization:

    data/
      raw/
        MCB_A/
      collected/
      processed/
      derived/
      indexes/
      splits/

Rules:
- Never upload raw datasets to GitHub.
- Never modify original MCB_A files in place.
- Keep raw and derived data separate.
- Record collection parameters and timestamps.
- Preserve object category and instance identifiers.
- Keep all frequency variants from one physical episode
  within the same dataset split.
- Do not silently regenerate existing experiment splits.

## 7. Collection and Simulation

Collection scripts belong in:

    scripts/collection/

Data processing scripts belong in:

    scripts/data/

Requirements:
- Preserve deterministic random seeds.
- Record simulation configuration.
- Record object identity and category.
- Record three-finger contact states.
- Record RGB timestamps and flow interval boundaries.
- Separate simulation ground truth from model inputs.
- Provide data-quality auditing.

## 8. Configuration

Experiment configurations belong in:

    configs/experiments/

Every formal experiment should record:
- Random seed.
- Dataset version.
- Dataset split.
- Model variant.
- Feature dimensions.
- Sampling configuration.
- Optimizer settings.
- Auxiliary loss weights.
- Hardware and software versions.

Do not hardcode machine-specific absolute paths in shared configs.

## 9. Testing Requirements

Run relevant tests after every modification.

General command:

    python -m pytest -v

Additional requirements:
- Test different RGB and Flow sequence lengths.
- Test causal prefix invariance.
- Test missing modalities.
- Test three-finger independence.
- Test gradient propagation.
- Test spatial coordinate consistency.
- Test temporal and spatial fusion.
- Test baseline compatibility.

Synthetic tests validate software behavior, not recognition performance.

## 10. Git Rules

- Never commit raw datasets.
- Never commit model checkpoints.
- Never commit generated training caches.
- Commit source code, tests, documentation and experiment configs.
- Never use git reset --hard or force-push without explicit approval.
- Preserve existing branches and commits.
- Review git diff before staging.
- Do not commit automatically.

## 11. Development Environment

Primary development:
- Windows 11
- Python 3.12
- PyTorch
- CUDA

Secondary environment:
- Ubuntu 22.04
- ROS 2 Humble where robotics integration is required.

Use standard PyTorch where practical.
Do not introduce unnecessary dependencies.

## 12. Scientific Integrity

Do not confuse:
- Contact force with measured tactile shear force.
- Optical-flow displacement with physical velocity.
- Simulation ground truth with deployable sensor input.
- Unit-test success with real-world recognition performance.
- More model parameters with scientific innovation.

All claimed improvements must be supported by controlled experiments.
