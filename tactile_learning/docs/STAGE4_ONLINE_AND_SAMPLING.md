# Stage 4 — causal online inference, sampling robustness, early recognition

## 1. Online state
RGB is a point observation at `t`. Optical flow is an interval observation `[t_start,t_end]`.
A flow feature is observable online only after `t_end`. The streaming buffer enforces this rule.

## 2. Sampling-rate robustness
We do not align sequences by frame index. FPS augmentation is timestamp-based. Consistency is applied
to semantic latent representations and optionally to class logits from physically equivalent episodes.
Raw RGB/flow values are never forced to be identical.

## 3. Early recognition
At each query time the model sees only the causal prefix. An early decision is accepted only after
the class remains above the confidence threshold for a configurable patience window. This avoids
declaring a class from a single noisy transient.

## 4. Important limitation
The buffer is an interface/state primitive. It must be wired into the final CTE-HGNN forward pass so
encoders and graph memory are updated incrementally rather than recomputing the full prefix.
