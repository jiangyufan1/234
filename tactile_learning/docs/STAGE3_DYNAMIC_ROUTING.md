# Stage 3 — Dynamic routing

The network uses two modality-specific pseudo-memory nodes. RGB and Flow are first aggregated within modality; the memories exchange information and dispatch it back. The routing is causal because only graph nodes that are available at the query time can enter the memory.

Physical rule: RGB is a point observation; Flow is an interval observation. A Flow interval is not available online until its end timestamp.
