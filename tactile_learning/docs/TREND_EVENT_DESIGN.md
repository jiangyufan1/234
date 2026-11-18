# Stage 1: causal continuous-time trend/event decomposition

This module is the first-stage component of CTE-HGNN. It operates on latent
features from **one modality at a time**; RGB/Flow fusion is deliberately left
to the later heterogeneous graph stage.

## Physical interpretation

The decomposition is:

\[
X_m(t) = T_m(t) + R_m(t), \qquad E_m(t)=G(R_m(\le t)).
\]

- `T`: slowly varying latent contact/shape state.
- `R`: instantaneous deviation from that state.
- `E`: sparse learned transient descriptor of the residual.

**Important:** `E` is a learned dynamic feature, not automatically "slip",
"impact", or any other physical event. Those labels require independent
simulation/sensor evidence.

### Trend

We use the continuous-time first-order state model

\[
\tau \dot T = X-T
\]

whose variable-step solution is

\[
T_i=e^{-\Delta t_i/\tau}T_{i-1}
+(1-e^{-\Delta t_i/\tau})X_i.
\]

Therefore a 15 Hz and 60 Hz sequence with the same physical elapsed time do
not receive different trend updates merely because the frame count differs.

The learned time constant is constrained to a configurable physical range
(0.02–0.50 s by default), preventing the trend branch from collapsing into an
instantaneous copy of the input.

### Event representation

The residual is **exactly** `R = X - T`. A causal Gabor bank integrates this
residual over completed observation intervals:

\[
g_k(\ell)=e^{-\ell^2/(2\sigma_k^2)}
\cos(\omega_k\ell+\phi_k), \quad \ell\ge0.
\]

The interval contribution uses its actual `dt` and the trapezoidal residual.
Only intervals whose right endpoint has arrived are included. Thus changing
future frames cannot alter an earlier event representation.

Frequencies are also capped by 0.45 times the effective sampling Nyquist
frequency of the current batch. This does not solve arbitrary aliasing, but it
prevents the learned event bank from intentionally operating above the
available temporal bandwidth.

### RGB and optical-flow semantics

RGB is a point observation at `t_i`.

A flow field computed from images at `(t_i,t_{i+1})` is an interval observation;
when mapped to a common time axis its natural timestamp is the midpoint

\[
t_i^{flow}=(t_i+t_{i+1})/2.
\]

The flow branch should therefore be aligned using timestamps/interval overlap,
not simply `flow[k] <-> rgb[k]` by array index.

If flow is normalized pixel displacement, dividing by `dt` gives image-plane
displacement rate. It is **not** a calibrated physical surface velocity without
camera/sensor geometry.

## Auxiliary losses

The module returns:

- `trend_smoothness`: limits excessive trend derivative per second.
- `event_sparsity`: keeps event gates sparse.
- `decomposition_reconstruction`: encourages trend + decoded event to preserve
  the input latent information.

Use the sum once:

```python
aux = CausalTrendEventDecomposer.total_auxiliary_loss(output)
loss = classification_loss + aux
```

The weights are already applied inside the returned loss dictionary.

## Causality status

The reference implementation is causally correct but currently recomputes the
short sequence for batched training. It is **not yet the optimized persistent
streaming API** for deployment. That belongs to Stage 4.

For the current ~32-frame episodes, the reference implementation is acceptable
for development and ablation.
