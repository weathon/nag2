# Why `angular_guidance` Uses `* 2.0` and Why Block-Level Application Needs `* 0.5`

## The `* 2.0` in `angular_guidance`

The function returns:

```python
return result * norm_c * 2.0
```

`result` is a unit vector (output of the slerp interpolation). Multiplying by `norm_c` restores the
magnitude of the original positive hidden state `h_cond`. The extra `* 2.0` doubles that magnitude
intentionally — it produces an *amplified* version of the guided direction.

This is designed for **attention-level** application (SD3/Flux NAG attn processor), where the guided
result is used *before* the residual connection and `to_out` projection.

---

## SD3 / NAG Attn Processor: Why `* 2.0` Is Safe

In `NAGJointAttnProcessor2_0` and `NAGFluxAttnProcessor2_0`, guidance is applied to the raw
**attention output** — before `to_out` and before the block residual:

```
block input:  h_in   (magnitude M)
     |
  norm + attention
     |
  attn_out             (magnitude << M, bounded by softmax + value projection)
     |
  angular_guidance(attn_out_pos, attn_out_neg)  --> 2 * |attn_out_pos| * direction
     |
  alpha blend          --> (2 * 0.25 + 0.75) * |attn_out_pos| = 1.25 * |attn_out_pos|
     |
  to_out projection    (learned linear, maps back to residual space)
     |
  residual add: h_out = h_in + to_out(guided_attn)
```

The key protection here is the **residual connection**. `h_in` has magnitude `M`. Even if the guided
attention term grows somewhat, `to_out(guided_attn)` contributes an *additive increment* to `h_in`.
The block output is dominated by `h_in`, so magnitude growth is naturally bounded each block.

Also, `attn_out` is the result of softmax-weighted sum of value vectors — its magnitude is
inherently moderate regardless of the input scale.

Over SD3's ~24 blocks, the cumulative per-block growth of `1.25x` (from the alpha blend) is
partially absorbed by the residual structure, and in practice the generation stays stable.

---

## Non-NAG Flux: Why `* 2.0` Causes Noise

In `transformer_flux.py` non-NAG mode, guidance is applied to the **full block output** — *after*
the residual connections and MLP:

```
block input:  h_in   (magnitude M)
     |
  FluxTransformerBlock / FluxSingleTransformerBlock
  (includes norm, attention, to_out, MLP, ALL residuals)
     |
  h_out               (magnitude ≈ M, residuals keep it stable)
     |
  angular_guidance(h_out[0], h_out[1])  --> 2 * M * direction
     |
  alpha blend          --> (2 * 0.25 + 0.75) * M = 1.25 * M
     |
  next block input:   1.25 * M
```

There is no residual to absorb the growth here. Each block's input magnitude is 1.25x the previous.
Over Flux's 19 double blocks + 38 single blocks = 57 total guided blocks:

```
1.25^57 ≈ 83,000x magnitude inflation
```

The AdaLayerNorm *inside* each block normalizes the input before computing attention/MLP, so the
blocks themselves still function. But the final `norm_out` and `proj_out` layers are calibrated for
normal magnitude ranges. An 83,000x inflated input to `proj_out` produces outputs far outside the
expected latent distribution — indistinguishable from random noise after the VAE decoder.

---

## The Fix: `* 0.5` at Block Level

Multiplying the `angular_guidance` output by `0.5` before blending cancels the extra `* 2.0`:

```
guided_scaled = angular_guidance(h_out[0], h_out[1]) * 0.5
             = norm_c * 1.0 * direction     (magnitude = M, not 2M)

blend = guided_scaled * nag_alpha + h_out[0] * (1 - nag_alpha)
      = M * nag_alpha * direction + M * (1 - nag_alpha) * direction_c
      ≈ M                                   (magnitude-preserving)
```

With `nag_alpha = 0.25`: `0.5 * 0.25 + 0.75 = 0.875` (slight decay) to `1.0` (exact preservation
when guided and positive directions align). This keeps the hidden state magnitude stable throughout
all 57 blocks.

---

## Summary

| Application level | Formula output | Residual protection | Per-block growth | Over 57 blocks |
|---|---|---|---|---|
| Attention (SD3/NAG attn) | `2 * norm_c` | Yes (h_in + to_out(...)) | ~1.25x | bounded |
| Block (non-NAG Flux, **before fix**) | `2 * norm_c` | No | 1.25x | **~83,000x** |
| Block (non-NAG Flux, **after fix**) | `1 * norm_c` | No | ~1.0x | **~1x** |

The `* 2.0` is not a bug in `angular_guidance` itself — it is the right choice for attention-level
use. The fix belongs at the *call site* when applying guidance at block level, where the residual
protection is absent.
