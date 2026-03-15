from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention
from ftfy import apply_plan


class NAGWanAttnProcessor2_0:
    def __init__(self, nag_scale=1.0, nag_tau=2.5, nag_alpha=0.25):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("WanAttnProcessor2_0 requires PyTorch 2.0. To use it, please upgrade PyTorch to 2.0.")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        apply_guidance = self.nag_scale > 1 and encoder_hidden_states is not None
        if apply_guidance:
            if len(encoder_hidden_states) == 2 * len(hidden_states):
                batch_size = len(hidden_states)
            else:
                apply_guidance = False

        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            encoder_hidden_states_img = encoder_hidden_states[:, :257]
            encoder_hidden_states = encoder_hidden_states[:, 257:]
            if apply_guidance:
                encoder_hidden_states_img = encoder_hidden_states_img[:batch_size]
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states

        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        key = key.unflatten(2, (attn.heads, -1)).transpose(1, 2)
        value = value.unflatten(2, (attn.heads, -1)).transpose(1, 2)

        if rotary_emb is not None:

            def apply_rotary_emb(hidden_states: torch.Tensor, freqs: torch.Tensor):
                x_rotated = torch.view_as_complex(hidden_states.to(torch.float64).unflatten(3, (-1, 2)))
                x_out = torch.view_as_real(x_rotated * freqs).flatten(3, 4)
                return x_out.type_as(hidden_states)

            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img = attn.add_k_proj(encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)
            value_img = attn.add_v_proj(encoder_hidden_states_img)

            key_img = key_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)
            value_img = value_img.unflatten(2, (attn.heads, -1)).transpose(1, 2)

            hidden_states_img = F.scaled_dot_product_attention(
                query, key_img, value_img, attn_mask=None, dropout_p=0.0, is_causal=False
            )
            hidden_states_img = hidden_states_img.transpose(1, 2).flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        if apply_guidance:
            key, key_negative = torch.chunk(key, 2, dim=0)
            value, value_negative = torch.chunk(value, 2, dim=0)
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        if apply_guidance:
            hidden_states_negative = F.scaled_dot_product_attention(
                query, key_negative, value_negative, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
            hidden_states_negative = hidden_states_negative.transpose(1, 2).flatten(2, 3)
            hidden_states_negative = hidden_states_negative.type_as(query)

            hidden_states_positive = hidden_states

            hidden_states_guidance = hidden_states_positive * self.nag_scale - hidden_states_negative * (self.nag_scale - 1)
            norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_positive.shape)
            norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_guidance.shape)

            scale = norm_guidance / norm_positive
            scale = torch.nan_to_num(scale, 10)
            hidden_states_guidance[scale > self.nag_tau] = \
                hidden_states_guidance[scale > self.nag_tau] / (norm_guidance[scale > self.nag_tau] + 1e-7) * norm_positive[scale > self.nag_tau] * self.nag_tau

            hidden_states = hidden_states_guidance * self.nag_alpha + hidden_states_positive * (1 - self.nag_alpha)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states
