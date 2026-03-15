from typing import Optional

import torch
import torch.nn.functional as F

from diffusers.models.attention_processor import Attention

def angular_guidance(h_cond, h_uncond, alpha, s1=1.0, s2=1.2):
    h_c = h_cond
    h_u = h_uncond
    
    norm_c = h_c.norm(dim=-1, keepdim=True)
    a = h_c / (norm_c + 1e-8)
    b = h_u / (h_u.norm(dim=-1, keepdim=True) + 1e-8)
    
    cos_theta = (a * b).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = cos_theta.acos()
    
    sin_t = theta.sin().clamp(min=1e-8)
    result = (torch.sin((1 + alpha * s1) * theta) / sin_t) * a - \
             (torch.sin(alpha * s2 * theta) / sin_t) * b
    
    result = torch.where(theta < 1e-5, a, result)
    
    return result * norm_c * 2.0

class NAGJointAttnProcessor2_0:
    def __init__(self, nag_scale: float = 1.0, nag_tau: float = 2.5, nag_alpha:float = 0.125, guidance_type: str = "angular"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        # print("Injected")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.guidance_type = guidance_type

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        batch_size = hidden_states.shape[0]

        apply_guidance = self.nag_scale > 1 and encoder_hidden_states is not None
        if apply_guidance:
            origin_batch_size = len(encoder_hidden_states) - batch_size
            assert len(encoder_hidden_states) / origin_batch_size in [2, 3, 4]

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        if apply_guidance:
            batch_size += origin_batch_size
            if batch_size == 2 * origin_batch_size:
                query = query.tile(2, 1, 1, 1)
                key = key.tile(2, 1, 1, 1)
                value = value.tile(2, 1, 1, 1)
            else:
                query = torch.cat([query, query[origin_batch_size:2 * origin_batch_size]], dim=0)
                key = torch.cat([key, key[origin_batch_size:2 * origin_batch_size]], dim=0)
                value = torch.cat([value, value[origin_batch_size:2 * origin_batch_size]], dim=0)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # Split the attention outputs.
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        if apply_guidance:
            hidden_states_negative = hidden_states[-origin_batch_size:]
            if batch_size == 2 * origin_batch_size:
                hidden_states_positive = hidden_states[:origin_batch_size]
            else:
                hidden_states_positive = hidden_states[origin_batch_size:2 * origin_batch_size]
            if self.nag_scale > 1:
                if self.guidance_type == "cfg_norm":
                        hidden_states_guidance = hidden_states_positive * self.nag_scale - hidden_states_negative * (self.nag_scale - 1)
                        norm_positive = torch.norm(hidden_states_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_positive.shape)
                        norm_guidance = torch.norm(hidden_states_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_guidance.shape)
                        scale = norm_guidance / (norm_positive + 1e-7)
                        hidden_states_guidance = hidden_states_guidance * torch.minimum(scale, scale.new_ones(1) * self.nag_tau) / (scale + 1e-7)

                        hidden_states_guidance = hidden_states_guidance * self.nag_alpha + hidden_states_positive * (1 - self.nag_alpha)

                elif self.guidance_type == "angular":
                    # hidden_states_guidance = angular_guidance(hidden_states_positive, hidden_states_negative, self.nag_scale)
                    # hidden_states_guidance = hidden_states_guidance * self.nag_alpha + hidden_states_positive * (1 - self.nag_alpha)
                    hidden_states_guidance = torch.cat((hidden_states_positive, hidden_states_negative), dim=0)
                else: 
                    raise ValueError(f"Unsupported guidance type: {self.guidance_type}")
            # scale = norm_guidance / (norm_positive + 1e-7)
            # hidden_states_guidance = hidden_states_guidance * torch.minimum(scale, scale.new_ones(1) * self.nag_tau) / (scale + 1e-7)

            # hidden_states_guidance = hidden_states_guidance * self.nag_alpha + hidden_states_positive * (1 - self.nag_alpha)

            if batch_size == 2 * origin_batch_size:
                hidden_states = hidden_states_guidance
            elif batch_size == 3 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance), dim=0)
            elif batch_size == 4 * origin_batch_size:
                hidden_states = torch.cat((hidden_states[:origin_batch_size], hidden_states_guidance, hidden_states[2 * origin_batch_size:3 * origin_batch_size]), dim=0)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states


class NAGPAGCFGJointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""
    def __init__(self, nag_scale: float = 1.0, nag_tau: float = 2.5, nag_alpha:float = 0.125, guidance_type: str = "angular"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.nag_scale = nag_scale
        self.nag_tau = nag_tau
        self.nag_alpha = nag_alpha
        self.guidance_type = guidance_type

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        context_input_ndim = encoder_hidden_states.ndim
        if context_input_ndim == 4:
            batch_size, channel, height, width = encoder_hidden_states.shape
            encoder_hidden_states = encoder_hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        identity_block_size = hidden_states.shape[
            1
        ]  # patch embeddings width * height (correspond to self-attention map width or height)

        # chunk
        hidden_states_uncond, hidden_states_org, hidden_states_ptb = hidden_states.chunk(3)
        hidden_states_org = torch.cat([hidden_states_uncond, hidden_states_org])

        (
            encoder_hidden_states_uncond,
            encoder_hidden_states_org,
            encoder_hidden_states_ptb,
            encoder_hidden_states_nag,
        ) = encoder_hidden_states.chunk(4)
        encoder_hidden_states_org = torch.cat([encoder_hidden_states_uncond, encoder_hidden_states_org, encoder_hidden_states_nag])

        ################## original path ##################
        batch_size = encoder_hidden_states_org.shape[0]
        origin_batch_size = batch_size // 3

        # `sample` projections.
        query_org = attn.to_q(hidden_states_org)
        key_org = attn.to_k(hidden_states_org)
        value_org = attn.to_v(hidden_states_org)

        query_org = torch.cat([query_org, query_org[-origin_batch_size:]], dim=0)
        key_org = torch.cat([key_org, key_org[-origin_batch_size:]], dim=0)
        value_org = torch.cat([value_org, value_org[-origin_batch_size:]], dim=0)

        # `context` projections.
        encoder_hidden_states_org_query_proj = attn.add_q_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_key_proj = attn.add_k_proj(encoder_hidden_states_org)
        encoder_hidden_states_org_value_proj = attn.add_v_proj(encoder_hidden_states_org)

        # attention
        query_org = torch.cat([query_org, encoder_hidden_states_org_query_proj], dim=1)
        key_org = torch.cat([key_org, encoder_hidden_states_org_key_proj], dim=1)
        value_org = torch.cat([value_org, encoder_hidden_states_org_value_proj], dim=1)

        inner_dim = key_org.shape[-1]
        head_dim = inner_dim // attn.heads
        query_org = query_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_org = key_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_org = value_org.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states_org = F.scaled_dot_product_attention(
            query_org, key_org, value_org, dropout_p=0.0, is_causal=False
        )
        hidden_states_org = hidden_states_org.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_org = hidden_states_org.to(query_org.dtype)

        # Split the attention outputs.
        hidden_states_org, encoder_hidden_states_org = (
            hidden_states_org[:, : residual.shape[1]],
            hidden_states_org[:, residual.shape[1] :],
        )

        hidden_states_org_negative = hidden_states_org[-origin_batch_size:]
        hidden_states_org_positive = hidden_states_org[-2 * origin_batch_size:-origin_batch_size]
        if self.guidance_type == "cfg_norm":
            raise NotImplementedError("cfg_norm guidance type is not yet implemented")
        else:  # "angular"
            hidden_states_org_guidance = hidden_states_org_positive * self.nag_scale - hidden_states_org_negative * (self.nag_scale - 1)
        norm_positive = torch.norm(hidden_states_org_positive, p=1, dim=-1, keepdim=True).expand(*hidden_states_org_positive.shape)
        norm_guidance = torch.norm(hidden_states_org_guidance, p=1, dim=-1, keepdim=True).expand(*hidden_states_org_guidance.shape)

        scale = norm_guidance / (norm_positive + 1e-7)
        hidden_states_org_guidance = hidden_states_org_guidance * torch.minimum(scale, scale.new_ones(1) * self.nag_tau) / (scale + 1e-7)

        hidden_states_org_guidance = hidden_states_org_guidance * self.nag_alpha + hidden_states_org_positive * (1 - self.nag_alpha)

        hidden_states_org = torch.cat((hidden_states_org[:origin_batch_size], hidden_states_org_guidance), dim=0)

        # linear proj
        hidden_states_org = attn.to_out[0](hidden_states_org)
        # dropout
        hidden_states_org = attn.to_out[1](hidden_states_org)
        if not attn.context_pre_only:
            encoder_hidden_states_org = attn.to_add_out(encoder_hidden_states_org)

        if input_ndim == 4:
            hidden_states_org = hidden_states_org.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_org = encoder_hidden_states_org.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################## perturbed path ##################

        batch_size = encoder_hidden_states_ptb.shape[0]

        # `sample` projections.
        query_ptb = attn.to_q(hidden_states_ptb)
        key_ptb = attn.to_k(hidden_states_ptb)
        value_ptb = attn.to_v(hidden_states_ptb)

        # `context` projections.
        encoder_hidden_states_ptb_query_proj = attn.add_q_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_key_proj = attn.add_k_proj(encoder_hidden_states_ptb)
        encoder_hidden_states_ptb_value_proj = attn.add_v_proj(encoder_hidden_states_ptb)

        # attention
        query_ptb = torch.cat([query_ptb, encoder_hidden_states_ptb_query_proj], dim=1)
        key_ptb = torch.cat([key_ptb, encoder_hidden_states_ptb_key_proj], dim=1)
        value_ptb = torch.cat([value_ptb, encoder_hidden_states_ptb_value_proj], dim=1)

        inner_dim = key_ptb.shape[-1]
        head_dim = inner_dim // attn.heads
        query_ptb = query_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key_ptb = key_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value_ptb = value_ptb.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # create a full mask with all entries set to 0
        seq_len = query_ptb.size(2)
        full_mask = torch.zeros((seq_len, seq_len), device=query_ptb.device, dtype=query_ptb.dtype)

        # set the attention value between image patches to -inf
        full_mask[:identity_block_size, :identity_block_size] = float("-inf")

        # set the diagonal of the attention value between image patches to 0
        full_mask[:identity_block_size, :identity_block_size].fill_diagonal_(0)

        # expand the mask to match the attention weights shape
        full_mask = full_mask.unsqueeze(0).unsqueeze(0)  # Add batch and num_heads dimensions

        hidden_states_ptb = F.scaled_dot_product_attention(
            query_ptb, key_ptb, value_ptb, attn_mask=full_mask, dropout_p=0.0, is_causal=False
        )
        hidden_states_ptb = hidden_states_ptb.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states_ptb = hidden_states_ptb.to(query_ptb.dtype)

        # split the attention outputs.
        hidden_states_ptb, encoder_hidden_states_ptb = (
            hidden_states_ptb[:, : residual.shape[1]],
            hidden_states_ptb[:, residual.shape[1] :],
        )

        # linear proj
        hidden_states_ptb = attn.to_out[0](hidden_states_ptb)
        # dropout
        hidden_states_ptb = attn.to_out[1](hidden_states_ptb)
        if not attn.context_pre_only:
            encoder_hidden_states_ptb = attn.to_add_out(encoder_hidden_states_ptb)

        if input_ndim == 4:
            hidden_states_ptb = hidden_states_ptb.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if context_input_ndim == 4:
            encoder_hidden_states_ptb = encoder_hidden_states_ptb.transpose(-1, -2).reshape(
                batch_size, channel, height, width
            )

        ################ concat ###############
        hidden_states = torch.cat([hidden_states_org, hidden_states_ptb])
        encoder_hidden_states = torch.cat([encoder_hidden_states_org[:2 * origin_batch_size], encoder_hidden_states_ptb, encoder_hidden_states_org[-origin_batch_size:]])

        return hidden_states, encoder_hidden_states