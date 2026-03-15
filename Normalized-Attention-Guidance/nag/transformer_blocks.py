# Copyright 2025 Stability AI, The HuggingFace Team and The InstantX Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from typing import Any

import torch
import torch.nn as nn

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin, SD3Transformer2DLoadersMixin
from diffusers.utils import logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import JointTransformerBlock
from diffusers.models.attention_processor import (
    Attention,
    FusedJointAttnProcessor2_0,
    JointAttnProcessor2_0,
)
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings, PatchEmbed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.attention import JointTransformerBlock 
from diffusers.models.transformers import SD3Transformer2DModel

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name



def softclip(x, tau):
    return tau * torch.tanh(x / tau)


def angular_guidance(h_cond, h_uncond, scale, s1=1.0, s2=1.0, tau=0.5):
    norm_c = h_cond.norm(dim=-1, keepdim=True)
    a = h_cond / (norm_c + 1e-8)
    b = h_uncond / (h_uncond.norm(dim=-1, keepdim=True) + 1e-8)

    cos_theta = (a * b).sum(dim=-1, keepdim=True).clamp(-1, 1)
    theta = cos_theta.acos()
    # print("theta:", theta.mean().item())
    # Clamp the turning angles (not theta), matching the paper's threshold((ω-1)γ, τ)
    # phi1 = (scale * s1 * theta).clamp(max=tau * math.pi)
    # phi2 = (scale * s2 * theta).clamp(max=tau * math.pi)
    phi1 = softclip(scale * s1 * theta, tau * theta)
    phi2 = softclip(scale * s2 * theta, tau * theta)

    sin_t = theta.sin().clamp(min=1e-8)
    result = (torch.sin(theta + phi1) / sin_t) * a - \
             (torch.sin(phi2) / sin_t) * b

    result = torch.where(theta < 1e-5, a, result)
    return result * norm_c * 1.0


def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output



@maybe_allow_in_graph
class NAGJointTransformerBlock(JointTransformerBlock): 
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        joint_attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        joint_attention_kwargs = joint_attention_kwargs or {}
        if self.use_dual_attention:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp, norm_hidden_states2, gate_msa2 = self.norm1(
                hidden_states, emb=temb
            )
        else:
            norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(
                encoder_hidden_states, emb=temb
            )

        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            **joint_attention_kwargs,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        if self.use_dual_attention:
            attn_output2 = self.attn2(hidden_states=norm_hidden_states2, **joint_attention_kwargs)
            attn_output2 = gate_msa2.unsqueeze(1) * attn_output2
            hidden_states = hidden_states + attn_output2

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        if self.guidance_type == "angular" and self.nag_scale > 1.0:
            # print(ff_output.shape)
            ff_output = angular_guidance(ff_output[: ff_output.shape[0] // 2], ff_output[ff_output.shape[0] // 2 :], self.nag_scale)
            hidden_states = hidden_states[:1]
            # ff_output = ff_output[:1]
            
        hidden_states = hidden_states + ff_output
        # if self.guidance_type == "angular":
        #     # print(ff_output.shape)
        #     hidden_states = angular_guidance(hidden_states[: hidden_states.shape[0] // 2], hidden_states[hidden_states.shape[0] // 2 :], self.nag_scale)
            
        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            encoder_hidden_states = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return encoder_hidden_states, hidden_states

class NAGSD3Transformer2DModel(SD3Transformer2DModel):
    """
    The Transformer model introduced in [Stable Diffusion 3](https://huggingface.co/papers/2403.03206).

    Parameters:
        sample_size (`int`, defaults to `128`):
            The width/height of the latents. This is fixed during training since it is used to learn a number of
            position embeddings.
        patch_size (`int`, defaults to `2`):
            Patch size to turn the input data into small patches.
        in_channels (`int`, defaults to `16`):
            The number of latent channels in the input.
        num_layers (`int`, defaults to `18`):
            The number of layers of transformer blocks to use.
        attention_head_dim (`int`, defaults to `64`):
            The number of channels in each head.
        num_attention_heads (`int`, defaults to `18`):
            The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, defaults to `4096`):
            The embedding dimension to use for joint text-image attention.
        caption_projection_dim (`int`, defaults to `1152`):
            The embedding dimension of caption embeddings.
        pooled_projection_dim (`int`, defaults to `2048`):
            The embedding dimension of pooled text projections.
        out_channels (`int`, defaults to `16`):
            The number of latent channels in the output.
        pos_embed_max_size (`int`, defaults to `96`):
            The maximum latent height/width of positional embeddings.
        dual_attention_layers (`tuple[int, ...]`, defaults to `()`):
            The number of dual-stream transformer blocks to use.
        qk_norm (`str`, *optional*, defaults to `None`):
            The normalization to use for query and key in the attention layer. If `None`, no normalization is used.
    """

    _supports_gradient_checkpointing = True
    _no_split_modules = ["NAGJointTransformerBlock"]
    _skip_layerwise_casting_patterns = ["pos_embed", "norm"]

    @register_to_config
    def __init__(
        self,
        sample_size: int = 128,
        patch_size: int = 2,
        in_channels: int = 16,
        num_layers: int = 18,
        attention_head_dim: int = 64,
        num_attention_heads: int = 18,
        joint_attention_dim: int = 4096,
        caption_projection_dim: int = 1152,
        pooled_projection_dim: int = 2048,
        out_channels: int = 16,
        pos_embed_max_size: int = 96,
        dual_attention_layers: tuple[
            int, ...
        ] = (),  # () for sd3.0; (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12) for sd3.5
        qk_norm: str | None = None,
    ):
        super().__init__()
        self.out_channels = out_channels if out_channels is not None else in_channels
        self.inner_dim = num_attention_heads * attention_head_dim

        self.pos_embed = PatchEmbed(
            height=sample_size,
            width=sample_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=self.inner_dim,
            pos_embed_max_size=pos_embed_max_size,  # hard-code for now.
        )
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(
            embedding_dim=self.inner_dim, pooled_projection_dim=pooled_projection_dim
        )
        self.context_embedder = nn.Linear(joint_attention_dim, caption_projection_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                NAGJointTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    context_pre_only=i == num_layers - 1,
                    qk_norm=qk_norm,
                    use_dual_attention=True if i in dual_attention_layers else False,
                )
                for i in range(num_layers)
            ]
        )

        self.norm_out = AdaLayerNormContinuous(self.inner_dim, self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out = nn.Linear(self.inner_dim, patch_size * patch_size * self.out_channels, bias=True)

        self.gradient_checkpointing = False


