import math
from typing import Optional, Tuple
from distutils.version import LooseVersion

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
from transformers.modeling_utils import set_module_tensor_to_device


class LlamaAttentionFused(nn.Module):
    def __init__(self, origin_attention):
        super().__init__()
        self.config = origin_attention.config
        self.hidden_size = origin_attention.hidden_size
        self.num_heads = origin_attention.num_heads
        self.head_dim = origin_attention.head_dim
        self.max_position_embeddings = origin_attention.max_position_embeddings

        self.qkv_proj = nn.Linear(
            origin_attention.hidden_size, origin_attention.num_heads * origin_attention.head_dim * 3, bias=False
        )
        fused_weight = torch.cat(
            [
                fc_node.weight.data
                for fc_node in [origin_attention.q_proj, origin_attention.k_proj, origin_attention.v_proj]
            ],
            dim=0,
        )
        set_module_tensor_to_device(
            self.qkv_proj, 'weight', fused_weight.device, value=fused_weight, dtype=fused_weight.dtype
        )
        self.o_proj = origin_attention.o_proj
        self.rotary_emb = origin_attention.rotary_emb

        origin_attention.q_proj = None
        origin_attention.k_proj = None
        origin_attention.v_proj = None

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()
        # use fused fc output to get qkv states
        qkv_states = self.qkv_proj(hidden_states).view(bsz, q_len, self.num_heads * 3, self.head_dim).transpose(1, 2)
        (query_states, key_states, value_states) = torch.chunk(qkv_states, 3, 1)

        is_causal = past_key_value is None

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value[0].shape[-2]
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)
        # [bsz, nh, t, hd]

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None
        if LooseVersion(torch.__version__) == LooseVersion('1.13.0'):
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                attn_output, attn_weights = F._scaled_dot_product_attention(
                    query_states, key_states, value_states, is_causal=is_causal
                )
        elif LooseVersion(torch.__version__) >= LooseVersion('2.0.0'):
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                attn_output = F.scaled_dot_product_attention(
                    query_states, key_states, value_states, is_causal=is_causal
                )
                attn_weights = None
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz * self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )

            if attention_mask is not None:
                if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                    raise ValueError(
                        f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is"
                        f" {attention_mask.size()}"
                    )
                attn_weights = attn_weights + attention_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_output = torch.matmul(attn_weights, value_states)

            if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )
        del query_states, key_states, value_states

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value
