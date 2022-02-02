# from torch.nn import functional as F
# from torch import Tensor
# from torch.nn.functional import linear
# from typing import Optional, Tuple
# import warnings
# import torch
# from torch.overrides import has_torch_function,handle_torch_function
#
# def multi_head_attention_forward(
#     query: Tensor,
#     key: Tensor,
#     value: Tensor,
#     embed_dim_to_check: int,
#     num_heads: int,
#     in_proj_weight: Tensor,
#     in_proj_bias: Tensor,
#     bias_k: Optional[Tensor],
#     bias_v: Optional[Tensor],
#     add_zero_attn: bool,
#     dropout_p: float,
#     out_proj_weight: Tensor,
#     out_proj_bias: Tensor,
#     training: bool = True,
#     key_padding_mask: Optional[Tensor] = None,
#     need_weights: bool = True,
#     attn_mask: Optional[Tensor] = None,
#     use_separate_proj_weight: bool = False,
#     q_proj_weight: Optional[Tensor] = None,
#     k_proj_weight: Optional[Tensor] = None,
#     v_proj_weight: Optional[Tensor] = None,
#     static_k: Optional[Tensor] = None,
#     static_v: Optional[Tensor] = None,
# ) -> Tuple[Tensor, Optional[Tensor]]:
#   r"""
#   Args:
#       query, key, value: map a query and a set of key-value pairs to an output.
#           See "Attention Is All You Need" for more details.
#       embed_dim_to_check: total dimension of the model.
#       num_heads: parallel attention heads.
#       in_proj_weight, in_proj_bias: input projection weight and bias.
#       bias_k, bias_v: bias of the key and value sequences to be added at dim=0.
#       add_zero_attn: add a new batch of zeros to the key and
#                      value sequences at dim=1.
#       dropout_p: probability of an element to be zeroed.
#       out_proj_weight, out_proj_bias: the output projection weight and bias.
#       training: apply dropout if is ``True``.
#       key_padding_mask: if provided, specified padding elements in the key will
#           be ignored by the attention. This is an binary mask. When the value is True,
#           the corresponding value on the attention layer will be filled with -inf.
#       need_weights: output attn_output_weights.
#       attn_mask: 2D or 3D mask that prevents attention to certain positions. A 2D mask will be broadcasted for all
#           the batches while a 3D mask allows to specify a different mask for the entries of each batch.
#       use_separate_proj_weight: the function accept the proj. weights for query, key,
#           and value in different forms. If false, in_proj_weight will be used, which is
#           a combination of q_proj_weight, k_proj_weight, v_proj_weight.
#       q_proj_weight, k_proj_weight, v_proj_weight, in_proj_bias: input projection weight and bias.
#       static_k, static_v: static key and value used for attention operators.
#
#
#   Shape:
#       Inputs:
#       - query: :math:`(L, N, E)` where L is the target sequence length, N is the batch size, E is
#         the embedding dimension.
#       - key: :math:`(S, N, E)`, where S is the source sequence length, N is the batch size, E is
#         the embedding dimension.
#       - value: :math:`(S, N, E)` where S is the source sequence length, N is the batch size, E is
#         the embedding dimension.
#       - key_padding_mask: :math:`(N, S)` where N is the batch size, S is the source sequence length.
#         If a ByteTensor is provided, the non-zero positions will be ignored while the zero positions
#         will be unchanged. If a BoolTensor is provided, the positions with the
#         value of ``True`` will be ignored while the position with the value of ``False`` will be unchanged.
#       - attn_mask: 2D mask :math:`(L, S)` where L is the target sequence length, S is the source sequence length.
#         3D mask :math:`(N*num_heads, L, S)` where N is the batch size, L is the target sequence length,
#         S is the source sequence length. attn_mask ensures that position i is allowed to attend the unmasked
#         positions. If a ByteTensor is provided, the non-zero positions are not allowed to attend
#         while the zero positions will be unchanged. If a BoolTensor is provided, positions with ``True``
#         are not allowed to attend while ``False`` values will be unchanged. If a FloatTensor
#         is provided, it will be added to the attention weight.
#       - static_k: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#         N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
#       - static_v: :math:`(N*num_heads, S, E/num_heads)`, where S is the source sequence length,
#         N is the batch size, E is the embedding dimension. E/num_heads is the head dimension.
#
#       Outputs:
#       - attn_output: :math:`(L, N, E)` where L is the target sequence length, N is the batch size,
#         E is the embedding dimension.
#       - attn_output_weights: :math:`(N, L, S)` where N is the batch size,
#         L is the target sequence length, S is the source sequence length.
#   """
#   tgt_len, bsz, embed_dim = query.size()
#
#   head_dim = embed_dim // num_heads
#   scaling = float(head_dim) ** -0.5
#
#   q, k, v = linear(query, in_proj_weight, in_proj_bias).chunk(3, dim=-1)
#
#   q = q * scaling
#
#   q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
#   k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#   v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
#   src_len = k.size(1)
#
#   attn_output_weights = torch.bmm(q, k.transpose(1, 2))
#
#   attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
#   #attn_output_weights = attn_output_weights.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2),float("-inf")) #old
#   attn_output_weights += key_padding_mask.unsqueeze(1).unsqueeze(2) #change is here
#   attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)
#
#   attn_output_weights = F.softmax(attn_output_weights, dim=-1)
#   attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)
#
#   attn_output = torch.bmm(attn_output_weights, v)
#   attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
#   attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
#
#   return attn_output, None