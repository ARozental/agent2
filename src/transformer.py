import torch.nn as nn
import torch
import math
from torch.nn import functional as F
from src.floating_attention import multi_head_attention_forward


F.multi_head_attention_forward = multi_head_attention_forward

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.0, max_len=256):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.dropout(self.pe[:x.size(0), :])

#todo: delete and use regular nn.TransformerEncoderLayer
#class EncoderLayer(nn.TransformerEncoderLayer):
EncoderLayer = nn.TransformerEncoderLayer
TransformerEncoder = nn.TransformerEncoder


# class EncoderLayer(nn.TransformerEncoderLayer):
#     def forward(self, src, src_mask=None, eos_positions=None, src_key_padding_mask=None) -> torch.Tensor:
#         r"""Pass the input through the encoder layer.
#
#         Args:
#             src: the sequence to the encoder layer (required).
#             src_mask: the mask for the src sequence (optional).
#             src_key_padding_mask: the mask for the src keys per batch (optional).
#             eos_positions:
#
#         Shape:
#             see the docs in Transformer class.
#
#         eos_positions => [batch,max_length,1]
#         """
#
#         ttt = torch.tensor([[[ 0.1044,  2.7534, -0.4628,  0.1192],[ 1.3539,  0.8297,  1.1808,  0.5982]],
#                             [[ 2.1954,  0.3700,  1.1908,  0.5982],[ 1.1953,  0.5169,  2.8376, -0.0195]],
#                             [[ 1.0933, -0.6032,  0.4644,  1.4896],[-0.1194,  1.3594,  1.3380,  0.5812]],
#                             [[-1.3659, -1.8948, -0.3575,  0.1278],[-1.3659, -1.8948, -0.3575,  0.1278]],
#                             [[ 1.9110, -0.3735, -0.2392, -1.0227],[-1.1085, -1.4616, -0.0251,  2.0344]],
#                             [[-0.4807, -1.0984,  0.5656,  0.9581],[ 1.9110, -0.3735, -0.2392, -1.0227]],
#                             [[ 1.1352, -0.6281,  0.5856,  0.9569], [ 1.1352, -0.6281,  0.5856,  0.9569]]])
#         src = src[:,0:2,:]
#         ttt = ttt[:,0:2,:]
#         src_key_padding_mask = src_key_padding_mask[0:2]
#
#         real_positions = torch.log((1 - src_key_padding_mask.float()))
#         print(real_positions)
#         src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=real_positions)[0]
#         #print("src_mask",src_mask)
#         #print("src_key_padding_mask",src_key_padding_mask)
#         #print("src2",src2.transpose(0, 1))
#         print("ttt",self.self_attn(ttt, ttt, ttt, attn_mask=src_mask, key_padding_mask=real_positions)[0].transpose(0, 1))
#         [x for x in range(100000)]
#         1+None
#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#
#         #src = src * (1 - eos_positions) + eos_matrix * eos_positions
#
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)
#
#         #src = src * (1 - eos_positions) + eos_matrix * eos_positions
#
#         return src
#
#
# class TransformerEncoder(nn.TransformerEncoder):
#     def forward(self, src, mask=None, src_key_padding_mask=None, eos_positions=None) -> torch.Tensor:
#         output = src
#         for mod in self.layers:
#             output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, eos_positions=eos_positions)
#
#         if self.norm is not None:
#             output = self.norm(output)
#
#         return output
