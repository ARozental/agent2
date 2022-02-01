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

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x, seq_dim=1):
        seq_len = x.shape[seq_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.cos_cached = emb.cos()[:, None, None, :]
            self.sin_cached = emb.sin()[:, None, None, :]
        return self.cos_cached, self.sin_cached

# rotary pos emb helpers:

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat(
        (-x2, x1), dim=x1.ndim - 1
    )   # dim=-1 triggers a bug in torch < 1.8.0

@torch.jit.script
def apply_rotary_pos_emb(q, k, cos, sin):
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class RotaryEncoderLayer(nn.TransformerEncoderLayer):
  def forward(self, src, src_mask=None, src_key_padding_mask=None):
    r"""Pass the input through the encoder layer.

    Args:
        src: the sequence to the encoder layer (required).
        src_mask: the mask for the src sequence (optional).
        src_key_padding_mask: the mask for the src keys per batch (optional).

    Shape:
        see the docs in Transformer class.
    """
    # rot = Rotary(src.shape[-1])
    # rot.forward(src)
    # q,k = apply_rotary_pos_emb(src, src,rot.cos_cached,rot.sin_cached)
    src2 = self.self_attn(src,src, src, attn_mask=src_mask,
                          key_padding_mask=src_key_padding_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)
    return src


EncoderLayer = RotaryEncoderLayer#nn.TransformerEncoderLayer
TransformerEncoder = nn.TransformerEncoder

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
