import torch.nn as nn
import torch
import math
from torch.nn import functional as F
from src.utils import gelu_new,prob_to_logit
import copy
def get_clones(module, N):
  return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


#F.multi_head_attention_forward = multi_head_attention_forward

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
    def __init__(self, seq_len, dim, base=10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
        self.short_matrix = None

        t = torch.arange(seq_len).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Parameter(emb.cos())
        self.sin_cached = nn.Parameter(emb.sin())

        m = [list(range(seq_len))]

        for i in range(seq_len - 1):
            y = [m[-1][0] + 1] + m[-1][:-1]
            m.append(y)
        self.short_matrix = nn.Parameter(-torch.tensor(m),requires_grad=False)

    def forward(self):
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


def attention(q, k, v, d_k, mask, att_prior_bias, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)


  scores += mask.unsqueeze(1).unsqueeze(2) #add floating attention
  scores += att_prior_bias.unsqueeze(0)



  scores = F.softmax(scores, dim=-1)



  if dropout is not None:
    scores = dropout(scores)

  output = torch.matmul(scores, v)
  return output


class MultiHeadAttention2(nn.Module):
  def __init__(self, heads, d_model, dropout=0.1):
    super().__init__()

    self.d_model = d_model
    self.d_k = d_model // heads
    self.h = heads

    self.q_linear = nn.Linear(d_model, d_model)
    self.v_linear = nn.Linear(d_model, d_model) #todo: delete here!
    self.k_linear = nn.Linear(d_model, d_model)
    self.dropout = nn.Dropout(dropout)
    self.out = nn.Linear(d_model, d_model)

    self.att_prior_bias = nn.Parameter(torch.tensor([math.log(p/(1-p)) for p in [ (2**j)/(2**(heads+1)) for j in range(1,heads+1) ]], requires_grad=True))
    self.sigmoid = nn.Sigmoid()
    self.batch_first = False
  def forward(self, q, k, v, rotary= None, mask=None, is_causal=False):
    bs = q.size(0)

    # perform linear operation and split into h heads
    q = self.q_linear(q)
    k = self.k_linear(k)
    v = self.v_linear(v)

    cos_cached, sin_cached = rotary()
    q, k = apply_rotary_pos_emb(q, k, cos_cached, sin_cached)

    q = q.view(bs, -1, self.h, self.d_k)
    k = k.view(bs, -1, self.h, self.d_k)
    v = v.view(bs, -1, self.h, self.d_k)


    prior_bias_matrices = self.sigmoid(self.att_prior_bias).unsqueeze(-1).unsqueeze(-1)*rotary.short_matrix#(heads,length,length)

    # transpose to get dimensions bs * h * sl * d_model
    k = k.transpose(1, 2)
    q = q.transpose(1, 2)
    v = v.transpose(1, 2)
    # calculate attention using function we will define next



    scores = attention(q, k, v, self.d_k, mask, prior_bias_matrices, self.dropout)

    # concatenate heads and put through final linear layer
    concat = scores.transpose(1, 2).contiguous() \
      .view(bs, -1, self.d_model)

    output = self.out(concat)

    return output


class EncoderLayer2(nn.TransformerEncoderLayer):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="gelu",
                 layer_norm_eps=1e-5,
                 device=None, dtype=None,rotary=None) -> None:
        super().__init__(d_model, nhead, dim_feedforward, dropout)
        self.rotary = rotary
        #self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, **factory_kwargs)
        self.self_attn = MultiHeadAttention2(nhead,d_model, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = gelu_new

    def forward(self, src, src_mask = None,
                src_key_padding_mask= None, is_causal=False):
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        #src.shape #[batch,length,hidden],[16, 21, 32]
        #src2 = self.self_attn(src, src, src, attn_mask=src_mask,key_padding_mask=src_key_padding_mask)[0]

        src2 = self.self_attn.forward(src, src, src, rotary=self.rotary, mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src) #todo: check if deep norm be like src = self.norm1(sqrt(2*_layer_num) * src + self.dropout1(src2))
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerEncoder2(nn.TransformerEncoder):
  def __init__(self, encoder_layer, num_layers, norm=None):
    super().__init__(encoder_layer, num_layers, norm)
    self.layers = get_clones(encoder_layer, num_layers)
    self.num_layers = num_layers
    self.norm = norm


#EncoderLayer = EncoderLayer____basic#RotaryEncoderLayer#nn.TransformerEncoderLayer
EncoderLayer = EncoderLayer2
TransformerEncoder = TransformerEncoder2

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
