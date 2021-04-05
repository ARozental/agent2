import torch.nn as nn
import torch
import math


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


class EncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, src, src_mask=None, eos_positions=None, src_key_padding_mask=None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).
            eos_positions:

        Shape:
            see the docs in Transformer class.

        eos_positions => [batch,max_length,1]
        """
        eos_matrix = src * eos_positions
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src = src * (1 - eos_positions) + eos_matrix * eos_positions

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        src = src * (1 - eos_positions) + eos_matrix * eos_positions

        return src


class TransformerEncoder(nn.TransformerEncoder):
    def forward(self, src, mask=None, src_key_padding_mask=None, eos_positions=None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, eos_positions=eos_positions)

        if self.norm is not None:
            output = self.norm(output)

        return output
