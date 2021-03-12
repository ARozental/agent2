from src.transformer import PositionalEncoding
from torch.nn.modules.transformer import _get_clones
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, embedding, eos, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15):
        super().__init__()

        self.embedding = embedding
        self.eos = eos
        self.embed_size = embed_size
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layer = nn.TransformerEncoderLayer(embed_size, num_head, num_hidden, dropout)
        self.encoder_layers = _get_clones(encoder_layer, num_layers)

        # Old
        # self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, src, mask):
        src = src.transpose(0, 1)
        eos_indices = torch.where(src == 2)
        src = self.embedding(src)  # * math.sqrt(self.embed_size)
        src[eos_indices] = 2
        src = self.pos_encoder(src)

        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_key_padding_mask=mask)
            output[eos_indices] = self.eos  # Reset back to the EoS tokens back

        return output.transpose(0, 1)

        # Old
        # return self.transformer_encoder(src, src_key_padding_mask=mask).transpose(0, 1)
