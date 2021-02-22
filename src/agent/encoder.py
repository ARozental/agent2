from src.transformer import PositionalEncoding
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15, num_tokens=None):
        super().__init__()
        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout)
        encoder_layers = nn.TransformerEncoderLayer(embed_size, num_head, num_hidden, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.embed_size = embed_size

    def forward(self, src, mask):
        src = src.transpose(0, 1)
        src = self.embedding(src)  # * math.sqrt(self.embed_size)
        src = self.pos_encoder(src)

        return self.transformer_encoder(src, src_key_padding_mask=mask).transpose(0, 1)
