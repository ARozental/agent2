from src.transformer import PositionalEncoding
import torch.nn as nn
from src.config import Config


class Encoder(nn.Module):
    def __init__(self,level,# config,embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15
                 ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        encoder_layers = nn.TransformerEncoderLayer(Config.vector_sizes[level], Config.num_heads[level],Config.vector_sizes[level], Config.drop_rate,activation="relu") # change to swiglu
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])
        self.embed_size = Config.vector_sizes[level]

    def forward(self, src, mask):
        src = src.transpose(0, 1)
        src = self.pos_encoder(src) # * math.sqrt(Config.vector_sizes[level])

        return self.transformer_encoder(src, src_key_padding_mask=mask).transpose(0, 1)
