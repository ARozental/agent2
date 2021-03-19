from src.transformer import PositionalEncoding,EncoderLayer, TransformerEncoder
import torch.nn as nn
from src.config import Config


class Encoder(nn.Module):
    def __init__(self,level,# config,embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15
                 ):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level],Config.vector_sizes[level], Config.drop_rate,activation="gelu") # change to swiglu
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])
        self.embed_size = Config.vector_sizes[level]

    def forward(self, src, mask,eos_positions):
        src = src.transpose(0, 1)
        eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1)

        eos_value = eos_positions*src
        src = src+self.pos_encoder(src) # * math.sqrt(Config.vector_sizes[level])
        src = eos_positions*eos_value+(1-eos_positions)*src


        return self.transformer_encoder(src, src_key_padding_mask=mask,eos_positions=eos_positions).transpose(0, 1)
