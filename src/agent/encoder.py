from src.transformer import PositionalEncoding, EncoderLayer, TransformerEncoder
from src.profiler import Profiler as xp
from src.utils import gelu_new
from src.config import Config
import torch.nn as nn
import torch


class Encoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.fnn_sizes[level],
                                      Config.drop_rate, activation="gelu")  # change to swiglu
        encoder_layers.activation = gelu_new
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])

    def forward(self, src, real_positions, eos_positions):
        with xp.Trace('Encoder'):
            src = src.transpose(0, 1)
            # eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1)
            att_add_mask = torch.log(real_positions.float())

            # eos_value = eos_positions * src
            src = src + self.pos_encoder(src)  # * math.sqrt(Config.vector_sizes[level])
            # src = eos_positions * eos_value + (1 - eos_positions) * src

            return self.transformer_encoder(src, src_key_padding_mask=att_add_mask).transpose(0, 1)
