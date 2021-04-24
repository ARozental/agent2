from src.transformer import PositionalEncoding, EncoderLayer, TransformerEncoder
import torch.nn as nn
from src.config import Config
from src.utils import gelu_new

class Encoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.fnn_sizes[level],
                                      Config.drop_rate, activation="gelu")  # change to swiglu
        encoder_layers.activation = gelu_new
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])

    def forward(self, src, mask, eos_positions):
        src = src.transpose(0, 1)
        eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1)

        eos_value = eos_positions * src
        src = src + self.pos_encoder(src)  # * math.sqrt(Config.vector_sizes[level])
        src = eos_positions * eos_value + (1 - eos_positions) * src

        return self.transformer_encoder(src, src_key_padding_mask=mask, eos_positions=eos_positions).transpose(0, 1)
