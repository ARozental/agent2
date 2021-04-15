from src.transformer import PositionalEncoding, EncoderLayer, TransformerEncoder
from src.config import Config
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)  # should it be always 0?
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.fnn_sizes[level],
                                      Config.drop_rate, activation="gelu")  # change to swiglu
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])

    def forward(self, src, mask, eos_positions):
        src = src.transpose(0, 1)

        # todo: fix?? due to the positional encoding not all eos are the same vec, #do we even need pos encoding here?
        eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1)

        eos_value = eos_positions * src
        src = src - self.pos_encoder(src)  # * math.sqrt(Config.vector_sizes[level])
        src = eos_positions * eos_value + (1 - eos_positions) * src

        return self.transformer_encoder(src, src_key_padding_mask=mask, eos_positions=eos_positions).transpose(0, 1)
