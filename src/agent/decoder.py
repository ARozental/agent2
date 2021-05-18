from src.transformer import PositionalEncoding, EncoderLayer, TransformerEncoder
from src.config import Config
import torch.nn as nn
import torch
from src.utils import gelu_new


class Decoder(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.level = level
        self.pos_encoder = PositionalEncoding(Config.vector_sizes[level], Config.drop_rate)  # should it be always 0?
        encoder_layers = EncoderLayer(Config.vector_sizes[level], Config.num_heads[level], Config.fnn_sizes[level],
                                      Config.drop_rate, activation="gelu")  # change to swiglu
        encoder_layers.activation = gelu_new
        self.transformer_encoder = TransformerEncoder(encoder_layers, Config.num_transformer_layers[level])

        # this is for RC
        if level > 0:
            self.d1 = nn.Linear(Config.vector_sizes[level], 4 * Config.vector_sizes[level])
            # self.d2 = nn.Linear(4 * Config.vector_sizes[level], 4 * Config.vector_sizes[level])
            self.out = nn.Linear(4 * Config.vector_sizes[level], Config.vector_sizes[level])

    def forward(self, src, real_positions, eos_positions):
        src = src.transpose(0, 1)
        att_add_mask = torch.log(real_positions)
        # todo: fix?? due to the positional encoding not all eos are the same vec, #do we even need pos encoding here?
        # eos_positions = eos_positions.transpose(0, 1).unsqueeze(-1)

        # eos_value = eos_positions * src
        src = src - self.pos_encoder(src)  # * math.sqrt(Config.vector_sizes[level])
        # src = eos_positions * eos_value + (1 - eos_positions) * src

        encoded = self.transformer_encoder(src, src_key_padding_mask=att_add_mask).transpose(0, 1)

        if self.level == 0:  # todo? remove if from here, have it outside
            return encoded

        x = torch.tanh(encoded)
        x = torch.tanh(self.d1(x))
        return self.out(x)
