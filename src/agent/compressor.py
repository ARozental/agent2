from src.debug.profiler import Profiler as xp
from src.config import Config
import torch.nn as nn
import torch


class Compressor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.recurrent = nn.LSTM(
            Config.vector_sizes[level],
            Config.vector_sizes[level + 1],
            dropout=0.0,
            batch_first=True,
        )

        self.LayerNorm = nn.LayerNorm(Config.vector_sizes[level + 1])

    def forward(self, x, real_positions):
        self.recurrent.flatten_parameters()  # Suppresses RNN weights single contiguous chunk of memory warning
        with xp.Trace('Compressor'):
            lengths = real_positions.sum(-1).long()
            places = (lengths - 1)
            r_out = self.recurrent(x)[0]
            out = r_out[torch.arange(x.size(0)), places]

        return out  # with consuming padding: self.recurrent(x)[0][:, -1, :]
