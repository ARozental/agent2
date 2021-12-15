from src.debug.profiler import Profiler as xp
from src.config import Config
import torch.nn as nn
import torch
from src.utils import gelu_new


class Compressor1(nn.Module):
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
class CnnCompressor(nn.Module):
  # needs the matrices object for to run; move to loss functions?
    def __init__(self, level):  # for the level
        super().__init__()
        self.num_filters = Config.vector_sizes[level+1]
        self.conv = nn.Conv1d(Config.vector_sizes[level], self.num_filters, 1)  # <input_vector_size, num_filters, 1=unigram>
        self.max_pool = nn.MaxPool1d(Config.sequence_lengths[level])
        self.act = gelu_new
        self.d1 = nn.Linear(self.num_filters, Config.vector_sizes[level+1])

    def forward(self, x, real_positions):  # gets matrices after decompress and decode
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = x+((1.0-real_positions.unsqueeze(1))*(-99999999))#so the maxpool will kill everything in a fake position
        x = self.max_pool(x).squeeze(-1)
        x = self.d1(self.act(x))
        return x

if Config.cnn_compressor:
  Compressor = CnnCompressor
else:
  Compressor = Compressor1