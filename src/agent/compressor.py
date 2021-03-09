import torch.nn as nn
from src.config import Config


class Compressor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.recurrent = nn.LSTM(Config.vector_sizes[level], Config.vector_sizes[level+1], dropout=0.0, batch_first=True)

    def forward(self, x):
        # TODO - Check if the padding mask needs to be passed to here?
        return self.recurrent(x)[0][:, -1, :]
