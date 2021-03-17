import torch.nn as nn
from src.config import Config
import torch
import numpy as np


class Compressor(nn.Module):
    def __init__(self, level):
        super().__init__()
        self.recurrent = nn.LSTM(Config.vector_sizes[level], Config.vector_sizes[level+1], dropout=0.0, batch_first=True)

    def forward(self, x,mask):
        lengths = (1 - mask.long()).sum(-1)
        places = (lengths - 1)
        r_out = self.recurrent(x)[0]
        out = torch.stack([r_out[i][places[i]] for i in range(x.shape[0])])

        return out #with consuming padding: self.recurrent(x)[0][:, -1, :]
