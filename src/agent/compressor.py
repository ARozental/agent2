import torch.nn as nn


class Compressor(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.recurrent = nn.LSTM(embed_size, embed_size * 2, dropout=0.2, batch_first=True)

    def forward(self, x):
        return self.recurrent(x)[0][:, -1, :]
