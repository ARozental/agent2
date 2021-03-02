import torch.nn as nn


class Compressor(nn.Module):
    def __init__(self, embed_size, parent_embed, dropout=0.2):
        super().__init__()
        self.recurrent = nn.LSTM(embed_size, parent_embed, dropout=dropout, batch_first=True)

    def forward(self, x):
        # TODO - Check if the padding mask needs to be passed to here?
        return self.recurrent(x)[0][:, -1, :]
