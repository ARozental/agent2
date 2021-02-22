import torch.nn as nn
import torch


class CoherenceChecker(nn.Module):
    def __init__(self, embed_size=200):
        super().__init__()
        self.d1 = nn.Linear(embed_size * 2, embed_size * 2)
        self.d2 = nn.Linear(embed_size * 2, embed_size * 2)
        self.out = nn.Linear(embed_size * 2, 1)

    # TODO - Add Dropout
    def forward(self, x):
        x = torch.tanh(self.d1(x))
        x = torch.tanh(self.d2(x))
        x = torch.sigmoid(self.out(x))

        return x
