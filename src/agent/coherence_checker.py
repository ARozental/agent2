import torch.nn as nn
import torch
from src.config import Config

class CoherenceChecker(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.d1 = nn.Linear(embed_size, 4 * embed_size)
        self.d2 = nn.Linear(4 * embed_size, 4 * embed_size)
        self.out = nn.Linear(4 * embed_size, 1)

    # TODO - Add Dropout
    def forward(self, x):
        x = torch.tanh(self.d1(x))
        x = torch.tanh(self.d2(x))
        x = torch.sigmoid(self.out(x)) * Config.max_coherence_noise

        return x
