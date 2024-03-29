from src.config import Config
import torch.nn as nn
import torch


class Generator(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.embed_size = embed_size
        self.d1 = nn.Linear(embed_size, 4 * embed_size)
        self.d2 = nn.Linear(4 * embed_size, 4 * embed_size)
        self.out = nn.Linear(4 * embed_size, embed_size)

    def forward(self, x):
        batch, vec_size = x.shape
        x = torch.randn(batch, vec_size, device=Config.device)  # normal dist is probably fine
        x = torch.tanh(self.d1(x))
        x = torch.tanh(self.d2(x))
        x = self.out(x)

        return x
