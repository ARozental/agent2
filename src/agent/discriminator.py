import torch.nn as nn
import torch


class Discriminator(nn.Module):
    #currently looks like CoherenceChecker; compare to a CNN Discriminator later
    def __init__(self, embed_size):
        super().__init__()
        self.d1 = nn.Linear(embed_size, 4*embed_size)
        self.d2 = nn.Linear(4*embed_size, 4*embed_size)
        self.out = nn.Linear(4*embed_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    # TODO - Add Dropout
    def forward(self, x):
        x = torch.tanh(self.d1(x))
        x = torch.tanh(self.d2(x))
        x = torch.sigmoid(self.out(x))
        return x

    def get_loss(self, x,labels):
        x = self.forward(x).squeeze(-1)
        return self.bce_loss(x,labels)
