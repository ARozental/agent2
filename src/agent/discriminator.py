import torch.nn as nn
import torch


class Discriminator(nn.Module):
    # currently looks like CoherenceChecker; compare to a CNN Discriminator later
    def __init__(self, vector_size):  # for level+1
        super().__init__()
        self.d1 = nn.Linear(vector_size, 4 * vector_size)
        self.d2 = nn.Linear(4 * vector_size, 4 * vector_size)
        self.out = nn.Linear(4 * vector_size, 1)
        self.bce_loss = nn.BCEWithLogitsLoss()

    # TODO - Add Dropout
    def forward(self, x):
        x = torch.tanh(self.d1(x))
        x = torch.tanh(self.d2(x))
        x = self.out(x)
        return x

    def get_loss(self, x, labels):
        x = self.forward(x).squeeze(-1)
        return self.bce_loss(x, labels)


class CnnDiscriminator(nn.Module):
    # needs the matrices object for to run; move to loss functions?
    def __init__(self, vector_size, sequence_length):  # for the level
        super().__init__()
        self.num_filters = 11
        self.conv = nn.Conv1d(vector_size, self.num_filters, 1)  # <input_vector_size, num_filters, 1=unigram>
        self.max_pool = nn.MaxPool1d(sequence_length)
        self.act = nn.ReLU()
        self.bce_loss = nn.BCEWithLogitsLoss()

        self.d1 = nn.Linear(self.num_filters, 1)

    def forward(self, x):  # gets matrices after decompress and decode
        x = torch.transpose(x, 1, 2)
        x = self.conv(x)
        x = self.max_pool(x).squeeze(-1)
        x = torch.sigmoid(self.d1(x)).squeeze(-1)
        return x

    def get_loss(self, x, labels):
        x = self.forward(x).squeeze(-1)
        return self.bce_loss(x, labels)
