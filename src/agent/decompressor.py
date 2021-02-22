import torch.nn as nn
import torch


class Decompressor(nn.Module):
    def __init__(self, embed_size, max_seq_length):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.embed_size = embed_size
        self.recurrent = nn.LSTM(embed_size, embed_size, dropout=0.2, batch_first=True)
        self.out_projection = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # (num_layers * num_directions, batch, hidden_size)
        state_h, state_c = (torch.zeros(1, 2, self.embed_size), torch.zeros(1, 2, self.embed_size))

        seq = []
        last_input = x.unsqueeze(1)
        for i in range(self.max_seq_length):
            output, (state_h, state_c) = self.recurrent(last_input, (state_h, state_c))
            seq.append(output)
            last_input = output

        seq = torch.stack(seq)
        seq = self.out_projection(self.dropout(seq))

        return seq.squeeze(2).transpose(0, 1)
