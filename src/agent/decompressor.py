import torch.nn as nn
import torch


class Decompressor(nn.Module):
    def __init__(self, embed_size, parent_embed, max_seq_length, dropout=0.2):
        super().__init__()

        self.max_seq_length = max_seq_length
        self.parent_embed = parent_embed
        self.recurrent = nn.LSTM(parent_embed, parent_embed, dropout=dropout, batch_first=True)
        self.out_projection = nn.Linear(parent_embed, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Source: https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html
        # (num_layers * num_directions, batch, hidden_size)
        state_h, state_c = (torch.zeros(1, x.size(0), self.parent_embed), torch.zeros(1, x.size(0), self.parent_embed))

        seq = []
        last_input = x.unsqueeze(1)
        for i in range(self.max_seq_length):
            output, (state_h, state_c) = self.recurrent(last_input, (state_h, state_c))
            seq.append(output)
            last_input = output

        seq = torch.stack(seq)
        seq = self.out_projection(self.dropout(seq))

        return seq.squeeze(2).transpose(0, 1)
