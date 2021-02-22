from src.agent import Compressor, Decompressor, Encoder
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15, num_tokens=None,
                 max_seq_length=None):
        super().__init__()

        self.out_transform = nn.Linear(embed_size, embed_size)

        self.encoder = Encoder(embed_size=embed_size, num_hidden=num_hidden, num_layers=num_layers, num_head=num_head,
                               dropout=dropout, num_tokens=num_tokens)
        self.compressor = Compressor(embed_size)
        self.decompressor = Decompressor(embed_size, max_seq_length)

    def forward(self, src, mask):
        encoded = self.encoder(src, mask)

        # This is the original code
        # TODO - The double transpose should not be necessary in the new code
        encoded = encoded.transpose(0, 1)
        output = self.out_transform(encoded)
        emb_weight = torch.transpose(self.encoder.embedding.weight, 0, 1).unsqueeze(0)
        output = torch.matmul(output, emb_weight)  # [batch, seq_length, num_tokens]
        output = output.transpose(1, 0)

        # TODO - This will be the new code once it works and it will replace the original above
        # vector = self.compressor(encoded)
        # decompressed = self.decompressor(vector)

        return output

    def decode(self, src, mask):
        logits = self.forward(src, mask)
        return torch.argmax(logits, dim=2)
