from src.agent import Compressor, Decompressor, Encoder, Decoder
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, embed_size=200, num_hidden=200, num_layers=2, num_head=2, dropout=0.15, num_tokens=None,
                 max_seq_length=None):
        super().__init__()

        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.encoder = Encoder(self.embedding, embed_size=embed_size, num_hidden=num_hidden, num_layers=num_layers,
                               num_head=num_head, dropout=dropout)
        self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        self.decoder = Decoder(self.embedding, embed_size=embed_size, num_hidden=num_hidden, num_layers=num_layers,
                               num_head=num_head, dropout=dropout)
        self.compressor = Compressor(embed_size)
        self.decompressor = Decompressor(embed_size, max_seq_length)

    def mlm(self, src, mask):
        encoded = self.encoder(src, mask)

        encoded = encoded.transpose(0, 1)
        output = self.encoder_transform(encoded)
        emb_weight = torch.transpose(self.encoder.embedding.weight, 0, 1).unsqueeze(0)
        output = torch.matmul(output, emb_weight)  # [batch, seq_length, num_tokens]

        return output.transpose(1, 0)

    def forward(self, src, mask):
        encoded = self.encoder(src, mask)
        vector = self.compressor(encoded)
        decompressed = self.decompressor(vector)
        output = self.decoder(tgt=src, memory=decompressed, tgt_key_padding_mask=mask)

        return output

    def decode(self, src, mask):
        logits = self.forward(src, mask)
        return torch.argmax(logits, dim=2)
