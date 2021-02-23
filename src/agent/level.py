from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch


class Level(nn.Module):
    def __init__(self, num_tokens, max_seq_length, embed_size, parent_embed, encoder=None, decoder=None):
        super().__init__()

        if encoder is None:
            encoder = {}

        if decoder is None:
            decoder = {}

        self.embedding = nn.Embedding(num_tokens, embed_size)
        self.encoder = Encoder(self.embedding, embed_size=embed_size, **encoder)
        self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        self.decoder = Decoder(self.embedding, embed_size=embed_size, **decoder)
        self.compressor = Compressor(embed_size, parent_embed)
        self.decompressor = Decompressor(embed_size, parent_embed, max_seq_length)

        self.coherence_checker = CoherenceChecker(embed_size)

    def mlm(self, src, mask):
        encoded = self.encoder(src, mask)

        encoded = encoded.transpose(0, 1)
        output = self.encoder_transform(encoded)
        emb_weight = torch.transpose(self.encoder.embedding.weight, 0, 1).unsqueeze(0)
        output = torch.matmul(output, emb_weight)  # [batch, seq_length, num_tokens]

        return output.transpose(1, 0)

    def coherence(self, src, mask):
        encoded = self.encoder(src, mask)
        vector = self.compressor(encoded)
        return self.coherence_checker(vector)

    def forward(self, src, mask):
        encoded = self.encoder(src, mask)
        vector = self.compressor(encoded)
        decompressed = self.decompressor(vector)
        output = self.decoder(tgt=src, memory=decompressed, tgt_key_padding_mask=mask)

        return output

    def decode(self, src, mask):
        logits = self.forward(src, mask)
        return torch.argmax(logits, dim=2)
