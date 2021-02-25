from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch


class Level(nn.Module):
    def __init__(self, num_tokens, max_seq_length, embed_size, parent_embed, is_base=False, encoder=None, decoder=None):
        super().__init__()

        if encoder is None:
            encoder = {}

        if decoder is None:
            decoder = {}

        self.is_base = is_base
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        if is_base:
            self.embedding = nn.Embedding(num_tokens, embed_size)
        else:
            self.embedding = nn.Embedding(num_tokens, embed_size)  # TODO - Don't let this get backpropped
            self.eos = nn.Parameter(torch.rand(embed_size))

        self.encoder = Encoder(self.embedding, embed_size=embed_size, **encoder)
        self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        self.decoder = Decoder(self.embedding, embed_size=embed_size, **decoder)
        self.compressor = Compressor(embed_size, parent_embed)
        self.decompressor = Decompressor(embed_size, parent_embed, max_seq_length)

        self.coherence_checker = CoherenceChecker(embed_size)

    def set_embedding(self, vectors):
        # TODO - Check that none of these get backpropped (except for the eos)
        weights = torch.cat([torch.stack([
            torch.zeros(self.embed_size),
            torch.zeros(self.embed_size),
            self.eos,
            torch.zeros(self.embed_size),
        ]), vectors])

        self.embedding = nn.Embedding.from_pretrained(weights)
        self.encoder.embedding = nn.Embedding.from_pretrained(weights)
        self.decoder.embedding = nn.Embedding.from_pretrained(weights)

    def reconstruct(self, src, mask):
        encoded = self.encoder(src, mask)
        vector = self.compressor(encoded)
        decompressed = self.decompressor(vector)
        output = self.decoder(tgt=decompressed, memory=decompressed)

        return output

    def encode(self, src, mask):
        encoded = self.encoder(src, mask)
        return self.compressor(encoded)

    def decode(self, vectors):
        decompressed = self.decompressor(vectors)

        # Alon - I believe something is wrong here, working on trying to identify
        src = torch.zeros((decompressed.size(0), self.max_seq_length), dtype=torch.long)
        output = self.decoder(tgt=decompressed, memory=decompressed)

        if self.is_base:
            return torch.argmax(output, dim=2)

        return output
