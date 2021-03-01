from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch


class AgentLevel(nn.Module):
    def __init__(self, level_num, num_tokens, max_seq_length, embed_size, parent_embed, encoder=None, decoder=None):
        super().__init__()

        if encoder is None:
            encoder = {}

        if decoder is None:
            decoder = {}

        self.level_num = level_num
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length
        self.embedding = nn.Embedding(num_tokens, embed_size)
        if self.level_num > 0:
            self.eos = nn.Parameter(torch.rand(embed_size))

        self.encoder = Encoder(self.embedding, embed_size=embed_size, **encoder)
        self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        self.decoder = Decoder(self.embedding, embed_size=embed_size, **decoder)
        self.compressor = Compressor(embed_size, parent_embed)
        self.decompressor = Decompressor(embed_size, parent_embed, max_seq_length)

        self.coherence_checker = CoherenceChecker(embed_size)

    def set_embedding(self, vectors):
        if self.level_num == 0:
            raise NotImplementedError  # Should not be here

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

    def debug_decode(self, vectors):
        decompressed = self.decompressor(vectors)
        output = self.decoder(tgt=decompressed, memory=decompressed)
        output = torch.argmax(output, dim=2)

        if self.level_num == 0:  # Let the tokenizer handle the convert from indices to characters
            return output

        # Convert to the corresponding embeddings
        output = self.embedding(output)

        return output
