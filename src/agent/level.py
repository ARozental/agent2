from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch


class AgentLevel(nn.Module):
    def __init__(self, level_num, num_tokens, max_seq_length, embed_size, parent_embed,
                 encoder=None, decoder=None, compressor=None, decompressor=None):
        super().__init__()

        if encoder is None:
            encoder = {}

        if decoder is None:
            decoder = {}

        if compressor is None:
            compressor = {}

        if decompressor is None:
            decompressor = {}

        self.level_num = level_num
        self.embed_size = embed_size
        self.max_seq_length = max_seq_length

        self.eos = nn.Parameter(torch.rand(embed_size))
        self.embedding = None  # Will be set later by the levels
        if self.level_num == 0:
            self.embedding_matrix = nn.Parameter(torch.rand((num_tokens, embed_size)))
            self.set_embedding(self.embedding_matrix)

        self.encoder = Encoder(self.do_embedding, embed_size=embed_size, **encoder)
        self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        self.decoder = Decoder(self.do_embedding, embed_size=embed_size, **decoder)
        self.compressor = Compressor(embed_size, parent_embed, **compressor)
        self.decompressor = Decompressor(embed_size, parent_embed, max_seq_length, **decompressor)

        self.coherence_checker = CoherenceChecker(parent_embed)

    def do_embedding(self, x):
        return self.embedding(x)

    def set_embedding(self, vectors):
        weights = torch.cat([torch.stack([
            torch.zeros(self.embed_size),
            torch.zeros(self.embed_size),
            self.eos,
            torch.zeros(self.embed_size),
        ]), vectors])

        self.embedding = nn.Embedding.from_pretrained(weights, freeze=True)

    def encode(self, src, mask):
        encoded = self.encoder(src, mask)
        return self.compressor(encoded)

    def debug_decode(self, vectors):
        decompressed = self.decompressor(vectors)
        output = self.decoder(tgt=decompressed, memory=decompressed)
        output = torch.matmul(output, self.embedding.weight.transpose(0, 1))
        output = torch.argmax(output, dim=2)

        if self.level_num == 0:  # Let the tokenizer handle the convert from indices to characters
            return output

        # Convert to the corresponding embeddings
        output = self.embedding(output)

        return output
