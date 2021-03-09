from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch
from src.config import Config


class AgentLevel(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.level = level
        self.encoder = Encoder(level)
        self.encoder_transform = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level])  #
        self.decoder = Decoder(level)
        self.compressor = Compressor(level)
        self.decompressor = Decompressor(level)
        self.coherence_checker = CoherenceChecker(Config.vector_sizes[level])
        self.eos_vector = torch.rand(Config.vector_sizes[level], requires_grad=True) #todo: initialize right, not uniform
        self.mask_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)
        self.pad_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)
        self.join_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)

        # these functions change the modes
    def realize_vectors(self, node_batch):
      #todo: realize join vectors here
      matrices = []
      masks = []
      for n in node_batch:
        mask = ([False for c in n.children]+[False]+([True]*Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
        matrix = ([c.vector for c in n.children]+[self.eos_vector]+([self.pad_vector]*Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
        matrix  = torch.stack(matrix)
        matrices.append(matrix)
        masks.append(mask)

      matrices = torch.stack(matrices) #[sentences in node_batch, max words in sentence, word vec size]
      #todo: calc losses
      #todo: set losses on nodes

      mask = torch.tensor(masks)
      vectors = self.compressor(self.encoder(matrices, mask))
      [n.set_vector(v) for n, v in zip(node_batch, vectors)]
      return

    def calc_mlm_loss(self, embedding_matrix, node_batch):
        return

    def calc_coherence_loss(self, embedding_matrix, node_batch):
        return

    def calc_reconstruction_loss(self, embedding_matrix, node_batch):
        return




        # if encoder is None:
        #     encoder = {}
        #
        # if decoder is None:
        #     decoder = {}
        #
        # if compressor is None:
        #     compressor = {}
        #
        # if decompressor is None:
        #     decompressor = {}
        #
        # self.level_num = level_num
        # self.embed_size = embed_size
        # self.max_seq_length = max_seq_length
        # self.embedding = nn.Embedding(num_tokens, embed_size)
        # if self.level_num > 0:
        #     self.eos = nn.Parameter(torch.rand(embed_size))
        #
        # self.encoder = Encoder(self.embedding, embed_size=embed_size, **encoder)
        # self.encoder_transform = nn.Linear(embed_size, embed_size)  # For the MLM Loss only
        # self.decoder = Decoder(self.embedding, embed_size=embed_size, **decoder)
        # self.compressor = Compressor(embed_size, parent_embed, **compressor)
        # self.decompressor = Decompressor(embed_size, parent_embed, max_seq_length, **decompressor)
        #
        # self.coherence_checker = CoherenceChecker(parent_embed)

    # def set_embedding(self, vectors):
    #     if self.level_num == 0:
    #         raise NotImplementedError  # Should not be here
    #
    #     # TODO - Check that none of these get backpropped (except for the eos)
    #     weights = torch.cat([torch.stack([
    #         torch.zeros(self.embed_size),
    #         torch.zeros(self.embed_size),
    #         self.eos,
    #         torch.zeros(self.embed_size),
    #     ]), vectors])
    #
    #     self.embedding = nn.Embedding.from_pretrained(weights)
    #     self.encoder.embedding = nn.Embedding.from_pretrained(weights)
    #     self.decoder.embedding = nn.Embedding.from_pretrained(weights)
    #
    # def encode(self, src, mask):
    #     encoded = self.encoder(src, mask)
    #     return self.compressor(encoded)
    #
    # def debug_decode(self, vectors):
    #     decompressed = self.decompressor(vectors)
    #     output = self.decoder(tgt=decompressed, memory=decompressed)
    #     output = torch.matmul(output, self.embedding.weight.transpose(0, 1))
    #     output = torch.argmax(output, dim=2)
    #
    #     if self.level_num == 0:  # Let the tokenizer handle the convert from indices to characters
    #         return output
    #
    #     # Convert to the corresponding embeddings
    #     output = self.embedding(output)
    #
    #     return output
