from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch
from src.config import Config


class AgentLevel(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.level = level
        self.encoder = Encoder(level)
        self.encoder_transform = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level],bias=False)  #
        self.decoder = Decoder(level)
        self.compressor = Compressor(level)
        self.decompressor = Decompressor(level)
        self.coherence_checker = CoherenceChecker(Config.vector_sizes[level])
        self.pad_vector = torch.rand(Config.vector_sizes[level], requires_grad=True) #doesn't really requires_grad, it is here for debug
        self.eos_vector = torch.rand(Config.vector_sizes[level], requires_grad=True) #todo: initialize right, not uniform
        self.join_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)
        self.mask_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)

        # these functions change the modes
    def get_children(self, node_batch,embedding_matrix=None):
      if self.level == 0: #words => get token vecs
        lookup_ids = torch.LongTensor([x.get_padded_word_tokens() for x in node_batch])
        mask = lookup_ids == Config.pad_token_id
        matrices = torch.index_select(embedding_matrix, 0, lookup_ids.view(-1)).view(len(node_batch),Config.sequence_lengths[self.level],Config.vector_sizes[self.level])
        labels = torch.LongTensor([x.get_padded_word_tokens() for x in node_batch])
        return self.level,matrices,mask,embedding_matrix,labels

      elif self.level == 1: #sentences => get word vecs
        masks = []
        for n in node_batch:
          mask = ([False for c in n.children]+[False]+([True]*Config.sequence_lengths[1]))[0:Config.sequence_lengths[1]]
          masks.append(mask)
        mask = torch.tensor(masks)

        lookup_ids = [list(map(lambda x: x.distinct_lookup_id+2+int(Config.join_texts),n.children)) for n in node_batch] #+3 for pad, eos and join, also need to change the matrix here and also handle Join-Tokens; todo: +2 if join is not in config
        lookup_ids = torch.LongTensor([(x+[1]+([0]*Config.sequence_lengths[1]))[0:Config.sequence_lengths[1]] for x in lookup_ids]) #0 for pad, 1 for eos, because that was the concate order with the word embedding matrix
        lookup_ids = torch.LongTensor(lookup_ids).view(-1)
        matrices = torch.index_select(embedding_matrix, 0, lookup_ids).view(len(node_batch),Config.sequence_lengths[1],Config.vector_sizes[1])
        labels = lookup_ids.view(len(node_batch),Config.sequence_lengths[1])

        return self.level,matrices, mask, embedding_matrix, labels

      else:
        matrices = []
        masks = []

        # flat all sentence nodes with pre_order_ids
        # zip(pre_id,range) to DefaultDict(int)
        # make an embedding matrix of all sentence vectors.
        # make labels by taking all ids; padding with 0; looking up in dict

        # todo somewhere: remove the pad vector from the matrix and labels-=1; we don't want any chance to select it


        for n in node_batch:
          mask = ([False for c in n.children]+[False]+([True]*Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
          masks.append(mask)
          matrix = ([c.vector for c in n.children]+[self.eos_vector]+([self.pad_vector]*Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
          matrix  = torch.stack(matrix)
          matrices.append(matrix)
        mask = torch.tensor(masks)
        matrices = torch.stack(matrices) #[sentences in node_batch, max words in sentence, word vec size]

        embedding = matrices.view(matrices.shape[0]*matrices.shape[1],matrices.shape[2]) #
        print(mask)
        print(matrices.shape)
        print(embedding.shape)
        #labels.shape should be [batch,seq_length] with id of the right place in the embedding matrix

        labels = torch.range(0,embedding.shape[0]).view(matrices.shape[0],matrices.shape[1])
        return self.level,matrices,mask,embedding,labels


      # labels = torch.tensor([x.get_padded_word_tokens() for x in node_batch])
      # batch_mask = torch.tensor([([False]*len(n.tokens)+[True]*Config.sequence_lengths[0])[0:Config.sequence_lengths[0]] for n in node_batch])
      # all_char_matrices = torch.index_select(local_char_embedding_matrix, 0, lookup_ids)  #[words_in_batch,max_chars_in_word,char_vector_size]
      # mlm = calc_mlm_loss(self.agent_levels[0],all_char_matrices,batch_mask,self.char_embedding_layer.weight,labels)
      # # labels [batch,seq_length,1] 1=>id in embedding matrix

      return self.level,matrices, mask, embedding_matrix, labels

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
