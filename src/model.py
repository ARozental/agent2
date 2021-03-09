from src.losses import MLMLoss, CoherenceLoss, ReconstructionLoss
from src.agent import AgentLevel
from src.utils import find_level
from typing import Iterator
from torch.nn import Parameter
import numpy as np
import torch.nn as nn
import torch
from src.config import Config


class AgentModel(nn.Module):
    def __init__(self,tree_tokenizer):
        super().__init__()
        self.tree_tokenizer = tree_tokenizer
        self.agent_levels = nn.ModuleList()
        for i in range(Config.agent_level + 1):
            agent_level = AgentLevel(i)

            self.agent_levels.append(agent_level)

        self.char_embedding_matrix = nn.Embedding(len(tree_tokenizer.letter_tokenizer.keys()), Config.vector_sizes[0])


    def set_word_vectors(self,batch_tree):
      node_batch = batch_tree.level_nodes[0]
      local_char_embedding_matrix = torch.LongTensor(batch_tree.distinct_word_embedding_tokens)
      mask = local_char_embedding_matrix==1 #True => position to mask
      local_char_embedding_matrix = self.char_embedding_matrix(local_char_embedding_matrix)
      word_embedding_matrix = self.agent_levels[0].compressor(self.agent_levels[0].encoder(local_char_embedding_matrix, mask)) #[distinct_words_in_batch,word_vector_size]
      lookup_ids = torch.LongTensor([x.distinct_lookup_id for x in node_batch])
      all_word_vectors = torch.index_select(word_embedding_matrix, 0, lookup_ids)  #[words_in_batch,word_vector_size]

      #all_char_matrices = torch.index_select(local_char_embedding_matrix, 0, lookup_ids)  #[words_in_batch,max_chars_in_word,char_vector_size]
      #todo: calc losses here

      [n.set_vector(v) for n,v in zip(node_batch,all_word_vectors)]
      return

    def set_text_vectors(self,batch_tree):
      self.set_word_vectors(batch_tree)
      for i in range(1,Config.agent_level+1):
        self.agent_levels[i].realize_vectors(batch_tree.level_nodes[i])



      #print("there")

    # def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
    #     for name, param in self.named_parameters(recurse=recurse):
    #         if 'embedding' in name and not name.startswith('levels.0.'):  # Ignore embedding above base level
    #             continue
    #
    #         yield param
    #
    # def convert_vectors_to_indices(self, vectors, level):
    #     vectors = vectors.detach().numpy()
    #     unique_vectors = np.unique(vectors, axis=0)
    #
    #     self.levels[level].set_embedding(torch.tensor(unique_vectors))
    #
    #     inputs = np.array([np.argwhere((vec == unique_vectors).all(1))[0][0] for vec in vectors])
    #     inputs += 4  # Make room for the pad, mask, etc tokens
    #
    #     self.losses[level]['mlm'].num_tokens = len(unique_vectors) + 4
    #
    #     return inputs
    #
    # def create_inputs_mask(self, inputs, level):
    #     # Add EoS Token
    #     # TODO - Reference the EoS token straight from the tokenizer so that it will be dynamic
    #     inputs = [seq + [2] for seq in inputs]
    #
    #     mask = [[0] * len(seq) + [1] * (self.max_seq_length[level] - len(seq)) for seq in inputs]
    #     inputs = [seq + [0] * (self.max_seq_length[level] - len(seq)) for seq in inputs]
    #     inputs = torch.tensor(inputs)
    #     mask = torch.tensor(mask)
    #
    #     return inputs, mask
    #
    # def fit(self, inputs, level=None):
    #     if level is None:
    #         level = find_level(inputs[0])  # Taking [0] because it's a batch
    #
    #     if level > 0:
    #         lengths = [len(seq) for seq in inputs]
    #         inputs = [item for seq in inputs for item in seq]
    #         vectors, loss_m, loss_c, loss_r = self.fit(inputs, level=level - 1)
    #
    #         inputs = self.convert_vectors_to_indices(vectors, level)
    #         inputs = np.split(inputs, lengths)
    #
    #         # TODO - Identify why this happens (there is an empty sequence in there for some reason)
    #         inputs = [seq.tolist() for seq in inputs if len(seq) > 0]
    #     else:
    #         loss_m = []
    #         loss_c = []
    #         loss_r = []
    #
    #     inputs, mask = self.create_inputs_mask(inputs, level)
    #
    #     print('Level', level)
    #     mlm_loss = self.losses[level]['mlm'](inputs, mask)
    #     coherence_loss = self.losses[level]['coherence'](inputs, mask)
    #     reconstruct_loss = self.losses[level]['reconstruct'](inputs, mask)
    #
    #     print('mlm_loss', mlm_loss.item())
    #     print('coherence_loss', coherence_loss.item())
    #     print('reconstruct_loss', reconstruct_loss.item())
    #
    #     loss_m.append(mlm_loss)
    #     loss_c.append(coherence_loss)
    #     loss_r.append(reconstruct_loss)
    #
    #     vectors = self.levels[level].encode(inputs, mask)
    #
    #     return vectors, loss_m, loss_c, loss_r
    #
    # def forward(self, src, mask):
    #     raise NotImplementedError
    #
    # def encode(self, inputs, level=None):
    #     if level is None:
    #         level = find_level(inputs[0])  # Taking [0] because it's a batch
    #
    #     if level > 0:
    #         lengths = [len(seq) for seq in inputs]
    #         inputs = [item for seq in inputs for item in seq]
    #         vectors = self.encode(inputs, level=level - 1)
    #
    #         inputs = self.convert_vectors_to_indices(vectors, level)
    #         inputs = np.split(inputs, lengths)
    #
    #         # TODO - Identify why this happens (there is an empty sequence in there for some reason)
    #         inputs = [seq.tolist() for seq in inputs if len(seq) > 0]
    #
    #     inputs, mask = self.create_inputs_mask(inputs, level)
    #     return self.levels[level].encode(inputs, mask)
    #
    # # return_word_vectors is temporary now while debugging
    # def debug_decode(self, vectors, level=None, return_word_vectors=False):
    #     if level is None:
    #         # TODO - Check if can use this
    #         # level = find_level(vectors)
    #         level = len(vectors.size()) - 2
    #
    #     if level == 0:
    #         # Don't reshape when doing word level eval
    #         # TODO - Make this dynamic in the future and not so rigid
    #         need_reshape = len(vectors.size()) > 2
    #         if need_reshape:
    #             original_shape = vectors.size()
    #             shape = (vectors.size(0) * vectors.size(1), vectors.size(2))
    #             vectors = vectors.reshape(shape)
    #     decoded = self.levels[level].debug_decode(vectors)
    #     if level == 0:
    #         if need_reshape:
    #             decoded = decoded.reshape((original_shape[0], original_shape[1], decoded.size(1)))
    #         return decoded
    #
    #     if level == 1 and return_word_vectors:
    #         return decoded
    #
    #     return self.debug_decode(decoded, level=level - 1)
