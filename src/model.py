from src.agent import AgentLevel
import torch.nn as nn
import torch
from src.config import Config
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import calc_reconstruction_loss


class AgentModel(nn.Module):
    def __init__(self, tree_tokenizer):
        super().__init__()
        self.tree_tokenizer = tree_tokenizer
        self.agent_levels = nn.ModuleList()
        for i in range(Config.agent_level + 2):
            agent_level = AgentLevel(i)
            self.agent_levels.append(agent_level)

        self.char_embedding_layer = nn.Embedding(len(tree_tokenizer.letter_tokenizer.keys()), Config.vector_sizes[0])

    def set_word_vectors(self, batch_tree):
        node_batch = batch_tree.level_nodes[0]
        local_char_embedding_tokens = torch.LongTensor(batch_tree.distinct_word_embedding_tokens)
        mask = local_char_embedding_tokens == Config.pad_token_id  # True => position to mask
        local_char_embedding_matrix = self.char_embedding_layer(local_char_embedding_tokens)
        word_embedding_matrix = self.agent_levels[0].compressor(
            self.agent_levels[0].encoder(local_char_embedding_matrix,
                                         mask))  # [distinct_words_in_batch,word_vector_size]

        if Config.join_texts:
            special_vectors = torch.stack([
                self.agent_levels[1].eos_vector,
                self.agent_levels[1].pad_vector,
                self.agent_levels[1].join_vector,
            ])  # {0: eos, 1:pad, 2:join}
            word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)
            lookup_ids = torch.LongTensor([x.distinct_lookup_id for x in node_batch]) + 3
        else:
            special_vectors = torch.stack([
                self.agent_levels[1].eos_vector,
                self.agent_levels[1].pad_vector,
            ])  # {0: eos, 1:pad}
            word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)
            lookup_ids = torch.LongTensor([x.distinct_lookup_id for x in node_batch]) + 2

        all_word_vectors = torch.index_select(word_embedding_matrix, 0, lookup_ids)  # [words_in_batch,word_vector_size]
        [n.set_vector(v) for n, v in zip(node_batch, all_word_vectors)]
        return word_embedding_matrix

    def set_text_vectors(self, batch_tree):
        word_embedding_matrix = self.set_word_vectors(batch_tree)
        for i in range(1, Config.agent_level + 1):
            self.agent_levels[i].realize_vectors(batch_tree.level_nodes[i])
        return word_embedding_matrix

    def forward(self, batch_tree):
        word_embedding_matrix = self.set_text_vectors(batch_tree)
        embedding_matrices = {0: self.char_embedding_layer.weight, 1: word_embedding_matrix}
        total_loss = 0
        loss_object = {}
        for i in range(Config.agent_level + 1):
            node_batch = batch_tree.level_nodes[i]  # currently all the nodes in the level
            level, matrices, mask, embedding_matrix, labels = self.agent_levels[i].get_children(node_batch,
                                                                                                embedding_matrices[
                                                                                                    i % 2])  # we only care about 0 and 1
            mlm_loss = calc_mlm_loss(self.agent_levels[i], matrices, mask, embedding_matrix, labels)
            coherence_loss = calc_coherence_loss(self.agent_levels[i], matrices, mask, embedding_matrix)
            vectors = torch.stack([n.vector for n in node_batch])
            reconstruction_loss = calc_reconstruction_loss(self.agent_levels[i], vectors, mask, embedding_matrix,
                                                           labels)
            total_loss += (mlm_loss.mean() + coherence_loss.mean() + reconstruction_loss.mean()).sum()
            loss_object[i] = {'m': mlm_loss.mean().item(), "c": coherence_loss.mean().item(),
                              "r": reconstruction_loss.mean().item()}

            [setattr(n, 'mlm_loss', l) for n, l in zip(node_batch, mlm_loss.tolist())]
            [setattr(n, 'coherence_loss', l) for n, l in zip(node_batch, coherence_loss.tolist())]
            [setattr(n, 'reconstruction_loss', l) for n, l in zip(node_batch, reconstruction_loss.tolist())]

        return loss_object, total_loss  # todo: make loss object

    def debug_decode(self, batch_tree):
        node_batch = batch_tree.level_nodes[0]
        local_char_embedding_tokens = torch.LongTensor(batch_tree.distinct_word_embedding_tokens)
        mask = local_char_embedding_tokens == Config.pad_token_id  # True => position to mask

        mask = mask[0].unsqueeze(0)
        vector = node_batch[0].vector.unsqueeze(0)
        output = self.agent_levels[0].decompressor(vector)
        output = self.agent_levels[0].decoder(output, mask)

        # word_embedding_matrix = self.set_text_vectors(batch_tree)
        # embedding_matrices = {0: self.char_embedding_layer.weight, 1: word_embedding_matrix}
        # level, matrices, mask, embedding_matrix, labels = self.agent_levels[0].get_children(node_batch,
        #                                                                                     embedding_matrices[
        #                                                                                         0 % 2])

        output = torch.matmul(output, self.char_embedding_layer.weight.transpose(0, 1))
        output = torch.argmax(output, dim=2)

        # output = self.char_embedding_layer(output)
        return output
