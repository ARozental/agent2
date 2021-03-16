from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker
import torch.nn as nn
import torch
from src.config import Config


class AgentLevel(nn.Module):
    def __init__(self, level):
        super().__init__()

        self.level = level
        self.encoder = Encoder(level)
        self.encoder_transform = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level], bias=False)
        self.decoder = Decoder(level)
        self.compressor = Compressor(level)
        self.decompressor = Decompressor(level)
        self.coherence_checker = CoherenceChecker(Config.vector_sizes[level + 1])

        # doesn't really requires_grad, it is here for debug
        self.pad_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)

        # TODO - Initialize right (not uniform)
        self.eos_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)

        self.join_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)
        self.mask_vector = torch.rand(Config.vector_sizes[level], requires_grad=True)

    # these functions change the modes
    def get_children(self, node_batch, embedding_matrix=None):
        if self.level == 0:  # words => get token vectors
            lookup_ids = torch.LongTensor([x.get_padded_word_tokens() for x in node_batch])
            mask = lookup_ids == Config.pad_token_id
            eos_positions = (lookup_ids == Config.eos_token_id).float()
            matrices = torch.index_select(embedding_matrix, 0, lookup_ids.view(-1))
            matrices = matrices.view(
                len(node_batch),
                Config.sequence_lengths[self.level],
                Config.vector_sizes[self.level]
            )
            labels = torch.LongTensor([x.get_padded_word_tokens() for x in node_batch])

            return self.level, matrices, mask, eos_positions, embedding_matrix, labels
        elif self.level == 1:  # sentences => get word vectors
            masks = []
            eos_positions = []
            for n in node_batch:
                mask = ([False for c in n.children] + [False] + ([True] * Config.sequence_lengths[1]))[0:Config.sequence_lengths[1]]
                eos_position = ([0.0 for c in n.children] + [1.0] + ([0.0] * Config.sequence_lengths[1]))[0:Config.sequence_lengths[1]]
                masks.append(mask)
                eos_positions.append(eos_position)

            mask = torch.tensor(masks)
            eos_positions = torch.tensor(eos_positions)

            # +3 for pad, eos and join, also need to change the matrix here and also handle Join-Tokens
            # TODO - todo: +2 if join is not in config
            lookup_ids = [list(map(lambda x: x.distinct_lookup_id + 2 + int(Config.join_texts), n.children)) for n in
                          node_batch]

            # 0 for pad, 1 for eos, because that was the concate order with the word embedding matrix
            lookup_ids = torch.LongTensor(
                [(x + [0] + ([1] * Config.sequence_lengths[1]))[0:Config.sequence_lengths[1]] for x in
                 lookup_ids])
            lookup_ids = torch.LongTensor(lookup_ids).view(-1)
            matrices = torch.index_select(embedding_matrix, 0, lookup_ids)
            matrices = matrices.view(
                len(node_batch),
                Config.sequence_lengths[1],
                Config.vector_sizes[1]
            )
            labels = lookup_ids.view(len(node_batch), Config.sequence_lengths[1])

            return self.level, matrices, mask, eos_positions, embedding_matrix, labels

        else:
            matrices = []
            masks = []
            eos_positions = []
            all_children = []
            all_ids = []

            for n in node_batch:
                children = n.children
                mask = ([False for c in children] + [False] + ([True] * Config.sequence_lengths[self.level]))[
                       0:Config.sequence_lengths[self.level]]
                eos_position = ([0.0 for c in children] + [1.0] + ([0.0] * Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
                ids = ([c.id for c in children] + [0] + ([1] * Config.sequence_lengths[self.level]))[
                      0:Config.sequence_lengths[self.level]]
                all_children.extend(children)
                masks.append(mask)
                eos_positions.append(eos_position)
                all_ids.append(ids)
                matrix = ([c.vector for c in children] + [self.eos_vector] + (
                        [self.pad_vector] * Config.sequence_lengths[self.level]))[
                         0:Config.sequence_lengths[self.level]]
                matrix = torch.stack(matrix)
                matrices.append(matrix)
            mask = torch.tensor(masks)
            eos_positions = torch.tensor(eos_positions)

            # [sentences in node_batch, max words in sentence, word vec size] #after padding
            matrices = torch.stack(matrices)
            all_children_ids = [c.id for c in all_children]

            # 0 is saved for EoS,
            id_to_place = dict(zip(all_children_ids, range(1, matrices.shape[0] * matrices.shape[1] + 1)))

            def id_to_place2(i):
                return i if i <= 1 else id_to_place[i]

            embedding = [c.vector for c in all_children]
            labels = torch.tensor([[id_to_place2(i) for i in x] for x in all_ids])
            embedding = torch.stack([self.eos_vector] + embedding)

            return self.level, matrices, mask, eos_positions, embedding, labels

        # labels = torch.tensor([x.get_padded_word_tokens() for x in node_batch])
        # batch_mask = torch.tensor([([False]*len(n.tokens)+[True]*Config.sequence_lengths[0])[0:Config.sequence_lengths[0]] for n in node_batch])
        # all_char_matrices = torch.index_select(local_char_embedding_matrix, 0, lookup_ids)  #[words_in_batch,max_chars_in_word,char_vector_size]
        # mlm = calc_mlm_loss(self.agent_levels[0],all_char_matrices,batch_mask,self.char_embedding_layer.weight,labels)
        # # labels [batch,seq_length,1] 1=>id in embedding matrix

        return self.level, matrices, mask, embedding_matrix, labels

    def realize_vectors(self, node_batch):
        # todo: realize join vectors here
        matrices = []
        masks = []
        eos_positions = []
        for n in node_batch:
            mask = ([False for c in n.children] + [False] + ([True] * Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
            eos_position = ([0.0 for c in n.children] + [1.0] + ([0.0] * Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
            matrix = ([c.vector for c in n.children] + [self.eos_vector] + (
                    [self.pad_vector] * Config.sequence_lengths[self.level]))[0:Config.sequence_lengths[self.level]]
            matrix = torch.stack(matrix)
            matrices.append(matrix)
            masks.append(mask)
            eos_positions.append(eos_position)

        matrices = torch.stack(matrices)  # [sentences in node_batch, max words in sentence, word vec size]

        mask = torch.tensor(masks)
        eos_positions = torch.tensor(eos_positions)
        vectors = self.compressor(self.encoder(matrices, mask,eos_positions), mask)
        [n.set_vector(v) for n, v in zip(node_batch, vectors)]
