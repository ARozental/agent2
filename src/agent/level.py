from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker, Generator, Discriminator, CnnDiscriminator
import torch.nn.functional as F
import torch.nn as nn
import torch
from src.config import Config


class AgentLevel(nn.Module):
    def __init__(self, level, num_letters):
        super().__init__()

        self.level = level
        self.encoder = Encoder(level)
        self.encoder_transform = nn.Linear(Config.vector_sizes[level], Config.vector_sizes[level], bias=False)
        self.decoder = Decoder(level)
        self.compressor = Compressor(level)
        self.decompressor = Decompressor(level)
        self.coherence_checker = CoherenceChecker(Config.vector_sizes[level + 1])
        self.generator = Generator(Config.vector_sizes[level + 1])
        self.discriminator = Discriminator(Config.vector_sizes[level + 1])
        self.cnn_discriminator = CnnDiscriminator(Config.vector_sizes[level], Config.sequence_lengths[level])

        if self.level == 0:
            self.token_bias = nn.Parameter(torch.zeros(num_letters, requires_grad=True))
        else:
            self.token_bias = None

        self.classifier1w = nn.Parameter(2.2 * torch.ones(1, requires_grad=True))  # with sane init
        self.classifier1b = nn.Parameter((-1.1) * torch.ones(1, requires_grad=True))  # with sane init
        self.classifier1act = nn.ELU()

        # TODO - Initialize right (not uniform?)
        self.eos_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.join_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.mask_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.pad_vector = nn.Parameter(torch.zeros(Config.vector_sizes[level], requires_grad=True))  # True for debug

    def eos_classifier1(self, dot):
        # needed to make sure w1 can never be negative
        return self.classifier1act(dot * self.classifier1w.abs()) + self.classifier1b

    def get_children(self, node_batch, embedding_matrix=None):
        if self.level == 0:  # words => get token vectors
            lookup_ids = torch.LongTensor([node.get_padded_word_tokens() for node in node_batch])
            lookup_ids = lookup_ids.to(Config.device)
            mask = lookup_ids == Config.pad_token_id
            eos_positions = (lookup_ids == Config.eos_token_id).float()
            matrices = torch.index_select(embedding_matrix, 0, lookup_ids.view(-1))
            matrices = matrices.view(
                len(node_batch),
                Config.sequence_lengths[self.level],
                Config.vector_sizes[self.level]
            )

            # lookup_ids is also labels
            return matrices, mask, eos_positions, embedding_matrix, lookup_ids
        elif self.level == 1:  # sentences => get word vectors
            max_length = Config.sequence_lengths[1]
            masks = [[False] * (len(node.children) + 1) for node in node_batch]
            eos_positions = [[0.0] * len(node.children) + [1.0] for node in node_batch]

            masks = [item + [True] * (max_length - len(item)) for item in masks]
            eos_positions = [item + [0.0] * (max_length - len(item)) for item in eos_positions]

            mask = torch.tensor(masks).to(Config.device)
            eos_positions = torch.tensor(eos_positions).to(Config.device)

            # +3 for pad, eos and join, also need to change the matrix here and also handle Join-Tokens
            # TODO - todo: +2 if join is not in config
            add_value = 2 + int(Config.join_texts)
            lookup_ids = [[child.distinct_lookup_id + add_value for child in node.children] for node in node_batch]
            lookup_ids = torch.LongTensor([item + [0] + [1] * (max_length - (len(item) + 1)) for item in lookup_ids])
            lookup_ids = lookup_ids.to(Config.device).view(-1)
            matrices = torch.index_select(embedding_matrix, 0, lookup_ids)
            matrices = matrices.view(
                len(node_batch),
                Config.sequence_lengths[1],
                Config.vector_sizes[1]
            )
            labels = lookup_ids.view(len(node_batch), Config.sequence_lengths[1])

            return matrices, mask, eos_positions, embedding_matrix, labels
        else:
            max_length = Config.sequence_lengths[self.level]
            masks = [[False] * (len(node.children) + 1) for node in node_batch]
            eos_positions = [[0.0] * len(node.children) + [1.0] for node in node_batch]
            all_ids = [[child.id for child in node.children] + [0] for node in node_batch]
            matrices = [[child.vector for child in node.children] + [self.eos_vector] for node in node_batch]

            masks = [item + [True] * (max_length - len(item)) for item in masks]
            eos_positions = [item + [0.0] * (max_length - len(item)) for item in eos_positions]
            all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]
            matrices = [torch.stack(item + [self.pad_vector] * (max_length - len(item))) for item in matrices]

            all_children = [child for node in node_batch for child in node.children]

            mask = torch.tensor(masks).to(Config.device)
            eos_positions = torch.tensor(eos_positions).to(Config.device)

            # [sentences in node_batch, max words in sentence, word vec size] #after padding
            matrices = torch.stack(matrices).to(Config.device)
            # all_children_ids = [c.id for c in all_children]
            all_children_ids = [i for x in all_ids for i in x if i > 1]  #

            # 0 is saved for EoS,
            id_to_place = dict(zip(all_children_ids, range(1, matrices.shape[0] * matrices.shape[1] + 1)))

            def id_to_place2(i):
                return i if i <= 1 else id_to_place[i]

            embedding = [c.vector for c in all_children]
            labels = torch.tensor([[id_to_place2(i) for i in x] for x in all_ids]).to(Config.device)
            embedding = torch.stack([self.eos_vector] + embedding).to(Config.device)

            return matrices, mask, eos_positions, embedding, labels

    def realize_vectors(self, node_batch):
        # todo: realize join vectors here
        max_length = Config.sequence_lengths[self.level]
        masks = [[False] * (len(node.children) + 1) for node in node_batch]
        eos_positions = [[0.0] * len(node.children) + [1.0] for node in node_batch]
        matrices = [[child.vector for child in node.children] + [self.eos_vector] for node in node_batch]

        masks = [item + [True] * (max_length - len(item)) for item in masks]
        eos_positions = [item + [0.0] * (max_length - len(item)) for item in eos_positions]
        matrices = [torch.stack(item + [self.pad_vector] * (max_length - len(item))) for item in matrices]

        mask = torch.tensor(masks).to(Config.device)
        eos_positions = torch.tensor(eos_positions).to(Config.device)

        # [sentences_in_node_batch, max_words_in sentence, word vec size]
        matrices = torch.stack(matrices).to(Config.device)

        vectors = self.compressor(self.encoder(matrices, mask, eos_positions), mask)
        [n.set_vector(v) for n, v in zip(node_batch, vectors)]

    def vecs_to_children_vecs(self, vecs):
        # 0th-element is the eos token; X is a vector
        decompressed = self.decompressor(vecs)

        eos_vector = self.eos_vector.unsqueeze(0).unsqueeze(0)
        dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm()).sum(dim=-1,
                                                                                                           keepdim=True)
        logits = self.eos_classifier1(dot).squeeze(-1)

        max_softmax = F.softmax(logits, dim=1).max(dim=-1).values
        max_sigmoid = torch.sigmoid(logits).max(dim=-1).values
        is_valid = torch.logical_and(max_softmax > 0.5, max_sigmoid > 0.5)
        num_tokens = torch.where(is_valid, torch.argmax(logits, dim=-1), logits.size(1))

        range_matrix = torch.arange(logits.size(1)).repeat(logits.size(0), 1).to(Config.device)
        mask = range_matrix > num_tokens.unsqueeze(-1)
        eos_positions = (range_matrix == num_tokens.unsqueeze(-1)).long()

        post_decoder = self.decoder(decompressed, mask, eos_positions)

        # There can be a word that has only the EoS token so words need at least one token
        # But for all other levels we can assume one less token
        if self.level == 0:
            num_tokens += 1

        children_vectors = [decoded[:num] for num, decoded in zip(num_tokens, post_decoder)]

        return children_vectors, post_decoder, mask

    def node_to_children_vecs(self, node):
        vecs = node.vector.unsqueeze(0)
        children_vecs_arr, _, _ = self.vecs_to_children_vecs(vecs)
        return children_vecs_arr[0]
