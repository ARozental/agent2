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
        self.coherence_checker = CoherenceChecker(level)
        self.generator = Generator(Config.vector_sizes[level + 1])
        self.discriminator = Discriminator(Config.vector_sizes[level + 1])
        self.cnn_discriminator = CnnDiscriminator(Config.vector_sizes[level], Config.sequence_lengths[level])

        self.previous_level = None
        self.LayerNorm = nn.LayerNorm(Config.vector_sizes[level])

        if self.level == 0:
            self.token_bias = nn.Parameter(torch.zeros(num_letters, requires_grad=True))
        else:
            self.token_bias = None

        self.classifier1w = nn.Parameter(2.2 * torch.ones(1, requires_grad=True))  # with sane init
        self.classifier1b = nn.Parameter((-1.1) * torch.ones(1, requires_grad=True))  # with sane init

        if Config.join_texts:
            self.join_classifier_w = nn.Parameter(2.2 * torch.ones(1, requires_grad=True))  # with sane init
            self.join_classifier_b = nn.Parameter((-1.1) * torch.ones(1, requires_grad=True))  # with sane init

        # TODO - Initialize right (not uniform?)
        # TODO - Can we remove the join token for the base level??
        self.eos_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.join_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.mask_vector = nn.Parameter(torch.rand(Config.vector_sizes[level], requires_grad=True))
        self.pad_vector = nn.Parameter(torch.zeros(Config.vector_sizes[level]), requires_grad=False)

    def eos_classifier1(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.classifier1w * torch.sign(self.classifier1w)) + self.classifier1b

    def join_classifier(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.join_classifier_w * torch.sign(self.join_classifier_w)) + self.join_classifier_b

    def get_children(self, node_batch, char_embedding=None):
        max_length = Config.sequence_lengths[self.level]

        if self.level == 0:  # words => get token vectors
            lookup_ids = torch.LongTensor([node.get_padded_word_tokens() for node in node_batch]).to(Config.device)
            real_positions = (lookup_ids != Config.pad_token_id).float()
            eos_positions = (lookup_ids == Config.eos_token_id).float()
            matrices = torch.index_select(char_embedding, 0, lookup_ids.view(-1))
            matrices = matrices.view(
                lookup_ids.size(0),
                Config.sequence_lengths[self.level],
                Config.vector_sizes[self.level]
            )

            # lookup_ids is also labels
            return matrices, real_positions, eos_positions, None, char_embedding, lookup_ids
        else:
            if self.level == 1:
                id_name = 'distinct_lookup_id'
            else:
                id_name = 'id'

            add_value = 2 + int(Config.join_texts)

            masks = [[False] * (len(node.children) + 1) for node in node_batch]
            eos_positions = [[0.0] * len(node.children) + [1.0] for node in node_batch]
            join_positions = [[1 if child.is_join() else 0 for child in node.children] for node in node_batch]
            all_ids = [[getattr(child, id_name) + add_value for child in node.children] + [0] for node in node_batch]
            matrices = [[child.vector for child in node.children] + [self.eos_vector] for node in node_batch]

            masks = [item + [True] * (max_length - len(item)) for item in masks]
            eos_positions = [item + [0.0] * (max_length - len(item)) for item in eos_positions]
            join_positions = [item + [0.0] * (max_length - len(item)) for item in join_positions]
            all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]
            matrices = [torch.stack(item + [self.pad_vector] * (max_length - len(item))) for item in matrices]

            # These arrays may be longer than the max_length because they assume that the EoS token exists
            # But some sentences, etc don't have the EoS at all if they were split
            masks = [item[:max_length] for item in masks]
            eos_positions = [item[:max_length] for item in eos_positions]
            all_ids = [item[:max_length] for item in all_ids]
            matrices = [item[:max_length] for item in matrices]

            all_children = [child for node in node_batch for child in node.children]
            mask = torch.tensor(masks).to(Config.device)
            eos_positions = torch.tensor(eos_positions).to(Config.device)
            join_positions = torch.tensor(join_positions).to(Config.device)

            # [sentences in node_batch, max words in sentence, word vec size]
            matrices = torch.stack(matrices).to(Config.device)

            # 0 for EoS, 1 for pad, 2 is for join (so start counting at 1 or 2)
            all_children_ids = [i for x in all_ids for i in x if i > add_value - 1]
            all_children_ids = list(dict.fromkeys(all_children_ids))
            id_to_place = {child_id: i + add_value for i, child_id in enumerate(all_children_ids)}
            id_to_place[0] = 0
            id_to_place[1] = 1
            if Config.join_texts:
                id_to_place[2] = 2

            labels = torch.tensor([[id_to_place[i] for i in x] for x in all_ids]).to(Config.device)

            if self.level == 1:
                id_to_vec = {getattr(child, id_name): child.vector for node in node_batch for child in node.children}
                # Subtract add_value to get to the real id
                child_vectors = [id_to_vec[child_id - add_value] for child_id in all_children_ids]
            else:
                child_vectors = [child.vector for child in all_children]

            embedding = torch.stack(
                [self.eos_vector] +
                [self.pad_vector] +
                ([self.join_vector] if Config.join_texts else []) +
                child_vectors
            ).to(Config.device)

            real_positions = (1 - mask.float())
            return matrices, real_positions, eos_positions, join_positions, embedding, labels

    def realize_vectors(self, node_batch):
        # todo: realize join vectors here
        max_length = Config.sequence_lengths[self.level]
        masks = [[False] * (len(node.children) + 1) for node in node_batch]
        eos_positions = [[0.0] * len(node.children) + [1.0] for node in node_batch]
        matrices = [[child.vector for child in node.children] + [self.eos_vector] for node in node_batch]

        masks = [item + [True] * (max_length - len(item)) for item in masks]
        eos_positions = [item + [0.0] * (max_length - len(item)) for item in eos_positions]
        matrices = [torch.stack(item + [self.pad_vector] * (max_length - len(item))) for item in matrices]

        # These arrays may be longer than the max_length because they assume that the EoS token exists
        # But some sentences, etc don't have the EoS at all if they were split
        masks = [item[:max_length] for item in masks]
        eos_positions = [item[:max_length] for item in eos_positions]
        matrices = [item[:max_length] for item in matrices]

        mask = torch.tensor(masks).to(Config.device)
        real_positions = (1 - mask.float())
        eos_positions = torch.tensor(eos_positions).to(Config.device)

        # [sentences_in_node_batch, max_words_in sentence, word vec size]
        matrices = torch.stack(matrices).to(Config.device)

        vectors = self.compressor(self.encoder(matrices, real_positions, eos_positions), mask)
        [n.set_vector(v) for n, v in zip(node_batch, vectors)]

    def vecs_to_children_vecs(self, vecs):
        # 0th-element is the eos token; X is a vector
        decompressed = self.decompressor(vecs)

        # Find EoS token and num_tokens
        eos_vector = self.eos_vector.unsqueeze(0).unsqueeze(0)
        eos_dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm())
        eos_dot = eos_dot.sum(dim=-1, keepdim=True)
        eos_logits = self.eos_classifier1(eos_dot).squeeze(-1)

        eos_softmax = F.softmax(eos_logits, dim=1).max(dim=-1).values
        eos_sigmoid = torch.sigmoid(eos_logits).max(dim=-1).values
        is_eos = torch.logical_and(eos_softmax > 0.5, eos_sigmoid > 0.5)
        num_tokens = torch.where(is_eos, torch.argmax(eos_logits, dim=-1), eos_logits.size(1))

        range_matrix = torch.arange(eos_logits.size(1)).repeat(eos_logits.size(0), 1).to(Config.device)
        mask = range_matrix > num_tokens.unsqueeze(-1)
        real_positions = (1 - mask.float())
        eos_positions = (range_matrix == num_tokens.unsqueeze(-1)).long()

        # Find join token
        if Config.join_texts and self.level > 0:
            join_vector = self.join_vector.unsqueeze(0).unsqueeze(0)
            join_dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * join_vector / join_vector.norm())
            join_dot = join_dot.sum(dim=-1, keepdim=True)
            join_logits = self.join_classifier(join_dot).squeeze(-1)

            is_join = torch.sigmoid(join_logits) > 0.5

        post_decoder = self.decoder(decompressed, real_positions, eos_positions)

        # There can be a word that has only the EoS token so words need at least one token
        # But for all other levels we can assume one less token
        if self.level == 0:
            num_tokens += 1

        children_vectors = [decoded[:num] for num, decoded in zip(num_tokens, post_decoder)]
        if Config.join_texts and self.level > 0:
            children_vectors = [[vector if not j else None for vector, j in zip(child, joins)] for child, joins in
                                zip(children_vectors, is_join)]

        return children_vectors, is_eos, post_decoder, real_positions
