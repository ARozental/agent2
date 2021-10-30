from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker, Generator, Discriminator, CnnDiscriminator, \
    Pndb
from src.losses.eos import decompressed_to_cdot, cdot_to_probs, calc_eos_loss
from src.config import Config
from src.utils import group_by_root, distinct
import torch.nn.functional as F
import torch.nn as nn
import torch
import random
from src.pre_processing import TreeTokenizer


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
        self.generator = None# Generator(Config.vector_sizes[level + 1])
        self.discriminator = None# Discriminator(Config.vector_sizes[level + 1])
        self.cnn_discriminator = None# CnnDiscriminator(Config.vector_sizes[level], Config.sequence_lengths[level])

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

        self.pndb = None
        if self.level == 1 and (Config.use_pndb1 is not None or Config.use_pndb2 is not None):
            self.pndb = Pndb()

    def eos_classifier1(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.classifier1w * torch.sign(self.classifier1w)) + self.classifier1b

    def join_classifier(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.join_classifier_w * torch.sign(self.join_classifier_w)) + self.join_classifier_b

    def get_children(self, node_batch, inputs, embedding=None, word_embedding0=None, batch_index=0, debug=False):
        max_length = Config.sequence_lengths[self.level]
        inputs = inputs[str(self.level) + '-' + str(batch_index)]  # Simplifying the inputs here
        if self.level == 0:  # words => get token vectors
            lookup_ids = inputs['lookup_ids']

            real_positions = (lookup_ids != Config.pad_token_id).float()
            eos_positions = (lookup_ids == Config.eos_token_id).float()
            matrices = torch.index_select(embedding, 0, lookup_ids.view(-1))
            matrices = matrices.view(lookup_ids.size(0), Config.sequence_lengths[self.level],
                                     Config.vector_sizes[self.level])

            word_lookup_ids = inputs['word_lookup_ids']
            vectors = torch.index_select(word_embedding0, 0, word_lookup_ids)
            # vectors = self.compressor(self.encoder(matrices, real_positions, eos_positions), real_positions) #same but less efficient

            # create_coherence_matrixes
            with torch.no_grad():
                random_ids = inputs['random_ids']
                random_matrices = torch.index_select(embedding, 0, random_ids.view(-1))  # .detach()
                random_matrices = random_matrices.view(lookup_ids.size(0), Config.sequence_lengths[self.level],
                                                       Config.vector_sizes[self.level])

            # lookup_ids is also labels
            return matrices, real_positions, eos_positions, None, embedding, lookup_ids, vectors, 0, None, None, random_matrices
        elif self.level == 1:
            num_dummy = 0

            all_ids = inputs['all_ids']
            with torch.no_grad():
                random_ids = inputs['random_ids']
                random_matrices = torch.index_select(embedding, 0, random_ids.flatten())
                random_matrices = random_matrices.reshape(
                    (random_ids.size(0), random_ids.size(1), random_matrices.size(1)))

            mask = (all_ids == Config.pad_token_id).bool()
            # TODO - Which is faster? int() or float()?
            eos_positions = (all_ids == 0).int()  # 0 is for EOS
            join_positions = (all_ids == 2).int()  # 2 is for JOIN

            matrices = torch.index_select(embedding, 0, all_ids.flatten())
            matrices = matrices.reshape((all_ids.size(0), all_ids.size(1), matrices.size(1)))

            labels = all_ids

            real_positions = (1 - mask.float())
            vectors = self.compressor(self.encoder(matrices, real_positions, eos_positions), real_positions)
            if debug or Config.agent_level > 1:
                [n.set_vector(v.detach()) for n, v in zip(node_batch, vectors)]

            # pndb
            A1s, pndb_lookup_ids = None, None
            if Config.use_pndb1:
                # continuous ids verify, todo: if debug
                # md5s = [n.root_md5 for n in node_batch]
                # seen = set([])
                # for i in range(1,len(md5s)):
                #   if md5s[i] in seen and md5s[i]!=md5s[i-1]:
                #     raise("WTF pndb") #happened after 4000+ batches
                #   seen.add(md5s[i])

                current_root_md5 = node_batch[0].root_md5
                start_index = 0
                end_index = 0
                A1s = []
                pndb_lookup_ids = []
                top_doc_id = 0
                for n in node_batch:
                    if n.root_md5 == current_root_md5:
                        end_index += 1
                    else:
                        A1s.append(self.pndb.create_A_matrix(matrices[start_index:end_index],
                                                             real_positions[start_index:end_index]))
                        start_index = end_index
                        end_index += 1
                        current_root_md5 = n.root_md5
                        top_doc_id += 1
                    pndb_lookup_ids.append(top_doc_id)
                A1s.append(
                    self.pndb.create_A_matrix(matrices[start_index:end_index], real_positions[start_index:end_index]))
                A1s = torch.stack(A1s)
                pndb_lookup_ids = torch.tensor(pndb_lookup_ids, device=matrices.device)

            return matrices, real_positions, eos_positions, join_positions, embedding, labels, vectors, num_dummy, A1s, pndb_lookup_ids, random_matrices
        else:
            add_value = 2 + int(Config.join_texts)
            num_dummy = 0
            if Config.use_tpu:
                raise NotImplementedError("WTF level 2 TPU is not implemented")

            id_to_index = {v: i for i, v in
                           enumerate([c.id for node in node_batch for c in node.children if not (c.is_join())])}

            all_ids = [[2 if child.is_join() else id_to_index[child.id] + add_value for child in node.children] + [0]
                       for node in node_batch]  # [0] is EOS, 2 is JOIN
            all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]  # 1 is PAD

            # This array may be longer than the max_length because it assumes that the EoS token exists
            # But some sentences, etc don't have the EoS at all if they were split
            all_ids = [item[:max_length] for item in all_ids]

            # [sentences in node_batch, max words in sentence, word vec size]
            all_ids = torch.tensor(all_ids, device=Config.device, dtype=torch.long)

            # embedding:
            all_vectors = [c.vector for node in node_batch for c in node.children if not (c.is_join())]
            embedding = torch.stack([self.eos_vector, self.pad_vector, self.join_vector] + all_vectors)

            mask = (all_ids == Config.pad_token_id).bool()
            # TODO - Which is faster? int() or float()?
            eos_positions = (all_ids == 0).int()  # 0 is for EOS
            join_positions = (all_ids == 2).int()  # 2 is for JOIN

            matrices = [[child.vector for child in node.children] + [self.eos_vector] for node in node_batch]
            matrices = [torch.stack(item + [self.pad_vector] * (max_length - len(item))) for item in matrices]
            matrices = [item[:max_length] for item in matrices]
            matrices = torch.stack(matrices)

            labels = all_ids

            real_positions = (1 - mask.float())
            vectors = self.compressor(self.encoder(matrices, real_positions, eos_positions), real_positions)
            if debug:
                [n.set_vector(v.detach()) for n, v in zip(node_batch, vectors)]
            return matrices, real_positions, eos_positions, join_positions, embedding, labels, vectors, num_dummy, None, None, None

    def vecs_to_children_vecs(self, vecs, A1s, pndb_lookup_ids,embedding_matrix):
        "should have 2 modes, get reconstructed children_vecs and get selected children_vecs from embedding_matrix"
        decompressed = self.decompressor(vecs)

        # todo: use PNDB2 if exists
        # if Config.use_pndb2 is not None and self.level == 1:
        #  decompressed = self.pndb.old_get_data_from_A_matrix(pndb2, decompressed)

        batch, seq_length, _ = decompressed.shape
        _, projected_eos_positions = calc_eos_loss(self, decompressed,
                                                   torch.zeros(batch, seq_length, device=decompressed.device))
        real_positions_for_mask = (1 - torch.cumsum(projected_eos_positions, dim=1))
        post_decoder = self.decoder(decompressed, real_positions_for_mask, None)
        if Config.use_pndb1 is not None and self.level == 1:
            # post_decoder = pndb.old_get_data_from_A_matrix(pndb.create_A_matrix(matrices, real_positions), post_decoder)
            post_decoder = self.pndb.get_data_from_A_matrix(A1s, pndb_lookup_ids, post_decoder)

        _, eos_mask = calc_eos_loss(self, post_decoder, torch.zeros(batch, seq_length, device=decompressed.device))

        eos_mask_max = eos_mask.max(dim=-1).values
        is_eos = eos_mask_max > 0.3
        num_tokens = torch.where(eos_mask_max > 0.4, torch.argmax(eos_mask, dim=-1),
                                 eos_mask.size(1))  # todo fix fails to decode when torch.argmax(eos_mask, dim=-1) is 0

        range_matrix = torch.arange(eos_mask.size(1), device=decompressed.device).repeat(eos_mask.size(0), 1)

        mask = range_matrix > num_tokens.unsqueeze(-1)
        real_positions = (1 - mask.float())

        # Find join token
        if Config.join_texts and self.level > 0:
            join_vector = self.join_vector.unsqueeze(0).unsqueeze(0)
            join_dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * join_vector / join_vector.norm())
            join_dot = join_dot.sum(dim=-1, keepdim=True)
            join_logits = self.join_classifier(join_dot).squeeze(-1)

            is_join = torch.sigmoid(join_logits) > 0.5

        # There can be a word that has only the EoS token so words need at least one token
        # But for all other levels we can assume one less token
        if self.level == 0:
            num_tokens += 1

        children_vectors = [decoded[:num] for num, decoded in zip(num_tokens, post_decoder)]
        if Config.join_texts and self.level > 0:
            children_vectors = [[vector if not j else None for vector, j in zip(child, joins)] for child, joins in
                                zip(children_vectors, is_join)]

        children_vectors_from_embedding = children_vectors
        if embedding_matrix is not None:
          children_vectors_from_embedding = []
          logits = torch.matmul(post_decoder, torch.transpose(embedding_matrix, 0, 1))
          all_lookup_ids = torch.argmax(logits, dim=2)

          all_word_vectors = [torch.index_select(embedding_matrix, 0,lookup_ids) for lookup_ids in all_lookup_ids]
          for i in range(len(all_word_vectors)):
            children_vectors_from_embedding.append([x for x in all_word_vectors[i][0:num_tokens[i]]])

        return children_vectors,children_vectors_from_embedding, is_eos, post_decoder, real_positions
