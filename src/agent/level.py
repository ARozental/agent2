from . import Compressor, Decompressor, Encoder, Decoder, CoherenceChecker, Generator, Discriminator, CnnDiscriminator, Pndb
from src.losses.eos import decompressed_to_cdot, cdot_to_probs, calc_eos_loss
from src.config import Config
from src.utils import group_by_root,distinct
import torch.nn.functional as F
import torch.nn as nn
import torch


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


        self.pndb = None
        if self.level==1 and (Config.use_pndb1 is not None or Config.use_pndb2 is not None):
            self.pndb = Pndb()


    def eos_classifier1(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.classifier1w * torch.sign(self.classifier1w)) + self.classifier1b

    def join_classifier(self, dot):
        # needed to make sure w1 can never be negative
        return F.elu(dot * self.join_classifier_w * torch.sign(self.join_classifier_w)) + self.join_classifier_b

    def get_children(self, node_batch, embedding=None, previous_vectors=None, debug=False):
        max_length = Config.sequence_lengths[self.level]

        if self.level == 0:  # words => get token vectors
            lookup_ids = torch.tensor([node.get_padded_word_tokens() for node in node_batch], dtype=torch.long,
                                      device=Config.device)
            real_positions = (lookup_ids != Config.pad_token_id).float()
            eos_positions = (lookup_ids == Config.eos_token_id).float()
            matrices = torch.index_select(embedding, 0, lookup_ids.view(-1))
            matrices = matrices.view(
                lookup_ids.size(0),
                Config.sequence_lengths[self.level],
                Config.vector_sizes[self.level]
            )

            # lookup_ids is also labels
            return matrices, real_positions, eos_positions, None, embedding, lookup_ids, None, 0, None, None
        elif self.level == 1:
            id_name = 'distinct_lookup_id'
            add_value = 2 + int(Config.join_texts)
            num_dummy = 0
            if Config.use_tpu: #add pad vectors to ensure constant word embedding matrix size
                total_possible = len(node_batch) * max_length
                extra_dummy = total_possible - embedding.size(0)
                embedding = torch.cat((embedding,torch.stack([self.pad_vector] * extra_dummy)), 0)

            all_ids = [[2 if child.is_join() else getattr(child, id_name) + add_value for child in node.children] + [0]
                       for node in node_batch]  # [0] is EOS, 2 is JOIN
            all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]  # 1 is PAD

            # This array may be longer than the max_length because it assumes that the EoS token exists
            # But some sentences, etc don't have the EoS at all if they were split
            all_ids = [item[:max_length] for item in all_ids]

            # [sentences in node_batch, max words in sentence, word vec size]
            all_ids = torch.tensor(all_ids, device=Config.device, dtype=torch.long)

            mask = (all_ids == Config.pad_token_id).bool()
            # TODO - Which is faster? int() or float()?
            eos_positions = (all_ids == 0).int()  # 0 is for EOS
            join_positions = (all_ids == 2).int()  # 2 is for JOIN

            matrices = torch.index_select(embedding, 0, all_ids.flatten())
            matrices = matrices.reshape((all_ids.size(0), all_ids.size(1), matrices.size(1)))

            labels = all_ids

            real_positions = (1 - mask.float())
            vectors = self.compressor(self.encoder(matrices, real_positions, eos_positions), mask)
            if debug:
                [n.set_vector(v) for n, v in zip(node_batch, vectors)]


            #pndb
            A1s, pndb_lookup_ids = None,None
            if Config.use_pndb1:
              #continuous ids verify, todo: if debug
              md5s = [n.root_md5 for n in node_batch]
              seen = set([])
              for i in range(1,len(md5s)):
                if md5s[i] in seen and md5s[i]!=md5s[i-1]:
                  raise("WTF pndb")
                seen.add(md5s[i])

              current_root_md5 = node_batch[0].root_md5
              start_index=0
              end_index = 0
              A1s=[]
              pndb_lookup_ids=[]
              top_doc_id = 0
              for n in node_batch:
                if n.root_md5 == current_root_md5:
                  end_index+=1
                else:
                  A1s.append(self.pndb.create_A_matrix(matrices[start_index:end_index], real_positions[start_index:end_index]))
                  start_index = end_index
                  end_index+=1
                  current_root_md5 = n.root_md5
                  top_doc_id+=1
                pndb_lookup_ids.append(top_doc_id)
              A1s.append(self.pndb.create_A_matrix(matrices[start_index:end_index], real_positions[start_index:end_index]))
              A1s = torch.stack(A1s)
              pndb_lookup_ids = torch.tensor(pndb_lookup_ids,device=Config.device)


            return matrices, real_positions, eos_positions, join_positions, embedding, labels, vectors, num_dummy,A1s,pndb_lookup_ids
        else:
            raise("WTF level 2 is not implemented")


    def vecs_to_children_vecs(self, vecs):
        decompressed = self.decompressor(vecs)
        batch, seq_length, _ = decompressed.shape
        _, projected_eos_positions = calc_eos_loss(self, decompressed,
                                                   torch.zeros(batch, seq_length, device=Config.device))
        real_positions_for_mask = (1 - torch.cumsum(projected_eos_positions, dim=1))
        post_decoder = self.decoder(decompressed, real_positions_for_mask, None)
        _, eos_mask = calc_eos_loss(self, post_decoder, torch.zeros(batch, seq_length, device=Config.device))

        eos_mask_max = eos_mask.max(dim=-1).values
        is_eos = eos_mask_max > 0.3
        num_tokens = torch.where(eos_mask_max > 0.3, torch.argmax(eos_mask, dim=-1),
                                 eos_mask.size(1))  # todo fix fails to decode when torch.argmax(eos_mask, dim=-1) is 0

        range_matrix = torch.arange(eos_mask.size(1), device=Config.device).repeat(eos_mask.size(0), 1)

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
        return children_vectors, is_eos, post_decoder, real_positions
