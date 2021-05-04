from src.agent import AgentLevel
from src.config import Config
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import make_reconstruction_loss_fn#,calc_reconstruction_loss, calc_reconstruction_loss_with_pndb
from src.losses.generation import calc_generation_loss
from src.pre_processing import Node, TreeTokenizer
from src.utils import iter_even_split
import torch.nn as nn
import torch
from src.agent.pndb import Pndb


class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_letters = len(TreeTokenizer.letter_tokenizer.keys())
        self.agent_levels = nn.ModuleList([AgentLevel(i, num_letters) for i in range(Config.agent_level + 1)])
        [setattr(self.agent_levels[i], "previous_level", self.agent_levels[i-1]) for i in range(1,Config.agent_level+1)]


        self.char_embedding_layer = nn.Embedding(num_letters, Config.vector_sizes[0])

        self.pndb = None
        if Config.use_pndb1 is not None or Config.use_pndb2 is not None:
            self.pndb = Pndb()

        self.reconstruction_fns = {k:make_reconstruction_loss_fn(k) for k in range(Config.agent_level + 1)}

    def set_word_vectors(self, node_batch):
        distinct_ids = list(dict.fromkeys([node.distinct_lookup_id for node in node_batch]))
        id_to_tokens = {node.distinct_lookup_id: node.get_padded_word_tokens() for node in node_batch}

        id_to_place = {node_id: i for i, node_id in enumerate(distinct_ids)}
        distinct_word_embedding_tokens = [id_to_tokens[node_id] for node_id in distinct_ids]
        local_char_embedding_tokens = torch.LongTensor(distinct_word_embedding_tokens).to(Config.device)
        real_positions = (local_char_embedding_tokens != Config.pad_token_id).float()
        eos_positions = local_char_embedding_tokens == Config.eos_token_id
        local_char_embedding_matrix = self.char_embedding_layer(local_char_embedding_tokens)

        # first encoder call
        word_embedding_matrix = self.agent_levels[0].compressor(
            self.agent_levels[0].encoder(local_char_embedding_matrix, real_positions, eos_positions.float()),
          real_positions)  # [distinct_words_in_batch,word_vector_size]

        special_vectors = torch.stack(
            [
                self.agent_levels[1].eos_vector,
                self.agent_levels[1].pad_vector,
            ] +
            ([self.agent_levels[1].join_vector] if Config.join_texts else [])
        ).to(Config.device)  # {0: eos, 1:pad, 2:join}
        word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)
        lookup_ids = torch.LongTensor([id_to_place[x.distinct_lookup_id] for x in node_batch]).to(Config.device)
        lookup_ids += 2 + int(Config.join_texts)

        all_word_vectors = torch.index_select(word_embedding_matrix, 0, lookup_ids)  # [words_in_batch,word_vector_size]
        [n.set_vector(v) for n, v in zip(node_batch, all_word_vectors)]

    def forward(self, batch_tree, generate=False, debug=False):
        total_g_loss, total_disc_loss, total_loss = 0, 0, 0
        loss_object = {}
        for level_num in range(Config.agent_level + 1):
            # All the nodes in this level (not including join tokens if on lowest level)
            real_nodes = [node for node in batch_tree.level_nodes[level_num] if level_num > 0 or not node.is_join()]

            # todo: pndb_map = {md5 => tensor}

            for batch_num, node_batch in enumerate(iter_even_split(real_nodes, Config.batch_sizes[level_num])):
                if level_num == 0:
                    self.set_word_vectors(node_batch)
                else:
                    self.agent_levels[level_num].realize_vectors(node_batch)

                matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels = self.agent_levels[
                    level_num].get_children(
                    node_batch,
                    self.char_embedding_layer.weight)
                mlm_loss,mlm_diff_loss = calc_mlm_loss(self.agent_levels[level_num], matrices, real_positions, eos_positions, embedding_matrix,
                                         labels)
                coherence_loss,total_cd_loss = calc_coherence_loss(self.agent_levels[level_num], matrices, real_positions, eos_positions,
                                                     embedding_matrix)

                vectors = torch.stack([node.vector for node in node_batch])
                decompressed = self.agent_levels[level_num].decompressor(vectors)

                reconstruction_diff_loss, reconstruction_loss, rc_loss, re_loss, rj_loss, rm_loss, rm_diff_loss, total_rcd_loss = self.reconstruction_fns[level_num](
                        self.agent_levels[level_num],
                        matrices, decompressed, real_positions,
                        eos_positions,
                        join_positions,
                        embedding_matrix, labels, self.pndb)


                #does it help to do it on decompressed (pre decoded)? maybe but very noisy
                eos_loss,_ = calc_eos_loss(self.agent_levels[level_num], decompressed, eos_positions)
                if Config.join_texts and level_num >= 1:
                    join_loss = calc_join_loss(self.agent_levels[level_num], decompressed, join_positions)
                else:
                    join_loss = torch.tensor([0.0] * matrices.size(0)).to(Config.device)

                losses = {
                    'm': mlm_loss.sum(),
                    'md': mlm_diff_loss.sum(),
                    "c": coherence_loss.sum(),
                    "r": reconstruction_loss.sum(),
                    "e": eos_loss.sum(),
                    "j": join_loss.sum(),
                    "d": reconstruction_diff_loss.sum(),
                    "rc": rc_loss.sum(),
                    "re": re_loss.sum(),
                    "rj": rj_loss.sum(),
                    "rm": rm_loss.sum(),
                    "rmd": rm_diff_loss.sum(),
                    "cd": total_cd_loss,
                    "rcd": total_rcd_loss
                }
                if level_num not in loss_object:  # On the first node_batch
                    loss_object[level_num] = losses
                else:
                    for label, value in losses.items():
                        loss_object[level_num][label] += value

                if generate and batch_num == 0:  # Only run generate on the first batch of nodes
                    g_loss, disc_loss = calc_generation_loss(self.agent_levels[level_num], vectors, matrices, mask)
                    loss_object[level_num]["g"] = g_loss.item()
                    loss_object[level_num]["disc"] = disc_loss.item()
                    total_g_loss += g_loss
                    total_disc_loss += disc_loss

                # If the lengths are not equal then let's catch this
                # assert len(node_batch) == mlm_loss.size(0)
                # assert len(node_batch) == reconstruction_loss.size(0)

                if debug:
                    for i, node in enumerate(node_batch):
                        node.mlm_loss = mlm_loss[i]
                        node.mlm_diff_loss = mlm_diff_loss[i]
                        node.coherence_loss = coherence_loss[i]
                        node.reconstruction_loss = reconstruction_loss[i]
                        node.eos_loss = eos_loss[i]
                        node.join_loss = join_loss[i]
                        node.reconstruction_diff_loss = reconstruction_diff_loss[i]
                        node.rc_loss = rc_loss[i]
                        node.re_loss = re_loss[i]
                        node.rj_loss = rj_loss[i]
                        node.rm_loss = rm_loss[i]
                        node.rm_diff_loss = rm_diff_loss[i]

            current_losses = []
            for label, loss in loss_object[level_num].items():
                if label not in ['g', 'disc']:
                    loss /= len(real_nodes)
                    current_losses.append(loss)
                    loss_object[level_num][label] = loss  # Keep loss_object as a tensor for custom backwards
                    # loss_object[level_num][label] = loss.item()  # Pull out of the GPU for logging
            total_loss += torch.stack(current_losses).to(Config.device).sum()

        return total_g_loss, total_disc_loss, total_loss, loss_object

    def debug_decode(self, batch_tree, node_batch=None):
        if node_batch is None:
            node_batch = batch_tree.level_nodes[0]
        tokens = [n.get_padded_word_tokens() for n in node_batch]
        real_positions = (torch.tensor(tokens) == Config.pad_token_id).float()
        eos_positions = (torch.tensor(tokens) == Config.eos_token_id).float()
        vectors = torch.stack([n.vector for n in node_batch])
        output = self.agent_levels[0].decompressor(vectors)
        output = self.agent_levels[0].decoder(output, real_positions, eos_positions)

        output = torch.matmul(output, self.char_embedding_layer.weight.transpose(0, 1))
        output = torch.argmax(output, dim=2)
        # pred = [dataset.tree_tokenizer.detokenize(w) for w in words]

        return output

    # todo: refactor it to not get embedding_matrices as a parameter (only the char matrix is needed and it belongs to self)
    def full_decode(self, nodes):
        assert len(set([node.level for node in nodes])) == 1  # All nodes must be on the same level

        agent_level = self.agent_levels[nodes[0].level]
        node_vectors = [node.vector for node in nodes if not node.is_join()]
        if len(node_vectors) == 0:  # If all of the nodes are joins
            return [(-1, True) for _ in nodes]
        node_vectors = torch.stack(node_vectors).to(Config.device)
        children_vectors, children_eos, _, _ = agent_level.vecs_to_children_vecs(node_vectors)
        children_eos = children_eos.tolist()

        if nodes[0].level == 0:
            index = 0
            # TODO - use numpy.split here for even faster batching
            result = []
            for node in nodes:
                if node.is_join():
                    result.append((-1, True))
                    continue

                output = children_vectors[index].unsqueeze(0)
                output = torch.matmul(output, self.char_embedding_layer.weight.transpose(0, 1))
                output = torch.argmax(output, dim=2).squeeze(0)
                result.append((output.tolist(), children_eos[index]))
                index += 1
            return result

        # TODO - use numpy.split here for even faster batching
        results = []
        index = 0

        for node in nodes:
            if node.is_join():
                results.append((-1, True))
                continue

            children_nodes = []
            for v in children_vectors[index]:
                n = Node()
                if v is None:
                    n.tokens = -1
                    n.struct = -1
                else:
                    n.vector = v
                n.level = node.level - 1
                n.parent = node
                children_nodes.append(n)
            results.append((self.full_decode(children_nodes), children_eos[index]))
            index += 1

        return results

    def generate_texts(self, level, num_texts=1):
        vecs = self.agent_levels[level].generator(torch.zeros(num_texts, Config.vector_sizes[level + 1]))
        nodes = []
        for v in vecs:
            n = Node()
            n.vector = v
            n.level = level
            nodes.append(n)
        decoded = self.full_decode(nodes)
        return [TreeTokenizer.deep_detokenize(r[0], level) for r in decoded]
