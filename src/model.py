from src.agent import AgentLevel
from src.config import Config
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import calc_reconstruction_loss
from src.losses.generation import calc_generation_loss
from src.pre_processing import Node, TreeTokenizer
import torch.nn as nn
import torch


class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_letters = len(TreeTokenizer.letter_tokenizer.keys())
        self.agent_levels = nn.ModuleList([AgentLevel(i, num_letters) for i in range(Config.agent_level + 1)])
        self.char_embedding_layer = nn.Embedding(num_letters, Config.vector_sizes[0])

    def set_word_vectors(self, batch_tree):
        node_batch = batch_tree.level_nodes[0]
        local_char_embedding_tokens = torch.LongTensor(batch_tree.distinct_word_embedding_tokens).to(Config.device)
        mask = local_char_embedding_tokens == Config.pad_token_id  # True => position to mask
        eos_positions = local_char_embedding_tokens == Config.eos_token_id  # True => position to mask
        local_char_embedding_matrix = self.char_embedding_layer(local_char_embedding_tokens)

        # first encoder call
        word_embedding_matrix = self.agent_levels[0].compressor(
            self.agent_levels[0].encoder(local_char_embedding_matrix, mask, eos_positions.float()),
            mask)  # [distinct_words_in_batch,word_vector_size]

        if Config.join_texts:
            special_vectors = torch.stack([
                self.agent_levels[1].eos_vector,
                self.agent_levels[1].pad_vector,
                self.agent_levels[1].join_vector,
            ])  # {0: eos, 1:pad, 2:join}
            word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)
            lookup_ids = torch.LongTensor([x.distinct_lookup_id for x in node_batch]).to(Config.device) + 3
        else:
            special_vectors = torch.stack([
                self.agent_levels[1].eos_vector,
                self.agent_levels[1].pad_vector,
            ])  # {0: eos, 1:pad}
            word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)
            lookup_ids = torch.LongTensor([x.distinct_lookup_id for x in node_batch]).to(Config.device) + 2

        all_word_vectors = torch.index_select(word_embedding_matrix, 0, lookup_ids)  # [words_in_batch,word_vector_size]
        [n.set_vector(v) for n, v in zip(node_batch, all_word_vectors)]
        return word_embedding_matrix

    def set_text_vectors(self, batch_tree):
        word_embedding_matrix = self.set_word_vectors(batch_tree)
        for i in range(1, Config.agent_level + 1):
            self.agent_levels[i].realize_vectors(batch_tree.level_nodes[i])
        return word_embedding_matrix

    def forward(self, batch_tree, with_debug=False, generate=None, epoch=0):
        word_embedding_matrix = self.set_text_vectors(batch_tree)
        embedding_matrices = {0: self.char_embedding_layer.weight, 1: word_embedding_matrix}
        total_g_loss, total_disc_loss, total_loss = 0, 0, 0
        loss_object = {}
        for i in range(Config.agent_level + 1):
            # All the nodes in this level (not including join tokens if on lowest level)
            node_batch = [node for node in batch_tree.level_nodes[i] if i > 0 or not node.is_join()]

            matrices, mask, eos_positions, join_positions, embedding_matrix, labels = self.agent_levels[i].get_children(
                node_batch,
                embedding_matrices[
                    i % 2])  # we only care about 0 and 1
            mlm_loss = calc_mlm_loss(self.agent_levels[i], matrices, mask, eos_positions, embedding_matrix, labels)
            coherence_loss = calc_coherence_loss(self.agent_levels[i], matrices, mask, eos_positions, embedding_matrix)

            # TODO - Check if this grabbing of vectors is correct
            vectors = torch.stack([node.vector for node in node_batch])
            decompressed = self.agent_levels[i].decompressor(vectors)
            reconstruction_diff_loss, reconstruction_loss = calc_reconstruction_loss(self.agent_levels[i], matrices,
                                                                                     decompressed, mask, eos_positions,
                                                                                     embedding_matrix, labels)
            eos_loss = calc_eos_loss(self.agent_levels[i], decompressed, eos_positions)

            if Config.join_texts and i >= 1:
                join_loss = calc_join_loss(self.agent_levels[i], decompressed, join_positions)
            else:
                join_loss = torch.tensor([0.0] * matrices.size(0))

            total_loss += (
                    mlm_loss.mean() +
                    coherence_loss.mean() +
                    reconstruction_loss.mean() +
                    eos_loss.mean() +
                    join_loss.mean() +
                    reconstruction_diff_loss.mean()
            ).sum()

            loss_object[i] = {
                'm': mlm_loss.mean().item(),
                "c": coherence_loss.mean().item(),
                "r": reconstruction_loss.mean().item(),
                "e": eos_loss.mean().item(),
                "j": join_loss.mean().item(),
                "d": reconstruction_diff_loss.mean().item()
            }

            if generate:
                g_loss, disc_loss = calc_generation_loss(self.agent_levels[i], vectors, matrices, mask)
                loss_object[i]["g"] = g_loss.item()
                loss_object[i]["disc"] = disc_loss.item()
                total_g_loss += g_loss
                total_disc_loss += disc_loss

            # If the lengths are not equal then let's catch this
            assert len(node_batch) == mlm_loss.size(0)
            assert len(node_batch) == reconstruction_loss.size(0)

            [setattr(n, 'mlm_loss', l) for n, l in zip(node_batch, mlm_loss.tolist())]
            [setattr(n, 'coherence_loss', l) for n, l in zip(node_batch, coherence_loss.tolist())]
            [setattr(n, 'reconstruction_loss', l) for n, l in zip(node_batch, reconstruction_loss.tolist())]
            [setattr(n, 'eos_loss', l) for n, l in zip(node_batch, eos_loss.tolist())]
            [setattr(n, 'join_loss', l) for n, l in zip(node_batch, join_loss.tolist())]
            [setattr(n, 'reconstruction_diff_loss', l) for n, l in zip(node_batch, reconstruction_diff_loss.tolist())]

            embedding_matrices[i] = embedding_matrix

        return total_g_loss, total_disc_loss, total_loss, loss_object

    def debug_decode(self, batch_tree, node_batch=None):
        if node_batch is None:
            node_batch = batch_tree.level_nodes[0]
        tokens = [n.get_padded_word_tokens() for n in node_batch]
        mask = torch.tensor(tokens) == Config.pad_token_id  # True => position to mask
        eos_positions = (torch.tensor(tokens) == Config.eos_token_id).float()
        vectors = torch.stack([n.vector for n in node_batch])
        output = self.agent_levels[0].decompressor(vectors)
        output = self.agent_levels[0].decoder(output, mask, eos_positions)

        output = torch.matmul(output, self.char_embedding_layer.weight.transpose(0, 1))
        output = torch.argmax(output, dim=2)
        # pred = [dataset.tree_tokenizer.detokenize(w) for w in words]

        return output

    # todo: refactor it to not get embedding_matrices as a parameter (only the char matrix is needed and it belongs to self)
    def full_decode(self, nodes):
        assert len(set([node.level for node in nodes])) == 1  # All nodes must be on the same level

        agent_level = self.agent_levels[nodes[0].level]
        node_vectors = torch.stack([node.vector for node in nodes if not node.is_join()]).to(Config.device)
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
