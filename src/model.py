from src.agent import AgentLevel
from src.config import Config
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import calc_reconstruction_loss
from src.losses.generation import calc_generation_loss
from src.pre_processing import Node, TreeTokenizer
from src.utils import make_noise, prepare_inputs
from src.debug.profiler import Profiler as xp
if Config.use_8bit:
    import bitsandbytes as bnb
import torch.nn as nn
import torch
import numpy as np
from src.losses.calc import loss_object_to_main_loss


class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_letters = len(TreeTokenizer.letter_tokenizer.keys())
        if Config.use_8bit:
            self.char_embedding_layer = bnb.nn.StableEmbedding(num_letters, Config.vector_sizes[0])
        else:
            self.char_embedding_layer = nn.Embedding(num_letters, Config.vector_sizes[0])

        self.agent_levels = nn.ModuleList([AgentLevel(i, num_letters) for i in range(Config.agent_level + 1)])
        for i in range(1, Config.agent_level + 1):
            self.agent_levels[i].previous_level = self.agent_levels[i - 1]

    def set_word_vectors(self, batch_tree, inputs, debug=False):
        local_char_embedding_tokens = inputs['set_word_vectors']['local_char_embedding_tokens']
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
        )  # {0: eos, 1:pad, 2:join}
        word_embedding_matrix = torch.cat([special_vectors, word_embedding_matrix], 0)

        if debug:
            lookup_ids = inputs['set_word_vectors']['lookup_ids']
            all_word_vectors = torch.index_select(word_embedding_matrix, 0,
                                                  lookup_ids).detach()  # [words_in_batch,word_vector_size]
            [n.set_vector(v) for n, v in zip(batch_tree.level_nodes[0], all_word_vectors)]

        return None, word_embedding_matrix, batch_tree.num_dummy_distinct

    def forward(self, batch_tree, inputs, generate=False, debug=False, noise_levels=None, global_step=0, xm=None):
        device = inputs['set_word_vectors']['lookup_ids'].device
        if Config.multi_gpu:
            inputs = prepare_inputs(inputs, squeeze=True, to_device=False)
            gpu_index = int(str(device).replace('cuda:', ''))
            batch_tree = batch_tree[gpu_index]

            for parent_key, values in inputs.items():
                if parent_key == 'lengths':
                    continue

                for key, value in values.items():
                    current_length = inputs['lengths'][parent_key + '-' + key]
                    inputs[parent_key][key] = value[:current_length]
            del inputs['lengths']

        # If don't do this then get an "int object is not iterable" error when Config.multi_gpu
        total_g_loss = torch.tensor(0, dtype=torch.float32, device=device)
        total_disc_loss = torch.tensor(0, dtype=torch.float32, device=device)
        total_loss = torch.tensor(0, dtype=torch.float32, device=device)
        first_A1s, first_pndb_lookup_ids = [], []  # for when we want to debug just the first 5 texts, todo: remove after full_decode uses the reconstruction loss function
        # print("emb: ",len(batch_tree.distinct_word_embedding_tokens))
        # print("level 0: ",len(batch_tree.level_nodes[0]))
        # print("level 1: ",len(batch_tree.level_nodes[1]))
        # print("----------------")
        if len(inputs['set_word_vectors']['local_char_embedding_tokens']) > Config.max_word_embedding_size:
            if global_step == 0:
                print("First batch is too big for embedding")
            return total_g_loss, total_disc_loss, total_loss, None, None, first_A1s, first_pndb_lookup_ids  # todo: move to pre processing + pad embedding and batches for TPU here

        loss_object = {}
        word_embedding_matrix = None
        for level_num in range(Config.agent_level + 1):
            # All the nodes in this level (not including join tokens if on lowest level)=>shouldn't have those
            # full_node_batch = [node for node in batch_tree.level_nodes[level_num] if level_num > 0 or not node.is_join()]
            full_node_batch = batch_tree.level_nodes[level_num]
            num_real_nodes = len(full_node_batch)  # TODO - Is this correct for TPU?
            if level_num == 0:
                with xp.Trace('SetWordVectors'):
                    vectors, wm, num_dummy0_embed = self.set_word_vectors(batch_tree, inputs, debug=debug)
                    word_embedding_matrix = wm

            for batch_index, node_batch in enumerate(batch_tree.level_batches[level_num]):
                loss_object, main_loss, first_A1s, first_pndb_lookup_ids = self.forward_node_batch(level_num,
                                                                                                   batch_index,
                                                                                                   node_batch, inputs,
                                                                                                   word_embedding_matrix,
                                                                                                   debug,
                                                                                                   num_dummy0_embed,
                                                                                                   first_A1s,
                                                                                                   first_pndb_lookup_ids,
                                                                                                   noise_levels,
                                                                                                   num_real_nodes, xm,
                                                                                                   generate,
                                                                                                   loss_object)
                total_loss += main_loss

                del inputs[str(level_num) + '-' + str(batch_index)]

        return total_g_loss, total_disc_loss, total_loss, loss_object, word_embedding_matrix, first_A1s, first_pndb_lookup_ids

    def forward_node_batch(self, level_num, batch_index, node_batch, inputs, word_embedding_matrix, debug,
                           num_dummy0_embed, first_A1s, first_pndb_lookup_ids, noise_levels, num_real_nodes, xm,
                           generate,
                           loss_object):
        num_dummy_nodes = 0
        if Config.use_tpu:
            num_dummy_nodes = len([True for node in node_batch if node.is_dummy])

        with xp.Trace('GetChildren' + str(level_num)):
            if level_num == 0:
                matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy, A1s, pndb_lookup_ids, random_matrices = \
                    self.agent_levels[
                        level_num].get_children(
                        node_batch,
                        inputs,
                        self.char_embedding_layer.weight,
                        word_embedding_matrix,
                        batch_index=batch_index,
                        debug=debug)
            elif level_num == 1:
                matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy, A1s, pndb_lookup_ids, random_matrices = \
                    self.agent_levels[
                        level_num].get_children(
                        node_batch,
                        inputs,
                        word_embedding_matrix,
                        batch_index=batch_index,
                        debug=debug)
                num_dummy += num_dummy0_embed
                if debug and first_A1s == [] and (
                    Config.use_pndb1 or Config.use_pndb2):  # todo: cancat and have it working on all batch nodes later
                    first_A1s, first_pndb_lookup_ids = A1s.detach(), pndb_lookup_ids.detach()
            else:
                matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy, A1s, pndb_lookup_ids, random_matrices = \
                    self.agent_levels[
                        level_num].get_children(
                        node_batch,
                        inputs,
                        None,
                        None, debug=debug)
                num_dummy += num_dummy0_embed

        if Config.use_tpu:
            with xp.Trace('DummyLogitBias' + str(level_num)):
                dummy_logit_bias = np.zeros(embedding_matrix.size(0))
                dummy_logit_bias[-1 * num_dummy:] = 999999999999
                dummy_logit_bias = torch.tensor(dummy_logit_bias, device=matrices.device)
        else:
            dummy_logit_bias = None

        with xp.Trace('CoherenceLoss' + str(level_num)):
            coherence_loss = calc_coherence_loss(self.agent_levels[level_num], matrices,
                                                 real_positions,
                                                 eos_positions,
                                                 embedding_matrix,
                                                 random_matrices,
                                                 num_dummy=num_dummy)
        del random_matrices

        with xp.Trace('MLMLoss' + str(level_num)):
            mlm_loss, rm_loss = calc_mlm_loss(self.agent_levels[level_num], matrices, real_positions,
                                              eos_positions,
                                              embedding_matrix,
                                              labels, num_dummy=num_dummy,
                                              dummy_logit_bias=dummy_logit_bias)

        with xp.Trace('CallingDecompressor' + str(level_num)):
            if not Config.noise or debug or level_num == Config.agent_level:
                decompressed = self.agent_levels[level_num].decompressor(vectors)
            else:
                decompressed = self.agent_levels[level_num].decompressor(
                    make_noise(vectors, noise_levels[level_num + 1]))

        with xp.Trace('ReconstructionLoss' + str(level_num)):
            reconstruction_diff_loss, reconstruction_loss, eos_loss, re_loss, rj_loss, rc_loss = \
                calc_reconstruction_loss(
                    self.agent_levels[level_num],
                    matrices, decompressed, real_positions,
                    eos_positions,
                    join_positions,
                    embedding_matrix, labels, self.agent_levels[1].pndb, A1s, pndb_lookup_ids,
                    num_dummy=num_dummy, dummy_logit_bias=dummy_logit_bias)

        with xp.Trace('JoinLoss' + str(level_num)):
            if Config.join_texts and level_num >= 1:
                join_loss = calc_join_loss(self.agent_levels[level_num], decompressed, join_positions)
            else:
                join_loss = torch.tensor([0.0] * matrices.size(0), device=matrices.device)

        with xp.Trace('LossKeeper' + str(level_num)):
            loss_keeper = np.ones(matrices.size(0))
            if num_dummy_nodes > 0:
                loss_keeper[-1 * num_dummy_nodes:] = 0
            loss_keeper = torch.tensor(loss_keeper, device=matrices.device)

        with xp.Trace('CalculateLosses' + str(level_num)):
            losses = {
                'm': (mlm_loss * loss_keeper).sum(),
                # 'md': (mlm_diff_loss * loss_keeper).sum(),
                "c": (coherence_loss * loss_keeper).sum(),
                "r": (reconstruction_loss * loss_keeper).sum(),
                "e": (eos_loss * loss_keeper).sum(),
                "j": (join_loss * loss_keeper).sum(),
                "d": (reconstruction_diff_loss * loss_keeper).sum(),

                "rc": (rc_loss.view(matrices.shape[:2]) * loss_keeper.unsqueeze(-1)).sum(),
                "re": (re_loss * loss_keeper).sum(),
                "rj": (rj_loss * loss_keeper).sum(),
                "rm": (rm_loss * loss_keeper).sum(),
                # "rmd": (rm_diff_loss * loss_keeper).sum(),
                # "cd": (cd_loss * loss_keeper).mean(),  # This is disabled in coherence loss
                # "rcd": (rcd_loss.view(-1, 2) * loss_keeper.unsqueeze(-1)).mean(),  # TODO - Check if correct
            }

            # TODO - Shouldn't this be divided by len(node_batch)?
            main_loss = loss_object_to_main_loss({level_num: losses}) / num_real_nodes

            if Config.use_accelerator:
                Config.accelerator.backward(main_loss, retain_graph=True)
            else:
                main_loss.backward(retain_graph=True)

            if Config.use_tpu and not Config.profile_tpu:
                xm.mark_step()

            if level_num not in loss_object:  # On the first node_batch
                loss_object[level_num] = losses
                for label, value in losses.items():
                    loss_object[level_num][label] = value.detach() / num_real_nodes
            else:
                for label, value in losses.items():
                    loss_object[level_num][label] += value.detach() / num_real_nodes

        if generate:  # todo: fix here
            g_loss, disc_loss = calc_generation_loss(self.agent_levels[level_num], vectors, matrices, real_positions)
            loss_object[level_num]["g"] = g_loss.detach()
            loss_object[level_num]["disc"] = disc_loss.detach()
        # assert len(node_batch) == mlm_loss.size(0)

        # If the lengths are not equal then let's catch this
        # assert len(node_batch) == reconstruction_loss.size(0)

        if debug:
            for i, node in enumerate(node_batch):
                node.mlm_loss = mlm_loss[i].detach()
                # node.mlm_diff_loss = mlm_diff_loss[i].detach()
                node.coherence_loss = coherence_loss[i].detach()
                node.reconstruction_loss = reconstruction_loss[i].detach()
                node.eos_loss = eos_loss[i].detach()
                node.join_loss = join_loss[i].detach()
                node.reconstruction_diff_loss = reconstruction_diff_loss[i].detach()
                node.rc_loss = rc_loss[i].detach()
                node.re_loss = re_loss[i].detach()
                node.rj_loss = rj_loss[i].detach()
                node.rm_loss = rm_loss[i].detach()
                # node.rm_diff_loss = rm_diff_loss[i].detach()

        return loss_object, main_loss.detach(), first_A1s, first_pndb_lookup_ids

    def compute_vectors(self, batch_tree, inputs):
        _, word_embedding_matrix, _ = self.set_word_vectors(batch_tree, inputs, debug=True)
        for level_num in range(1, Config.agent_level + 1):
            for batch_index, node_batch in batch_tree.level_batches[level_num]:
                _ = self.agent_levels[
                    level_num].get_children(
                    node_batch,
                    inputs,
                    word_embedding_matrix,
                    batch_index=batch_index,
                    debug=True)

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
    def full_decode(self, nodes, A1s, pndb_lookup_ids, embedding_matrix, from_embedding=False):
        assert len(set([node.level for node in nodes])) == 1  # All nodes must be on the same level

        agent_level = self.agent_levels[nodes[0].level]
        node_vectors = [node.vector for node in nodes if not node.is_join()]
        if len(node_vectors) == 0:  # If all of the nodes are joins
            return [(-1, True) for _ in nodes]
        if Config.multi_gpu:
            node_vectors = torch.stack(node_vectors).to(torch.device('cuda:0'))
        else:
            node_vectors = torch.stack(node_vectors)
        if agent_level.level == 1:
            children_vectors, children_vectors_from_embedding, children_eos, _, _ = agent_level.vecs_to_children_vecs(
                node_vectors, A1s, pndb_lookup_ids, embedding_matrix)
        else:
            children_vectors, children_vectors_from_embedding, children_eos, _, _ = agent_level.vecs_to_children_vecs(
                node_vectors, A1s, pndb_lookup_ids, None)

        if from_embedding:
            children_vectors = children_vectors_from_embedding

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
            results.append(
                (self.full_decode(children_nodes, A1s, pndb_lookup_ids, embedding_matrix), children_eos[index]))
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
        A1s, pndb_lookup_ids = None, None  # todo: just have 1 A1 here fix the generator
        decoded = self.full_decode(nodes)
        return [TreeTokenizer.deep_detokenize(r[0], level) for r in decoded]
