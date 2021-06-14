from src.agent import AgentLevel
from src.config import Config
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import calc_reconstruction_loss
from src.losses.generation import calc_generation_loss
from src.pre_processing import Node, TreeTokenizer
from src.utils import iter_even_split, group_by_root, make_noise, node_batch_to_small_batches, apply_recursive
from src.profiler import Profiler as xp
from src.agent.pndb import Pndb
import torch.nn as nn
import torch
import numpy as np
from src.losses.calc import loss_object_to_main_loss, loss_object_to_reconstruction_weights_loss
if Config.use_tpu:
    import torch_xla.core.xla_model as xm



class AgentModel(nn.Module):
    def __init__(self):
        super().__init__()
        num_letters = len(TreeTokenizer.letter_tokenizer.keys())
        self.char_embedding_layer = nn.Embedding(num_letters, Config.vector_sizes[0])

        self.agent_levels = nn.ModuleList([AgentLevel(i, num_letters) for i in range(Config.agent_level + 1)])
        for i in range(1, Config.agent_level + 1):
            self.agent_levels[i].previous_level = self.agent_levels[i - 1]

        self.reconstruction_params = [param for name, param in self.named_parameters() if (("decompressor" in name) or ("decoder" in name))]
        self.main_params = [param for name, param in self.named_parameters() if ("discriminator" not in name) and ("generator" not in name)]

    def set_word_vectors(self, batch_tree, debug=False):
        node_batch = batch_tree.level_nodes[0]
        #max_distinct_id = batch_tree.max_distinct_id
        #id_to_tokens = batch_tree.id_to_tokens #{node.distinct_lookup_id: node.get_padded_word_tokens() for node in node_batch}
        num_dummy_distinct = batch_tree.num_dummy_distinct

        # if Config.use_tpu:
        #     num_dummy_distinct = Config.node_sizes[0] - max_distinct_id
        #     for i in range(num_dummy_distinct):
        #         id_to_tokens[i + max_distinct_id + 1] = [4] + ([Config.pad_token_id] * (Config.sequence_lengths[0] - 1))
        # else:
        #     num_dummy_distinct = 0

        #distinct_word_embedding_tokens = [id_to_tokens[distinct_id] for distinct_id in range(max_distinct_id + num_dummy_distinct + 1)]
        distinct_word_embedding_tokens = batch_tree.distinct_word_embedding_tokens
        local_char_embedding_tokens = torch.tensor(distinct_word_embedding_tokens, dtype=torch.long,
                                                   device=Config.device)
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

        lookup_ids = torch.tensor([x.distinct_lookup_id for x in node_batch], dtype=torch.long, device=Config.device)
        lookup_ids += 2 + int(Config.join_texts)

        if debug:
            all_word_vectors = torch.index_select(word_embedding_matrix, 0, lookup_ids).detach()  # [words_in_batch,word_vector_size]
            [n.set_vector(v) for n, v in zip(node_batch, all_word_vectors)]

        return None, word_embedding_matrix, num_dummy_distinct

    def forward(self, batch_tree, generate=False, debug=False,last_obj={},global_step=0):
        total_g_loss, total_disc_loss, total_loss = 0, 0, 0
        loss_object = {}
        previous_vectors = None
        word_embedding_matrix= None
        for level_num in range(Config.agent_level + 1):
            # All the nodes in this level (not including join tokens if on lowest level)=>shouldn't have those
            #full_node_batch = [node for node in batch_tree.level_nodes[level_num] if level_num > 0 or not node.is_join()]
            full_node_batch = batch_tree.level_nodes[level_num]
            num_real_nodes = len(full_node_batch)
            if level_num == 0:
                with xp.Trace('SetWordVectors'):
                    vectors, wm, num_dummy0_embed = self.set_word_vectors(batch_tree, debug=debug)
                    word_embedding_matrix = wm

            if len(word_embedding_matrix)>Config.max_word_embedding_size:
              #print("embedding is too big:", len(word_embedding_matrix))
              return total_g_loss, total_disc_loss, total_loss, last_obj #todo: move to pre processing + pad embedding and batches for TPU here
            node_batchs=node_batch_to_small_batches(full_node_batch,level_num)

            # if global_step<Config.early_steps and (not debug):
            #   node_batchs = node_batchs[0:1]
            #   num_real_nodes = len(node_batchs[0])
            done_nodes = 0
            for node_batch in node_batchs:
                num_dummy_nodes = 0
                if Config.use_tpu:
                    num_dummy_nodes = len([True for node in node_batch if node.is_dummy])
                #real_node_num = (len(node_batch) - num_dummy_nodes)

                with xp.Trace('GetChildren' + str(level_num)):
                    if level_num == 0:
                        matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy,A1s, pndb_lookup_ids,random_matrices = \
                            self.agent_levels[
                                level_num].get_children(
                                node_batch,
                                self.char_embedding_layer.weight,
                                word_embedding_matrix,
                                done_nodes=done_nodes,
                                batch_tree=batch_tree,
                                debug=debug)
                    elif level_num == 1:
                        matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy, A1s, pndb_lookup_ids,random_matrices = \
                          self.agent_levels[
                            level_num].get_children(
                            node_batch,
                            word_embedding_matrix,
                            None,
                            done_nodes=done_nodes,
                            batch_tree=batch_tree,
                            debug=debug)
                        num_dummy += num_dummy0_embed
                    else:
                        matrices, real_positions, eos_positions, join_positions, embedding_matrix, labels, vectors, num_dummy, A1s, pndb_lookup_ids,random_matrices = \
                            self.agent_levels[
                                level_num].get_children(
                                node_batch,
                                None,
                                None, debug=debug)
                        num_dummy += num_dummy0_embed
                    done_nodes+=len(node_batch)


                if Config.use_tpu:
                    with xp.Trace('DummyLogitBias' + str(level_num)):
                        dummy_logit_bias = np.zeros(embedding_matrix.size(0))
                        dummy_logit_bias[-1 * num_dummy:] = 999999999999
                        dummy_logit_bias = torch.tensor(dummy_logit_bias, device=Config.device)
                else:
                    dummy_logit_bias = None

                with xp.Trace('CoherenceLoss' + str(level_num)):
                    coherence_loss = calc_coherence_loss(self.agent_levels[level_num], matrices,
                                                                  real_positions,
                                                                  eos_positions,
                                                                  embedding_matrix,
                                                                  random_matrices,
                                                                  num_dummy=num_dummy)
                    random_matrices = None

                with xp.Trace('MLMLoss' + str(level_num)):
                    mlm_loss,rm_loss = calc_mlm_loss(self.agent_levels[level_num], matrices, real_positions,
                                                            eos_positions,
                                                            embedding_matrix,
                                                            labels, num_dummy=num_dummy,
                                                            dummy_logit_bias=dummy_logit_bias)

                with xp.Trace('CallingDecompressor' + str(level_num)):
                    if Config.noise == 0 or debug:
                      decompressed = self.agent_levels[level_num].decompressor(vectors)
                    else:
                      decompressed = self.agent_levels[level_num].decompressor(make_noise(vectors))

                with xp.Trace('ReconstructionLoss' + str(level_num)):
                    reconstruction_diff_loss, reconstruction_loss, eos_loss, re_loss, rj_loss,rc_loss = \
                        calc_reconstruction_loss(
                            self.agent_levels[level_num],
                            matrices, decompressed, real_positions,
                            eos_positions,
                            join_positions,
                            embedding_matrix, labels, self.agent_levels[1].pndb,A1s, pndb_lookup_ids, num_dummy=num_dummy, dummy_logit_bias=dummy_logit_bias)


                with xp.Trace('JoinLoss' + str(level_num)):
                    if Config.join_texts and level_num >= 1:
                        join_loss = calc_join_loss(self.agent_levels[level_num], decompressed, join_positions)
                    else:
                        join_loss = torch.tensor([0.0] * matrices.size(0), device=Config.device)

                with xp.Trace('LossKeeper' + str(level_num)):
                    loss_keeper = np.ones(matrices.size(0))
                    if num_dummy_nodes > 0:
                        loss_keeper[-1 * num_dummy_nodes:] = 0
                    loss_keeper = torch.tensor(loss_keeper, device=Config.device)

                with xp.Trace('CalculateLosses' + str(level_num)):
                    losses = {
                        'm': (mlm_loss * loss_keeper).sum(),
                        #'md': (mlm_diff_loss * loss_keeper).sum(),
                        "c": (coherence_loss * loss_keeper).sum(),
                        "r": (reconstruction_loss * loss_keeper).sum(),
                        "e": (eos_loss * loss_keeper).sum(),
                        "j": (join_loss * loss_keeper).sum(),
                        "d": (reconstruction_diff_loss * loss_keeper).sum(),

                        "rc": (rc_loss.view(matrices.shape[:2]) * loss_keeper.unsqueeze(-1)).sum(),
                        "re": (re_loss * loss_keeper).sum(),
                        "rj": (rj_loss * loss_keeper).sum(),
                        "rm": (rm_loss * loss_keeper).sum(),
                        #"rmd": (rm_diff_loss * loss_keeper).sum(),
                        #"cd": (cd_loss * loss_keeper).mean(),  # This is disabled in coherence loss
                        #"rcd": (rcd_loss.view(-1, 2) * loss_keeper.unsqueeze(-1)).mean(),  # TODO - Check if correct
                    }

                    main_loss = loss_object_to_main_loss({level_num: losses}) / num_real_nodes
                    main_loss.backward(retain_graph=True) #after 20k batches this gave RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED => WTF

                    if Config.use_tpu and not Config.profile_tpu:
                      xm.mark_step()

                    if level_num not in loss_object:  # On the first node_batch
                        loss_object[level_num] = losses
                        for label, value in losses.items():
                            loss_object[level_num][label] = value.detach() / num_real_nodes
                    else:
                        for label, value in losses.items():
                            loss_object[level_num][label] += value.detach() / num_real_nodes


                    losses,main_loss,r_loss = None,None,None

                if generate:  # todo: fix here
                    g_loss, disc_loss = calc_generation_loss(self.agent_levels[level_num], vectors, matrices, real_positions)
                    loss_object[level_num]["g"] = g_loss.item()
                    loss_object[level_num]["disc"] = disc_loss.item()
                    total_g_loss += g_loss
                    total_disc_loss += disc_loss

                # If the lengths are not equal then let's catch this
                # assert len(node_batch) == mlm_loss.size(0)
                # assert len(node_batch) == reconstruction_loss.size(0)

                if debug:
                    for i, node in enumerate(node_batch):
                        node.mlm_loss = mlm_loss[i].detach()
                        #node.mlm_diff_loss = mlm_diff_loss[i].detach()
                        node.coherence_loss = coherence_loss[i].detach()
                        node.reconstruction_loss = reconstruction_loss[i].detach()
                        node.eos_loss = eos_loss[i].detach()
                        node.join_loss = join_loss[i].detach()
                        node.reconstruction_diff_loss = reconstruction_diff_loss[i].detach()
                        node.rc_loss = rc_loss[i].detach()
                        node.re_loss = re_loss[i].detach()
                        node.rj_loss = rj_loss[i].detach()
                        node.rm_loss = rm_loss[i].detach()
                        #node.rm_diff_loss = rm_diff_loss[i].detach()
                mlm_loss, mlm_diff_loss, coherence_loss, reconstruction_loss, eos_loss, join_loss, reconstruction_diff_loss, re_loss, rm_loss = None, None, None, None, None, None, None, None, None

        # with xp.Trace('ComputeTotalLoss' + str(level_num)):
        #   current_losses = []
        #   for label, loss in loss_object[level_num].items():
        #     if label not in ['g', 'disc', 'cd', 'rcd']:
        #       current_losses.append(loss)
        #       loss_object[level_num][label] = loss  # Keep loss_object as a tensor for custom backwards
        #       # loss_object[level_num][label] = loss.item()  # Pull out of the GPU for logging
        #   total_loss += torch.stack(current_losses).sum()

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
        node_vectors = torch.stack(node_vectors)
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
