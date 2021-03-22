from src.agent import AgentLevel
import torch.nn as nn
import torch
from src.config import Config
from src.losses.mlm import calc_mlm_loss
from src.losses.coherence import calc_coherence_loss
from src.losses.reconstruction import calc_reconstruction_loss
from src.losses.generation import calc_generation_loss
from src.pre_processing import Node


class AgentModel(nn.Module):
    def __init__(self, tree_tokenizer):
        super().__init__()
        self.tree_tokenizer = tree_tokenizer
        self.agent_levels = nn.ModuleList()
        for i in range(Config.agent_level + 2):
            agent_level = AgentLevel(i)
            if i==0:
                agent_level.token_bias = nn.Parameter(torch.zeros(len(tree_tokenizer.letter_tokenizer.keys()), requires_grad=True))
            self.agent_levels.append(agent_level)

        self.char_embedding_layer = nn.Embedding(len(tree_tokenizer.letter_tokenizer.keys()), Config.vector_sizes[0])

    def set_word_vectors(self, batch_tree):
        node_batch = batch_tree.level_nodes[0]
        local_char_embedding_tokens = torch.LongTensor(batch_tree.distinct_word_embedding_tokens)
        mask = local_char_embedding_tokens == Config.pad_token_id  # True => position to mask
        eos_positions = local_char_embedding_tokens == Config.eos_token_id  # True => position to mask
        local_char_embedding_matrix = self.char_embedding_layer(local_char_embedding_tokens)

        #first encoder call
        word_embedding_matrix = self.agent_levels[0].compressor(
            self.agent_levels[0].encoder(local_char_embedding_matrix,mask, eos_positions.float()),mask)  # [distinct_words_in_batch,word_vector_size]

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

    def do_debug(self,batch_tree,epoch,total_loss,loss_object):

      print("epoch:", epoch, "main loss:", total_loss.item(), "loss object:", loss_object)
      nodes = batch_tree.batch_root.children

      text2 = self.generate_texts(2, 1)[0]
      text1 = self.generate_texts(1, 1)[0]
      text0 = self.generate_texts(0, 1)[0]
      print("generated word:", text0)
      print("generated sentence:", text1)
      print("generated paragraph:", text2)


      print("l2w",self.agent_levels[2].classifier1w.item())
      print("l2b",self.agent_levels[2].classifier1b.item())


      res = [self.full_decode(node) for node in nodes]
      text = [self.tree_tokenizer.deep_detokenize(r, 3) for r in res]
      sizes1 = [len(r) for r in res]  # should be [2,1]
      sizes2 = [[len(c.children) for c in r.children] for r in nodes]  # should be [2,1]
      print("reconstructed text:", text)
      print("sizes", sizes1, sizes2)

      debug_object = text
      return debug_object

    def forward(self, batch_tree, with_debug=False, generate=None,epoch=0):
        word_embedding_matrix = self.set_text_vectors(batch_tree)
        embedding_matrices = {0: self.char_embedding_layer.weight, 1: word_embedding_matrix}
        total_g_loss, total_disc_loss, total_loss = 0,0,0
        loss_object = {}
        for i in range(Config.agent_level + 1):
            node_batch = batch_tree.level_nodes[i]  # currently all the nodes in the level
            level, matrices, mask, eos_positions, embedding_matrix, labels = self.agent_levels[i].get_children(node_batch,
                                                                                                embedding_matrices[
                                                                                                    i % 2])  # we only care about 0 and 1
            mlm_loss = calc_mlm_loss(self.agent_levels[i], matrices, mask, eos_positions, embedding_matrix, labels)
            coherence_loss = calc_coherence_loss(self.agent_levels[i], matrices, mask, eos_positions, embedding_matrix)
            vectors = torch.stack([n.vector for n in node_batch])
            reconstruction_diff_loss,eos_loss,reconstruction_loss = calc_reconstruction_loss(self.agent_levels[i], matrices, vectors, mask,eos_positions, embedding_matrix,labels,epoch=epoch)

            total_loss += (mlm_loss.mean() + coherence_loss.mean() + reconstruction_loss.mean() + eos_loss.mean() + reconstruction_diff_loss.mean()).sum()
            loss_object[i] = {'m': mlm_loss.mean().item(), "c": coherence_loss.mean().item(),
                              "r": reconstruction_loss.mean().item(),"e": eos_loss.mean().item(),
                              "d": reconstruction_diff_loss.mean().item()
                              }


            if generate ==True:
                g_loss, disc_loss = calc_generation_loss(self.agent_levels[i],vectors,matrices,mask)
                loss_object[i]["g"] = g_loss.item()
                loss_object[i]["disc_loss"] = disc_loss.item()
                total_g_loss += g_loss
                total_disc_loss += disc_loss




            [setattr(n, 'mlm_loss', l) for n, l in zip(node_batch, mlm_loss.tolist())]
            [setattr(n, 'coherence_loss', l) for n, l in zip(node_batch, coherence_loss.tolist())]
            [setattr(n, 'reconstruction_loss', l) for n, l in zip(node_batch, reconstruction_loss.tolist())]
            [setattr(n, 'eos_loss', l) for n, l in zip(node_batch, eos_loss.tolist())]
            [setattr(n, 'reconstruction_diff_loss', l) for n, l in zip(node_batch, reconstruction_diff_loss.tolist())]

            embedding_matrices[i] = embedding_matrix



        #for full decode test
        debug_object = None
        if with_debug == True:
          debug_object = self.do_debug(batch_tree,epoch,total_loss,loss_object)
        return debug_object, total_g_loss, total_disc_loss,total_loss,loss_object   # todo: make loss object

    def debug_decode(self, batch_tree,node_batch=None):
        if node_batch==None:
            node_batch = batch_tree.level_nodes[0]
        tokens = [n.get_padded_word_tokens() for n in node_batch]
        mask = torch.tensor(tokens) == Config.pad_token_id  # True => position to mask
        eos_positions = (torch.tensor(tokens) == Config.eos_token_id).float()
        vectors = torch.stack([n.vector for n in node_batch])
        output = self.agent_levels[0].decompressor(vectors)
        output = self.agent_levels[0].decoder(output, mask, eos_positions)

        output = torch.matmul(output, self.char_embedding_layer.weight.transpose(0, 1))
        output = torch.argmax(output, dim=2)
        #pred = [dataset.tree_tokenizer.detokenize(w) for w in words]

        return output

    def full_decode(self,node):
        #todo: refactor it to not get embedding_matrices as a parameter (only the char matrix is needed and it belongs to self)
        agent_level = self.agent_levels[node.level]
        children_vecs = agent_level.vec_to_children_vecs(node)
        if node.level==0:
            output = torch.matmul(torch.stack(children_vecs,dim=0).unsqueeze(0), self.char_embedding_layer.weight.transpose(0, 1))
            output = torch.argmax(output, dim=2)
            node.struct = output.tolist()
            return node.struct

        children_nodes = []
        for v in children_vecs:
            n = Node()
            n.vector = v
            n.level = node.level-1
            n.parent = node
            children_nodes.append(n)
        node.children = children_nodes
        return [self.full_decode(n) for n in children_nodes]

    def generate_texts(self, level, num_texts=1):
        vecs = self.agent_levels[level].generator(torch.zeros(num_texts, Config.vector_sizes[level + 1]))
        nodes = []
        for v in vecs:
            n = Node()
            n.vector = v
            n.level = level
            nodes.append(n)
        decoded = [self.full_decode(n) for n in nodes]
        return  [self.tree_tokenizer.deep_detokenize(r, level+1) for r in decoded]

