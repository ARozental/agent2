from src.utils import node_batch_to_small_batches
from src.pre_processing.node import Node
from src.config import Config
import numpy as np
import random
import torch


class BatchTree:
    FIRST_RUN = True

    def __init__(self, batch_root):
        self.level_nodes = {i: [] for i in range(
            Config.agent_level + 1)}  # {0: [sorted nodes for words], 1: [sorted nodes for sentences]}
        self.level_batches = {i: [] for i in range(Config.agent_level + 1)}
        self.batch_root = batch_root

        # spare some work for get_children and forward
        self.num_dummy_distinct = None

    def __batch_up_nodes1(self, node):
        self.level_nodes[node.level].append(node)
        if node.children is not None:
            [self.__batch_up_nodes1(c) for c in node.children]

    def batch_up_nodes(self):  # fills the self.level_nodes hash, called from tokenizer.batch_texts_to_trees(texts)
        self.level_nodes = {i: [] for i in range(Config.agent_level + 1)}
        [self.__batch_up_nodes1(c) for c in self.batch_root.children]

    # TODO - Make this work for any number of levels, only works when Config.agent_level == 1
    def fill_dummy_nodes(self):
        assert Config.agent_level <= Config.levels['SENTENCE']

        add0 = len(self.level_nodes[0]) % Config.node_sizes[0]
        add1 = len(self.level_nodes[1]) % Config.node_sizes[1]

        children_per = np.array_split(range(add0), add1)
        for num_per in children_per:
            dummy_parent = Node(level=1, tokens=[2])
            dummy_parent.is_dummy = True
            dummy_parent.children = []
            for i in range(len(num_per)):
                dummy_child = Node(level=0, tokens=[2])
                dummy_child.is_dummy = True
                self.level_nodes[0].append(dummy_child)
                dummy_parent.children.append(dummy_child)

            self.level_nodes[1].append(dummy_parent)

    def make_distinct_words(self):
        # the i-th element of distinct_word_embedding_tokens gets word tokens with full padding
        # each node in level_nodes[0] (word), gets distinct_lookup_id set to the relevant i from distinct_word_embedding_tokens
        # in the forward pass we will only embed self.distinct_word_embedding_tokens and fill the DVT word vector with lookup to this matrix
        word_nodes = [node for node in self.level_nodes[0] if not node.is_join()]
        tokens_to_nodes = {str(node.tokens): node.get_padded_word_tokens() for node in word_nodes}
        unique_words = list({str(node.tokens) for node in word_nodes})
        mapping = {tokens: i for i, tokens in enumerate(unique_words)}
        mapping['-1'] = -1
        for node in self.level_nodes[0]:
            node.distinct_lookup_id = mapping[str(node.tokens)]

    def make_node_batches(self):
        for level in range(Config.agent_level + 1):
            self.level_batches[level] = node_batch_to_small_batches(self.level_nodes[level], level)

    def valid_tree(self):
        for i in range(Config.agent_level + 1):
            node_batch = self.level_nodes[i]
            md5s = [n.root_md5 for n in node_batch]
            seen = set([])
            for j in range(1, len(md5s)):
                if md5s[j] in seen and md5s[j] != md5s[j - 1]:
                    print("bad batch ")
                    print(md5s)
                    print([x.build_struct() for x in node_batch])
                    print("--------")

                return False
        return True

    def prepare_batch(self, node_batch, level):
        add_value = 2 + int(Config.join_texts)
        if level == 0:
            return {
                'lookup_ids': torch.tensor([node.get_padded_word_tokens() for node in node_batch], dtype=torch.long),
                'word_lookup_ids': torch.tensor([node.distinct_lookup_id + add_value for node in node_batch],
                                                dtype=torch.long),
                'random_ids': torch.tensor([node.get_padded_random_tokens() for node in node_batch], dtype=torch.long),
            }
        elif level == 1:
            max_length = Config.sequence_lengths[1]
            add_value = 2 + int(Config.join_texts)
            all_ids = [[2 if child.is_join() else getattr(child, 'distinct_lookup_id') + add_value for child in
                        node.children] + [0] for node in node_batch]  # [0] is EOS, 2 is JOIN #inconsistant with level 0
            random_ids = [[2 if child.is_join() else getattr(child, 'random_lookup_id') + add_value for child in
                           node.children] + [0] for node in
                          node_batch]  # [0] is EOS, 2 is JOIN #inconsistant with level 0

            all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]  # 1 is PAD
            random_ids = [item + [1] * (max_length - len(item)) for item in random_ids]  # 1 is PAD

            # This array may be longer than the max_length because it assumes that the EoS token exists
            # But some sentences, etc don't have the EoS at all if they were split

            all_ids = [item[:max_length] for item in all_ids]
            random_ids = [item[:max_length] for item in random_ids]

            return {
                'all_ids': torch.tensor(all_ids, dtype=torch.long),
                'random_ids': torch.tensor(random_ids, dtype=torch.long),
            }
        else:
            raise NotImplementedError('Not implemented for level >= 2.')

    # adds to batch root everything that can be pre calculated so it can run in parallel in dataloader collate_fn
    def add_coherence_random_ids(self):
        tokens = [n.tokens for n in self.level_nodes[0]]
        shuffled_tokens = [item for sublist in tokens for item in sublist if item != Config.eos_token_id]
        shuffled_ids = [n.distinct_lookup_id for n in self.level_nodes[0]]
        random.shuffle(shuffled_tokens)
        random.shuffle(shuffled_ids)

        for n in self.level_nodes[0]:
            random_tokens = []
            for t in n.tokens:
                if t == Config.eos_token_id:
                    random_tokens.append(Config.eos_token_id)
                else:
                    random_tokens.append(shuffled_tokens.pop())
            n.random_tokens = random_tokens
            n.random_lookup_id = shuffled_ids.pop()

        # all other things to pre calc
        max_distinct_id = max([node.distinct_lookup_id for node in self.level_nodes[0]])
        id_to_tokens = {node.distinct_lookup_id: node.get_padded_word_tokens() for node in self.level_nodes[0]}

        if Config.use_tpu:
            self.num_dummy_distinct = Config.node_sizes[0] - max_distinct_id
            for i in range(self.num_dummy_distinct):
                # WTF is this line
                id_to_tokens[i + max_distinct_id + 1] = [4] + ([Config.pad_token_id] * (Config.sequence_lengths[0] - 1))
        else:
            self.num_dummy_distinct = 0

        return [id_to_tokens[distinct_id] for distinct_id in range(max_distinct_id + self.num_dummy_distinct + 1)]
