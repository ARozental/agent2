from src.pre_processing import Node
from src.config import Config
import numpy as np


class BatchTree:
    FIRST_RUN = True

    def __init__(self, batch_root):
        self.level_nodes = {i: [] for i in range(
            Config.agent_level + 1)}  # {0: [sorted nodes for words], 1: [sorted nodes for sentences]}
        self.batch_root = batch_root
        self.distinct_word_embedding_tokens = None  # the i-th element has word tokens

        # generalizes word embedding tokens
        self.distinct_embedding_tokens = {i: [] for i in range(
            Config.agent_level)}  # {0: [sorted nodes for words], 1: [sorted nodes for sentences]}

    def __batch_up_nodes1(self, node):
        self.level_nodes[node.level].append(node)
        if node.children is not None:
            [self.__batch_up_nodes1(c) for c in node.children]

    def batch_up_nodes(self):  # fills the self.level_nodes hash, called from tokenizer.batch_texts_to_trees(texts)
        self.level_nodes = {i: [] for i in range(Config.agent_level + 1)}
        [self.__batch_up_nodes1(c) for c in self.batch_root.children]

    # TODO - Make this work for any number of levels, only works when Config.agent_level == 1
    def fill_dummy_nodes(self):
        if not Config.use_tpu:  # This is only for the TPU
            return

        assert Config.agent_level <= Config.levels['SENTENCE']

        if Config.dynamic_node_sizes:
            if BatchTree.FIRST_RUN:  # On the first step of the model, do max batch size
                Config.node_sizes = Config.node_sizes_max
                BatchTree.FIRST_RUN = False
            else:
                # Dynamically change the node_sizes to be the smallest size that will fit all of the nodes
                for level in range(Config.agent_level + 1):
                    num_nodes = len(self.level_nodes[level])
                    values = [int(Config.node_sizes_max[level] * percent) for percent in [0.25, 0.5, 0.75] if
                              num_nodes < int(Config.node_sizes_max[level] * percent)]
                    if len(values) == 0:
                        Config.node_sizes[level] = Config.node_sizes_max[level]  # Set the max size
                    else:
                        Config.node_sizes[level] = values[0]  # Set the smallest value that fits

        # Trim the nodes down to fit the node_sizes
        # Need >= because the add0 and add1 need to be larger than 0; need to avoid having one be > 0 and one be == 0
        # TODO - Make this better to do level by level (delete sentences to get level 1 done, then trim out words to get level 0 done)
        while len(self.level_nodes[0]) >= Config.node_sizes[0] or len(self.level_nodes[1]) >= Config.node_sizes[1]:
            self.batch_root.children = self.batch_root.children[:-1]
            self.batch_up_nodes()

        add0 = Config.node_sizes[0] - len(self.level_nodes[0])
        add1 = Config.node_sizes[1] - len(self.level_nodes[1])
        assert add0 > 0
        assert add1 > 0

        children_per = np.array_split(range(add0), add1)
        for num_per in children_per:
            dummy_parent = Node(level=1, tokens=[0])
            dummy_parent.is_dummy = True
            dummy_parent.children = []
            for i in range(len(num_per)):
                dummy_child = Node(level=0, tokens=[0])
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

        # Get the words in order
        # self.distinct_word_embedding_tokens = [tokens_to_nodes[tokens] for tokens in unique_words]

    def make_distinct_texts(self, lvl):
        # generalizes make_distinct_words
        # the i-th element of distinct_word_embedding_tokens gets word tokens with full padding
        # each node in level_nodes[0] (word), gets distinct_lookup_id set to the relevant i from distinct_word_embedding_tokens
        # in the forward pass we will only embed self.distinct_word_embedding_tokens and fill the DVT word vector with lookup to this matrix
        mapping = {str(n.tokens): [i, n.get_padded_word_tokens()] for i, n in
                   zip(reversed(range(len(self.level_nodes[lvl]))), self.level_nodes[lvl])}
        for n in self.level_nodes[lvl]:
            n.distinct_lookup_id = mapping[str(n.tokens)][0]
        id_and_pads = list(mapping.values())
        id_and_pads.sort()  # by id (first element) is the default
        self.distinct_word_embedding_tokens = [x[lvl] for x in id_and_pads]
