from src.pre_processing.node import Node
from src.config import Config
import numpy as np
import random


class BatchTree:
    FIRST_RUN = True

    def __init__(self, batch_root):
        self.level_nodes = {i: [] for i in range(
            Config.agent_level + 1)}  # {0: [sorted nodes for words], 1: [sorted nodes for sentences]}
        self.batch_root = batch_root
        self.distinct_word_embedding_tokens = None  # the i-th element has word tokens

        #spare some work for get_children and forward
        self.random_ids0 = None
        self.all_ids1 = None
        self.random_ids1 = None
        self.level_0_lookup_ids = None
        #self.max_distinct_id = None
        self.id_to_tokens = None
        self.distinct_word_embedding_tokens = None
        self.num_dummy_distinct = None

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

    def trim_nodes(self):
        # Trim the nodes down to fit the node_sizes
        # Need >= because the add0 and add1 need to be larger than 0; need to avoid having one be > 0 and one be == 0
        # TODO - Make this better to do level by level (delete sentences to get level 1 done, then trim out words to get level 0 done)
        while len(self.level_nodes[0]) >= Config.node_sizes[0] or len(self.level_nodes[1]) >= Config.node_sizes[1]:
            self.batch_root.children = self.batch_root.children[:-1]
            self.batch_up_nodes()

    # TODO - Make this work for any number of levels, only works when Config.agent_level == 1
    def fill_dummy_nodes(self):
        assert Config.agent_level <= Config.levels['SENTENCE']

        if Config.dynamic_node_sizes:
            if BatchTree.FIRST_RUN:  # On the first step of the model, do max batch size
                Config.node_sizes = Config.node_sizes_max.copy()
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

        self.trim_nodes()

        add0 = Config.node_sizes[0] - len(self.level_nodes[0])
        add1 = Config.node_sizes[1] - len(self.level_nodes[1])
        assert add0 > 0
        assert add1 > 0

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


    #adds to batch root everything that can be pre calculated so it can run in parallel in dataloader collate_fn
    def add_coherence_random_ids(self):
      tokens = [n.tokens for n in self.level_nodes[0]]
      shuffled_tokens = [item for sublist in tokens for item in sublist if item!=Config.eos_token_id]
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



      #all other things to pre calc
      node_batch = self.level_nodes[0]
      max_distinct_id = max([node.distinct_lookup_id for node in node_batch])
      self.id_to_tokens = {node.distinct_lookup_id: node.get_padded_word_tokens() for node in node_batch}

      if Config.use_tpu:
        self.num_dummy_distinct = Config.node_sizes[0] - max_distinct_id
        for i in range(self.num_dummy_distinct):
          self.id_to_tokens[i + max_distinct_id + 1] = [4] + ([Config.pad_token_id] * (Config.sequence_lengths[0] - 1)) #TWF is this line
      else:
        self.num_dummy_distinct = 0
      self.distinct_word_embedding_tokens = [self.id_to_tokens[distinct_id] for distinct_id in range(max_distinct_id + self.num_dummy_distinct + 1)]

      self.random_ids0 = [node.get_padded_random_tokens() for node in node_batch]
      self.level_0_lookup_ids = [node.get_padded_word_tokens() for node in node_batch]

      #for get_children level 1:
      node_batch1 = self.level_nodes[1]

      max_length = Config.sequence_lengths[1]
      add_value = 2 + int(Config.join_texts)
      all_ids = [
        [2 if child.is_join() else getattr(child, 'distinct_lookup_id') + add_value for child in node.children] + [0]
        for node in node_batch1]  # [0] is EOS, 2 is JOIN #inconsistant with level 0
      random_ids = [
        [2 if child.is_join() else getattr(child, 'random_lookup_id') + add_value for child in node.children] + [0]
        for node in node_batch1]  # [0] is EOS, 2 is JOIN #inconsistant with level 0

      #all_ids is broken
      all_ids = [item + [1] * (max_length - len(item)) for item in all_ids]  # 1 is PAD
      random_ids = [item + [1] * (max_length - len(item)) for item in random_ids]  # 1 is PAD

      # This array may be longer than the max_length because it assumes that the EoS token exists
      # But some sentences, etc don't have the EoS at all if they were split


      self.all_ids1 = [item[:max_length] for item in all_ids]
      self.random_ids1 = [item[:max_length] for item in random_ids]


      return
