# this file should have a function that takes text and returns Tree with nodes
from src.config import Config
from collections import defaultdict
import nltk
import re

NODE_COUNTER = 0

class Node:
    def __init__(self, id=None, parent=None, children=None, level=2, struct=None,
                 tokens=None, vector=None, realized=False, type=None, tree_id=None):
        self.id = id  # pre order id
        self.struct = struct
        self.parent = parent
        self.children = children
        self.level = level
        self.tokens = tokens
        self.vector = vector
        self.realized = realized
        self.type = type  # leaf, inner, root, join_token, eos_token
        self.tree_id = tree_id  # not sure if we'll need it
        self.distinct_lookup_id = None
        self.mlm_loss = None
        self.coherence_loss = None
        self.reconstruction_loss = None
        self.reconstruction_diff_loss = None

    def get_padded_word_tokens(self):
        if self.level != 0:
            return
        return (self.tokens + [Config.eos_token_id] + [Config.pad_token_id] * Config.sequence_lengths[0])[
               0:Config.sequence_lengths[0]]

    def set_vector(self, v):
        self.vector = v

    def bebug_get_tree(self, attr="id"):
        if self.children is None:
            return getattr(self, attr)
        else:
            return [x.bebug_get_tree(attr) for x in self.children]

    def join_struct_short_children(self):
        # for level >= 2 => a paragraph(2) can join its sentences
        new_struct = [self.struct[0]]
        max_length = Config.sequence_lengths[self.level - 1]  # for each child
        for sub in self.struct[1:]:
            if len(new_struct[-1]) + 1 + len(sub) < max_length:
                # new_struct[-1].append(Config.join_token_id)
                new_struct[-1].append(-1)  # -1+3=2 #this is the most helpful comment ever; thanks for nothing past me!
                new_struct[-1].extend(sub)
            else:
                new_struct.append(sub)
        self.struct = new_struct
        return

    def split_struct_long_children(self):
        # for all levels can split a long text into 2 shorter ones, the first of which will be of max length with no place for eos token
        new_struct = []
        max_length = Config.sequence_lengths[self.level - 1]  # for each child
        for sub in self.struct:
            if isinstance(sub, int) or len(sub) <= max_length:
                new_struct.append(sub)  # no change
            else:
                subs = [sub[i:i + max_length] for i in range(0, len(sub), max_length)]
                new_struct.extend(subs)
        self.struct = new_struct
        return

    def expand_struct(self):  # move set ids to after we are done with joins
        counter = self.id

        def expand_struct1(self):
            # for each level create nodes; give them struct; delete own struct
            # word is the leaf here not letter; it'll make it faster
            nonlocal counter
            counter += 1
            self.id = counter
            if self.struct == Config.join_token_id:
                self.type = "join_token"
                self.tokens = Config.join_token_id
            elif self.level == 0:  # word
                self.tokens = self.struct
                self.type = "leaf"
            else:
                if self.level >= 1:
                    self.split_struct_long_children()
                if Config.join_texts is True and self.level >= 2 and self.type != "batch root":
                    self.join_struct_short_children()
                self.children = [
                    expand_struct1(Node(struct=x, parent=self, level=self.level - 1, type="inner"))
                    for x in self.struct]
            # self.struct = None #todo: delete struct later, this line is only commented for debugging and it take up space
            return self

        return expand_struct1(self)


class BatchTree:
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

    def batch_up_nodes(self):  # fills the self.level_nodes hash, called from tokinizer.batch_texts_to_trees(texts)
        [self.__batch_up_nodes1(c) for c in self.batch_root.children]

    def make_distinct_words(self):
        # the i-th element of distinct_word_embedding_tokens gets word tokens with full padding
        # each node in level_nodes[0] (word), gets distinct_lookup_id set to the relevant i from distinct_word_embedding_tokens
        # in the forward pass we will only embed self.distinct_word_embedding_tokens and fill the DVT word vector with lookup to this matrix
        mapping = {str(n.tokens): [i, n.get_padded_word_tokens()] for i, n in
                   zip(reversed(range(len(self.level_nodes[0]))), self.level_nodes[0])}
        for n in self.level_nodes[0]:
            n.distinct_lookup_id = mapping[str(n.tokens)][0]
        id_and_pads = list(mapping.values())
        id_and_pads.sort()  # by id (first element) is the default
        self.distinct_word_embedding_tokens = [x[1] for x in id_and_pads]
        return

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
        return


class TreeTokenizer:
    def __init__(self, char_file="../chars.txt"):
        self.letter_tokenizer = defaultdict(int, dict(
            zip([l.strip() for l in open(char_file, "r", encoding='utf-8').readlines()], range(7777))))
        self.reverse_tokenizer = {value: key for key, value in self.letter_tokenizer.items()}
        self.sentence_spliter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.split_functions = [self.paragraph_to_sentences, self.sentence_to_words]
        self.max_depth = len(self.split_functions)
        self.seperators = ['',' ','<s>','<p>','<c>']

    def tokenize_word(self, word):
        # "sheeבt" => [68, 57, 54, 54, 0, 69]
        return [self.letter_tokenizer[l] for l in word]

    def detokenize(self, struct):
        #struct=> [3,4,5,67,8]
        # vec/struct to text todo: make it
        res = ""
        for c in struct:
            if c==Config.eos_token_id:
                return res
            else:
                res+=self.reverse_tokenizer[c]
        return res

    def deep_detokenize(self, struct,level):
        if level==0: #isinstance(struct[0], int):
            return self.detokenize(struct)
        else:
            return self.seperators[level].join([self.deep_detokenize(s,level-1) for s in struct])
            #return " ".join([self.deep_detokenize(s) for s in struct])

    def sentence_to_words(self, sentence):
        # "I like big butts." => ['I', 'like', 'big', 'butts.']
        return re.split(' ', sentence)

    def paragraph_to_sentences(self, p):
        # "I like big butts. I can not lie." => ['I like big butts.', 'I can not lie.']
        return self.sentence_spliter.tokenize(p)

    def text_to_tree_struct(self, text, level=2):
        # "I like big butts. I can not lie." => [[[32], [61, 58, 60, 54], [51, 58, 56], [51, 70, 69, 69, 68, 10]], [[32], [52, 50, 63], [63, 64, 69], [61, 58, 54, 10]]]
        if level > 0:
            return [self.text_to_tree_struct(x, level - 1) for x in self.split_functions[self.max_depth - level](text)
                    if len(x) > 0]
        else:
            return self.tokenize_word(text)

    def batch_texts_to_trees(self, texts):  # todo: use level here to make ensure texts are in the right depth
        # input: ["I like big butts. I can not lie.","You other brothers can't deny"]
        structs = [self.text_to_tree_struct(text) for text in texts]
        batch_root = Node(struct=structs, type="batch root", id=0, level=Config.agent_level + 1)
        batch_root.expand_struct()
        batch_tree = BatchTree(batch_root)
        batch_tree.batch_up_nodes()
        batch_tree.make_distinct_words()

        # for i in range(self.max_depth):
        #   batch_tree.make_distinct_texts(i)
        return batch_tree


tt = TreeTokenizer()
# x = tt.tokenize_word("sheeבt")
# x = tt.text_to_tree_struct("I like big   butts. I can not lie.")
tree = tt.batch_texts_to_trees(["I like big butts. I can not lie.", "some other song"])
# x = tt.batch_texts_to_trees(["I am big. you are too.","I am big. you are too."] )
# print([[k,len(v)] for (k,v) in x.level_nodes.items()])
# print(x.batch_root.struct)

# print(x.batch_root.bebug_get_tree(attr="tokens"))
# print(set([x.tokens for x in tree.level_nodes[0]]))
# print({str(x.tokens) : 7 for x in tree.level_nodes[0]})
# print({str(n.tokens) : [i,n.tokens] for i,n in enumerate(tree.level_nodes[0])})
# mapping = {str(n.tokens) : [i,n.get_padded_word_tokens()] for i,n in zip(reversed(range(len(tree.level_nodes[0]))),tree.level_nodes[0])} #reversed so that if a word exists twice the lower id will take
# for n in tree.level_nodes[0]:
#   n.distinct_lookup_id = mapping[str(n.tokens)][0]
#
# print([n.distinct_lookup_id for n in tree.level_nodes[0]])
# ss = list(mapping.values())
# ss.sort()
# print(ss)

# tree.make_distinct_words()
# print("here")
# print(tree.distinct_word_embedding_tokens)
# print([n.distinct_lookup_id for n in tree.level_nodes[0]])

# with open('../chars.txt', encoding='utf-8') as f:
#   chars = [char.strip() for char in f.readlines()]
# print(chars)

# print(x.children[0].children[0].children[0].tokens)
# print(x.bebug_get_tree("tokens"))
# node = Node(struct=tt.text_to_tree_struct("I like big butts. I can not lie."),id=0,level=2,type="debug root") #level 0 is word node
# node.expand_struct()
# #print(node.children[-1].children[-1].tokens) #see that tokens are correct :)
# print("struct",node.struct)
# print("word ids",node.bebug_get_tree(attr="id"))
# print("tokens",node.bebug_get_tree(attr="tokens"))
# #print(node.bebug_get_tree())
# print({i:3 for i in range(5)})
