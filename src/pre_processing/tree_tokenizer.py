from src.pre_processing.batch_tree import BatchTree
from src.pre_processing.node import Node
from src.config import Config
from collections import defaultdict
import nltk
import re


class Splitters:
    sentence_splitter = nltk.data.load('tokenizers/punkt/english.pickle')

    @classmethod
    def sentence_to_words(cls, sentence):
        # "I like big butts." => ['I', 'like', 'big', 'butts.']
        return re.split(' ', sentence)

    @classmethod
    def paragraph_to_sentences(cls, p):
        # "I like big butts. I can not lie." => ['I like big butts.', 'I can not lie.']
        return cls.sentence_splitter.tokenize(p)


class TreeTokenizer:
    letter_tokenizer = defaultdict(int, {char.strip(): i for i, char in enumerate(open('chars.txt', encoding='utf-8'))})
    reverse_tokenizer = {index: char for char, index in letter_tokenizer.items()}
    split_functions = None  # [paragraph_to_sentences, self.sentence_to_words]
    max_depth = None
    separators = ['', ' ', '<s>', '<p>', '<c>']

    @classmethod
    def finalize(cls):
        """
        Called by the Dataset class after the split_functions are set.
        """
        cls.split_functions = cls.split_functions[:Config.agent_level]  # Truncate down to the max agent level
        cls.max_depth = len(cls.split_functions)

    @classmethod
    def tokenize_word(cls, word):
        # "sheeבt" => [68, 57, 54, 54, 0, 69]
        return [cls.letter_tokenizer[letter] for letter in word]

    @classmethod
    def detokenize(cls, struct):
        # struct=> [3,4,5,67,8]
        # vec/struct to text todo: make it
        res = ""
        for c in struct:
            if c == Config.eos_token_id:
                return res
            else:
                res += cls.reverse_tokenizer[c]
        return res

    @classmethod
    def deep_detokenize(cls, struct, level):
        if level == 0:
            return cls.detokenize(struct)

        return cls.separators[level - 1].join([cls.deep_detokenize(s, level - 1) for s in struct])

    @classmethod
    def text_to_tree_struct(cls, text, level):
        # "I like big butts. I can not lie." => [[[32], [61, 58, 60, 54], [51, 58, 56], [51, 70, 69, 69, 68, 10]], [[32], [52, 50, 63], [63, 64, 69], [61, 58, 54, 10]]]
        if level == 0:
            return cls.tokenize_word(text)

        max_length = Config.sequence_lengths[level - 1] - 1  # Truncate to fit in the EOS token.
        parts = cls.split_functions[level - 1](text)
        return [cls.text_to_tree_struct(part, level - 1)[:max_length] for part in parts if len(part) > 0]

    @classmethod
    def batch_texts_to_trees(cls, texts):  # todo: use level here to make ensure texts are in the right depth
        # input: ["I like big butts. I can not lie.","You other brothers can't deny"]
        structs = [cls.text_to_tree_struct(text, level=Config.agent_level) for text in texts]
        batch_root = Node(struct=structs, type="batch root", id=0, level=Config.agent_level + 1)
        batch_root.expand_struct()
        batch_tree = BatchTree(batch_root)
        batch_tree.batch_up_nodes()
        batch_tree.make_distinct_words()

        # for i in range(self.max_depth):
        #   batch_tree.make_distinct_texts(i)
        return batch_tree


if __name__ == '__main__':
    # x = tt.tokenize_word("sheeבt")
    # x = tt.text_to_tree_struct("I like big   butts. I can not lie.")
    tree = TreeTokenizer.batch_texts_to_trees(["I like big butts. I can not lie.", "some other song"])
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
