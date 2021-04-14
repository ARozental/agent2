from src.pre_processing.batch_tree import BatchTree
from src.pre_processing.node import Node
from src.config import Config
from collections import defaultdict
import nltk
import os
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
    dir_path = os.path.dirname(os.path.realpath(__file__))
    char_file = os.path.join(dir_path, '..', '..', 'chars.txt')
    letter_tokenizer = defaultdict(int, {char.strip(): i for i, char in enumerate(open(char_file, encoding='utf-8'))})
    reverse_tokenizer = {index: char for char, index in letter_tokenizer.items()}
    reverse_tokenizer[2] = ''
    split_functions = None  # [paragraph_to_sentences, self.sentence_to_words]
    max_depth = None
    separators = ['', ' ', '<s>', '<p>', '<c>']

    @classmethod
    def tokenize_word(cls, word):
        # "sheeבt" => [68, 57, 54, 54, 0, 69]
        # Add the EoS token here since it needs to be split later into its own word (if necessary based on word length)
        return [cls.letter_tokenizer[letter] for letter in word] + [Config.eos_token_id]

    @classmethod
    def detokenize(cls, struct):
        # struct => [3,4,5,67,8]
        return cls.separators[0].join([cls.reverse_tokenizer[c] for c in struct])

    @classmethod
    def deep_detokenize(cls, struct, level):
        if level == 0:
            if struct == -1:  # This is probably the parent call so returning some join token
                return '<join>'
            return cls.detokenize(struct)

        # Need to specifically handle the missing EoS token from words
        if level == 1:
            result = []
            continue_word = False
            for word, has_eos in struct:
                if word == -1:
                    # If the result is empty then don't do anything with the join
                    if len(result) == 0:
                        continue

                    if continue_word:
                        result.append(cls.separators[level + 1])
                    else:
                        result[-1] = cls.separators[level + 1]
                    continue_word = False
                    continue

                text = cls.deep_detokenize(word, level - 1)
                if continue_word:
                    result[-1] += text
                else:
                    result.append(text)

                if has_eos:
                    result.append(cls.separators[level])

                continue_word = not has_eos
            return ''.join(result[:-1])  # Delete the last separator

        # Combine parts
        new_struct = []
        continue_part = False
        for part, has_eos in struct:
            if part == -1:
                new_struct.append(-1)
                continue

            if continue_part and isinstance(new_struct[-1], list):
                new_struct[-1] += part
            else:
                new_struct.append(part)

            continue_part = not has_eos
        struct = new_struct

        result = []
        for part in struct:
            if part == -1:
                if len(result) == 0:
                    continue

                result[-1] = cls.separators[level + 1]
            else:
                result.append(cls.deep_detokenize(part, level - 1))
                result.append(cls.separators[level])

        return ''.join(result[:-1])

    @classmethod
    def text_to_tree_struct(cls, text, level):
        # "I like big butts. I can not lie." => [[[32], [61, 58, 60, 54], [51, 58, 56], [51, 70, 69, 69, 68, 10]], [[32], [52, 50, 63], [63, 64, 69], [61, 58, 54, 10]]]
        if level == 0:
            return cls.tokenize_word(text)

        parts = cls.split_functions[level - 1](text)
        return [cls.text_to_tree_struct(part, level - 1) for part in parts if len(part) > 0]

    @classmethod
    def find_max_lengths(cls, text):
        text = text[0]
        level_lengths = {i: 0 for i in range(Config.agent_level + 1)}

        def sub_find(item, level):
            if level == 0:
                length = len(item)
            else:
                parts = [part for part in cls.split_functions[level - 1](item) if len(part) > 0]
                [sub_find(part, level - 1) for part in parts]
                length = len(parts)

            if length > level_lengths[level]:
                level_lengths[level] = length

        sub_find(text, Config.agent_level)

        return level_lengths

    @classmethod
    def parse_extra_levels(cls, text):
        """
        The extra levels above Config.agent_level need to be processed to be broken down.

        len() in the if statements makes sure that the split function even exists.
        """
        text = [text]  # Make it an array just in case

        if Config.agent_level < Config.levels['BOOK'] <= len(cls.split_functions):
            text = [chapter.strip() for book in text for chapter in
                    cls.split_functions[Config.levels['BOOK'] - 1](book) if len(chapter.strip()) > 0]

        if Config.agent_level < Config.levels['CHAPTER'] <= len(cls.split_functions):
            text = [paragraph.strip() for chapter in text for paragraph in
                    cls.split_functions[Config.levels['CHAPTER'] - 1](chapter) if len(paragraph.strip()) > 0]

        if Config.agent_level < Config.levels['PARAGRAPH'] <= len(cls.split_functions):
            text = [sent.strip() for paragraph in text for sent in
                    cls.split_functions[Config.levels['PARAGRAPH'] - 1](paragraph) if len(sent.strip()) > 0]

        return text

    @classmethod
    def batch_texts_to_trees(cls, texts):  # todo: use level here to make ensure texts are in the right depth
        # input: ["I like big butts. I can not lie.","You other brothers can't deny"]

        texts = [item.strip() for text in texts for item in cls.parse_extra_levels(text)]

        structs = [cls.text_to_tree_struct(text, level=Config.agent_level) for text in texts]
        batch_root = Node(level=Config.agent_level + 1)
        batch_root.id = 0
        batch_root.expand_struct(structs)
        batch_tree = BatchTree(batch_root)
        batch_tree.batch_up_nodes()
        batch_tree.make_distinct_words()

        # for i in range(self.max_depth):
        #   batch_tree.make_distinct_texts(i)
        return batch_tree

    @classmethod
    def compute_struct_stats(cls, struct, stats, level):
        stats[level].append(len(struct))
        if level == 0:
            stats[level][-1] -= 1
            return stats

        for child in struct:
            stats = cls.compute_struct_stats(child, stats, level - 1)
        return stats

    @classmethod
    def compute_stats(cls, texts):
        texts = [item.strip() for text in texts for item in cls.parse_extra_levels(text)]
        structs = [cls.text_to_tree_struct(text, level=Config.agent_level) for text in texts]
        stats_object = {level: [] for level in range(Config.agent_level + 1)}
        for struct in structs:
            stats_object = cls.compute_struct_stats(struct, stats_object, Config.agent_level)

        return stats_object


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
