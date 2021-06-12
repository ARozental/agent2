from src.config import Config

NODE_COUNTER = 1


def get_unique_id():
    global NODE_COUNTER
    num = NODE_COUNTER
    NODE_COUNTER += 1
    return num


class Node:
    def __init__(self, level=2, tokens=None, has_eos=True, children=None,root_md5=None):
        if children is None:
            children = []

        self.id = None  # pre order id
        self.children = children
        self.level = level
        self.tokens = tokens
        self.random_tokens = None #for coherence
        self.vector = None
        self.distinct_lookup_id = None
        self.random_lookup_id = None #for coherence
        self.mlm_loss = None
        self.mlm_diff_loss = None
        self.coherence_loss = None
        self.reconstruction_loss = None
        self.reconstruction_diff_loss = None
        self.rc_loss = None
        self.re_loss = None
        self.rj_loss = None
        self.rm_loss = None
        self.rm_diff_loss = None
        self.root_md5 = root_md5
        self.has_eos = has_eos

        self.is_dummy = False

    def is_join(self):
        return self.tokens == -1

    def get_padded_word_tokens(self):
        return self.tokens + [Config.pad_token_id] * (Config.sequence_lengths[0] - len(self.tokens))
    def get_padded_random_tokens(self):
        return self.random_tokens + [Config.pad_token_id] * (Config.sequence_lengths[0] - len(self.tokens))

    def set_vector(self, v):
        self.vector = v

    @staticmethod
    def join_children(node):
        """
        Join children nodes into one if they can fit within the max length.
        This is just for level >= 2
        """
        max_length = Config.sequence_lengths[node.level - 1]
        new_children = [node.children[0]]
        # TODO: Figure out if need to set has_eos to False on this (probably not?)
        for i, child in enumerate(node.children[1:]):
            if len(new_children[-1].children) + 1 + len(child.children) < max_length:
                # tokens = -1 because when join is on we will add 3 (3 special tokens)
                new_children[-1].children.append(Node(level=child.level - 1, tokens=-1, root_md5=node.root_md5))  # TODO: WHY minus one level?
                new_children[-1].children += child.children
            else:
                new_children.append(child)
        node.children = new_children

    def split_words(self):
        max_length = Config.sequence_lengths[self.level - 1]
        new_words = []
        for word in self.children:
            if len(word.tokens) <= max_length:
                new_words.append(word)
                continue

            parts = [word.tokens[i:i + max_length] for i in range(0, len(word.tokens), max_length)]
            parts = [Node(level=word.level, tokens=part, has_eos=i == len(parts) - 1, root_md5=self.root_md5) for i, part in enumerate(parts)]
            new_words.extend(parts)
        self.children = new_words

    @staticmethod
    def split_node(node):
        """
        Split nodes that have too many children.
        (Disabled) Don't allow the last child to not have an EoS token.
        """
        max_length = Config.sequence_lengths[node.level]
        if len(node.children) <= max_length:
            return [node]

        result = []
        current = []
        for child in node.children:
            current.append(child)

            if len(current) == max_length:
                # # Don't split on a non EoS token
                # eos_indices = [b.has_eos for b in current]
                # try:
                #     last_eos_index = len(eos_indices) - 1 - eos_indices[::-1].index(True)
                # except ValueError:
                #     raise ValueError('The ')
                #
                # current_use = current[:last_eos_index + 1]
                # current = current[last_eos_index + 1:]

                result.append(Node(level=node.level, children=current, has_eos=False, root_md5=node.root_md5))
                current = []

        if len(current) > 0:
            result.append(Node(level=node.level, children=current, has_eos=False, root_md5=node.root_md5))
        result[-1].has_eos = True

        return result

    def expand_struct(self, struct,children_md5s=False):
        if self.level == 0:  # Word
            self.tokens = struct
            self.id = get_unique_id()
            return

        self.children = []
        for i,part in enumerate(struct):
            node = Node(level=self.level - 1)
            node.expand_struct(part)
            self.children.append(node)
            if children_md5s:
              node.root_md5 = children_md5s[i]
            else:
              node.root_md5 = self.root_md5

        if self.level == 1:
            self.split_words()
        else:
            # Join above level 2 but never at the root
            # TODO - Maybe should enable the root?
            if Config.join_texts is True and self.level >= 2 and self.level != Config.agent_level + 1:
                self.join_children(self)
            self.children = [node for child in self.children for node in self.split_node(child)]

        for child in self.children:
            child.id = get_unique_id()

    # For debugging purposes to get the struct from the current node
    def build_struct(self, return_eos=False):
        if self.is_join():
            if return_eos:
                return -1, True
            return -1

        if self.level == 0:
            if return_eos:
                return self.tokens, self.has_eos
            return self.tokens

        result = [child.build_struct(return_eos=return_eos) for child in self.children]
        if return_eos:
            return result, self.has_eos

        return result


if __name__ == '__main__':
    # root = Node(level=1)
    # root.expand_struct(struct=[[10, 11, 12, 13, 2]])

    root = Node(level=2)
    root.id = 0
    root.expand_struct(struct=[[[10, 11, 12, 13, 2], [14, 15, 16, 2], [17, 18, 19, 20, 21, 22, 2]]])
    print('Evaluate')
    print(root.children)
    print(root.build_struct())
