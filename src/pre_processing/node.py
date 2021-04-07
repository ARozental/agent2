from src.config import Config

NODE_COUNTER = 1


def get_unique_id():
    global NODE_COUNTER
    num = NODE_COUNTER
    NODE_COUNTER += 1
    return num


class Node:

    def __init__(self, level=2, tokens=None, has_eos=True, children=None):
        self.id = None  # pre order id
        self.children = children
        self.level = level
        self.tokens = tokens
        self.vector = None
        self.distinct_lookup_id = None
        self.mlm_loss = None
        self.coherence_loss = None
        self.reconstruction_loss = None
        self.reconstruction_diff_loss = None

        self.has_eos = has_eos

    def is_join(self):
        return self.tokens == -1

    def get_padded_word_tokens(self):
        return self.tokens + [Config.pad_token_id] * (Config.sequence_lengths[0] - len(self.tokens))

    def set_vector(self, v):
        self.vector = v

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

    def split_words(self):
        max_length = Config.sequence_lengths[self.level - 1]
        new_words = []
        for word in self.children:
            if len(word.tokens) <= max_length:
                new_words.append(word)
                continue

            parts = [word.tokens[i:i + max_length] for i in range(0, len(word.tokens), max_length)]
            parts = [Node(level=word.level, tokens=part, has_eos=i == len(parts) - 1) for i, part in enumerate(parts)]
            new_words.extend(parts)
        self.children = new_words

    @staticmethod
    def split_node(node):
        """
        Split nodes that have too many children.
        Don't allow the last child to not have an EoS token.
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

                result.append(Node(level=node.level, children=current, has_eos=False))
                current = []

        if len(current) > 0:
            result.append(Node(level=node.level, children=current, has_eos=False))
        result[-1].has_eos = True

        return result

    def expand_struct(self, struct):
        if self.level == 0:  # Word
            self.tokens = struct
            self.id = get_unique_id()
            return

        # if Config.join_texts is True and self.level >= 2 and self.type != "batch root":
        #     self.join_struct_short_children()

        self.children = []
        for part in struct:
            node = Node(level=self.level - 1)
            node.expand_struct(part)
            self.children.append(node)

        if self.level == 1:
            self.split_words()
        else:
            self.children = [node for child in self.children for node in self.split_node(child)]

        for child in self.children:
            child.id = get_unique_id()

    # For debugging purposes to get the struct from the current node
    def build_struct(self):
        if self.level == 0:
            return self.tokens

        return [child.build_struct() for child in self.children]


if __name__ == '__main__':
    # root = Node(level=1)
    # root.expand_struct(struct=[[10, 11, 12, 13, 2]])

    root = Node(level=2)
    root.id = 0
    root.expand_struct(struct=[[[10, 11, 12, 13, 2], [14, 15, 16, 2], [17, 18, 19, 20, 21, 22, 2]]])
    print('Evaluate')
    print(root.children)
    print(root.build_struct())
