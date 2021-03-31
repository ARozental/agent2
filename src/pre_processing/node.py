from src.config import Config


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

        # Focus on shortening the tokens, the EOS token should always be at the end no matter what.
        max_length = Config.sequence_lengths[0]
        tokens = self.tokens[:max_length - 1] + [Config.eos_token_id]
        return tokens + [Config.pad_token_id] * (max_length - len(tokens))

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
