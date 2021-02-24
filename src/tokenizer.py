class Tokenizer:
    SPECIAL_TOKENS = {
        '[PAD]': '[PAD]',
        '[MASK]': '[MASK]',
        '[EOS]': '.',
        '[SEP]': ' ',
    }

    SPECIAL_INDICES = {
        '[PAD]': -1,
        '[MASK]': -1,
        '[EOS]': -1,
        '[SEP]': -1,
    }

    def __init__(self, max_lengths):
        with open('../chars.txt', encoding='utf-8') as f:
            chars = [char.strip() for char in f.readlines()]

        self.tokenizer = {}
        self.reverse_tokenizer = {}
        for i, char in enumerate(chars):
            special_display = self.SPECIAL_TOKENS.get(char, None)
            if special_display is not None:
                self.SPECIAL_INDICES[char] = i
                self.tokenizer[special_display] = i
                self.reverse_tokenizer[i] = special_display
            else:
                self.tokenizer[char] = i
                self.reverse_tokenizer[i] = char

        self.max_lengths = max_lengths

    def build_empty(self, level):
        if level == 0:
            return self.SPECIAL_INDICES['[PAD]'], 1

        tokens, mask = self.build_empty(level - 1)
        tokens = [tokens] * self.max_lengths[level - 1]
        mask = [mask] * self.max_lengths[level - 1]

        return tokens, mask

    def tokenize(self, seq, level=None):
        # Find the number of levels automatically
        if level is None:
            current = seq
            level = 0
            while isinstance(current[0], list):
                current = current[0]
                level += 1

        if level == 0:  # Word level
            tokens = [self.tokenizer[char] for char in seq] + [self.SPECIAL_INDICES['[EOS]']]
            mask = [0 for _ in range(len(tokens))]
        else:  # All other levels
            results = [self.tokenize(item, level=level - 1) for item in seq]
            tokens = [r[0] for r in results]
            mask = [r[1] for r in results]

        if len(tokens) < self.max_lengths[level]:
            empty_token, empty_mask = self.build_empty(level)
            tokens = tokens + [empty_token] * (self.max_lengths[level] - len(tokens))
            mask = mask + [empty_mask] * (self.max_lengths[level] - len(mask))

        return tokens, mask

    def decode(self, seq, level=None):
        if not isinstance(seq, list):
            seq = seq.tolist()

        if level is None:
            current = seq
            level = 0
            while isinstance(current[0], list):
                current = current[0]
                level += 1

        if level == 0:
            try:
                eos_index = seq.index(self.SPECIAL_INDICES['[EOS]'])
                if eos_index is not None:
                    seq = seq[:eos_index]
            except ValueError:
                pass

            return ''.join([self.reverse_tokenizer[token] for token in seq])

        return ' '.join([self.decode(word, level=level-1) for word in seq])
