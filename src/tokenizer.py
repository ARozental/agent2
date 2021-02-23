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

    def __init__(self):
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

    def tokenize(self, word, pad_length):
        pad_length += 1  # Add an extra character for the EOS token

        tokens = [self.tokenizer[char] for char in word] + [self.SPECIAL_INDICES['[EOS]']]
        mask = [0 for _ in range(len(tokens))]
        if len(tokens) < pad_length:
            pad_extra = [self.SPECIAL_INDICES['[PAD]']] * (pad_length - len(tokens))
            pad_mask = [1 for _ in range(len(pad_extra))]

            tokens = tokens + pad_extra
            mask = mask + pad_mask

        return tokens, mask

    def decode(self, tokens):
        if not isinstance(tokens, list):
            tokens = tokens.tolist()

        try:
            eos_index = tokens.index(self.SPECIAL_INDICES['[EOS]'])
            if eos_index is not None:
                tokens = tokens[:eos_index]
        except ValueError:
            pass

        return [self.reverse_tokenizer[token] for token in tokens]
