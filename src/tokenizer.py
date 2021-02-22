class Tokenizer:
    def __init__(self):
        with open('../chars.txt', encoding='utf-8') as f:
            chars = [char.strip('\r\n') for char in f.readlines()]
            #chars = [char for char in f.readlines()]

        self.tokenizer = {char: i for i, char in enumerate(chars)}
        self.reverse_tokenizer = {i: char for i, char in enumerate(chars)}

    def tokenize(self, word, pad_length):
        tokens = [self.tokenizer[char] for char in word]
        mask = [0 for _ in range(len(tokens))]
        if len(tokens) < pad_length:
            pad_extra = [self.tokenizer['[PAD]']] * (pad_length - len(tokens))
            pad_mask = [1 for _ in range(len(pad_extra))]

            tokens = tokens + pad_extra
            mask = mask + pad_mask

        return tokens, mask

    def decode(self, tokens):
        return [self.reverse_tokenizer[token] for token in tokens]
