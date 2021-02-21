class Tokenizer:
    def __init__(self):
        with open('../chars.txt', encoding='utf-8') as f:
            chars = [char.strip('\r\n') for char in f.readlines()]
            #chars = [char for char in f.readlines()]

        self.tokenizer = {char: i for i, char in enumerate(chars)}
        self.reverse_tokenizer = {i: char for i, char in enumerate(chars)}

    def tokenize(self, word):
        return [self.tokenizer[char] for char in word]

    def decode(self, tokens):
        return [self.reverse_tokenizer[token] for token in tokens]
