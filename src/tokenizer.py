from src.utils import find_level


class Tokenizer:
    SPECIAL_TOKENS = {
        '[PAD]': '[PAD]',
        '[MASK]': '[MASK]',
        '[EOS]': '.',
    }

    SPECIAL_INDICES = {
        '[PAD]': -1,
        '[MASK]': -1,
        '[EOS]': -1,
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

    def tokenize(self, seq, level=None):
        if level is None:
            level = find_level(seq)

        if level == 0:  # Word level
            tokens = [self.tokenizer[char] for char in seq]
        else:  # All other levels
            tokens = [self.tokenize(item, level=level - 1) for item in seq]

        return tokens

    def decode(self, seq, level=None):
        if not isinstance(seq, list):
            seq = seq.tolist()

        if level is None:
            level = find_level(seq)

        if level == 0:
            try:
                eos_index = seq.index(self.SPECIAL_INDICES['[EOS]'])
                if eos_index is not None:
                    seq = seq[:eos_index]
            except ValueError:
                pass

            return ''.join([self.reverse_tokenizer[token] for token in seq])

        return ' '.join([self.decode(word, level=level - 1) for word in seq])
