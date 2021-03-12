from src.tokenizer import Tokenizer


class SimpleDataset:
    def __init__(self, max_level=1):
        with open('../dummy_data1.txt') as f:
            self.text = f.readlines()

        self.max_level = max_level
        self.tokenizer = Tokenizer()

        # TODO - Make this dynamic based on the number of levels
        if max_level == 2:
            self.text = 'We went to the store\nThis is a wonderful test'
            self.text = [[[char for char in word] for word in sent.split(' ')] for sent in self.text.split('\n')]
        else:
            self.text = 'something\nencyclopedia'
            self.text = [[char for char in word] for word in self.text.split('\n')]

    def iterator(self):
        yield [self.tokenizer.tokenize(item) for item in self.text]

    def debug_examples(self, level, num=2):
        if level == 1:
            for ex in self.iterator():
                return ex
        elif level == 0:
            if self.max_level == 2:
                for ex in self.iterator():
                    return [ex[0][0], ex[1][0]]
            elif self.max_level == 1:
                for ex in self.iterator():
                    return ex

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)

    def num_tokens(self):
        return len(self.tokenizer.tokenizer)


if __name__ == '__main__':
    dataset = SimpleDataset()
