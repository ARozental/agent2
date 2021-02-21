from src.tokenizer import Tokenizer


class SimpleDataset:
    def __init__(self):
        with open('../dummy_data1.txt') as f:
            self.text = f.readlines()
        self.text = 'The quick brown fox jumps over the lazy dog'
        #self.text = self.text.split(' ')
        self.text = [c for c in self.text]

        self.tokenizer = Tokenizer()
        self.data = [self.tokenizer.tokenize(item) for item in self.text]

    def iterator(self, entire_seq=False):
        if entire_seq:
            yield [token for word in self.data for token in word]
        else:
            for item in self.data:
                yield item

    def decode(self, tokens):
        return self.tokenizer.decode(tokens)


if __name__ == '__main__':
    dataset = SimpleDataset()
