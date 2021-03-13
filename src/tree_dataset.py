from src.pre_processing import TreeTokenizer


class TreeDataset:
    def __init__(self):
        self.tree_tokenizer = TreeTokenizer()
        self.texts = ["I like big butts. I can not lie.", "some other song"]  # hard coded for tests

    def iterator(self):
        yield self.tree_tokenizer.batch_texts_to_trees(self.texts)


if __name__ == '__main__':
    dataset = TreeDataset()
    print(dataset.iterator())
