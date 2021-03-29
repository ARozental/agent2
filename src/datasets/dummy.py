from src.dataset import Dataset
import os


class DummyDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(folder=os.path.join('datasets', 'dummy', '*.txt'), **kwargs)

    def _process(self, data):
        return self.tree_tokenizer.batch_texts_to_trees(data)
