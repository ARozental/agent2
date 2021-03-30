from src.pre_processing import Splitters, TreeTokenizer
from src.pre_processing.dataset import Dataset
import os


class DummyDataset(Dataset):
    def __init__(self, **kwargs):
        super().__init__(folder=os.path.join('datasets', 'dummy', '*.txt'), **kwargs)

    def init_tree_tokenizer(self):
        TreeTokenizer.split_functions = [
            Splitters.sentence_to_words,
            Splitters.paragraph_to_sentences,
        ]
