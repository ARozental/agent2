from src.dataset import Dataset
import os
import re


class BookDataset(Dataset):
    def __init__(self, no_stats=False, **kwargs):
        if no_stats:
            folder = os.path.join('datasets', 'no_stats_books', '*.txt')
        else:
            raise NotImplementedError
        super().__init__(folder, **kwargs)

    def _read_file(self, file):
        data = super()._read_file(file)
        data = re.sub('Chapter .*\n', '', data)  # Strip if not doing book level
        return data[:300]

    def _process(self, data):
        return self.tree_tokenizer.batch_texts_to_trees(data)
