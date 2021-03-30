from src.pre_processing import TreeTokenizer
from torch.utils.data import IterableDataset
import glob


class Dataset(IterableDataset):
    def __init__(self, folder, batch_size=2, max_num=None):
        self.tree_tokenizer = TreeTokenizer()
        self.batch_size = batch_size

        self.data = glob.glob(folder)
        if max_num is not None:
            self.data = self.data[:max_num]

    def _process(self, file):
        raise NotImplementedError

    def _read_file(self, file):
        with open(file, encoding='utf-8') as f:
            data = f.read()
        return data

    def __getitem__(self, index):
        file = self.data[index]
        yield self._process([self._read_file(file)])

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            files = self.data[i:i + self.batch_size]
            data = [self._read_file(file) for file in files]
            yield self._process(data)
