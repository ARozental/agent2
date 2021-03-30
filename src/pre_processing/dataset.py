from src.pre_processing import TreeTokenizer
from torch.utils.data import IterableDataset
import glob


class Dataset(IterableDataset):
    def __init__(self, folder, max_num=None):
        self.data = glob.glob(folder)
        if max_num is not None:
            self.data = self.data[:max_num]

        self.init_tree_tokenizer()
        TreeTokenizer.finalize()

    def init_tree_tokenizer(self):
        """
        Set the split functions in here
        """
        raise NotImplementedError

    def _read_file(self, file):
        with open(file, encoding='utf-8') as f:
            data = f.read()
        return data

    def __getitem__(self, index):
        file = self.data[index]
        return self._read_file(file)

    def __iter__(self):
        for file in self.data:
            yield self._read_file(file)
