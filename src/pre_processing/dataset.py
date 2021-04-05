from src.pre_processing import TreeTokenizer
from torch.utils.data import IterableDataset
import torch
import glob


# If using workers then make each dataset worker only process a certain chunk
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    num_workers = worker_info.num_workers

    dataset.data = dataset.data[worker_id::num_workers]
    dataset.init_tree_tokenizer()


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
