from torch.utils.data import IterableDataset
from src.config import Config
import torch


# If using workers then make each dataset worker only process a certain chunk
def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    if Config.use_accelerator:
        dataset = dataset.dataset
    num_workers = worker_info.num_workers

    if dataset.divide_data and (Config.max_dataset_len is None or (
            Config.max_dataset_len > (Config.batch_size * Config.num_dataset_workers))):
        dataset.data = dataset.data[worker_id::num_workers]

    dataset.init_tree_tokenizer()


class Dataset(IterableDataset):
    def __init__(self, divide_data=True, **kwargs):
        self.init_tree_tokenizer()
        self.divide_data = divide_data

    def init_tree_tokenizer(self):
        """
        Set the split functions in here
        """
        raise NotImplementedError

    def __getitem__(self, index):
        raise NotImplementedError

    def __iter__(self):
        raise NotImplementedError
