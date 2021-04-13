from src.config import Config
from src.datasets import BookDataset, DummyDataset, WikiDataset
from src.pre_processing import TreeTokenizer, worker_init_fn
from src.utils import seed_torch
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd

"""
Run this to compute the stats per level for a dataset.
"""

seed_torch(0)  # 0 learns 2 doesn't (before no cnn layer)


# Need to wrap in a function for the child workers
def calculate():
    # dataset = DummyDataset(max_num=2)
    # dataset = BookDataset(no_stats=True, max_num=2)
    dataset = WikiDataset(max_num=50000)

    dataloader = DataLoader(
        dataset,
        batch_size=Config.batch_size,
        collate_fn=TreeTokenizer.compute_stats,
        worker_init_fn=worker_init_fn,
        num_workers=0,
        # persistent_workers=True  # This is helpful when num_workers > 0
    )

    num_batches = len(dataloader)
    tenth = int(num_batches / 10)  # Print progress every 10%

    # TODO: Count number of unique characters and words
    stats = {level: [] for level in range(Config.agent_level + 1)}
    for step, batch in enumerate(dataloader):
        for level in range(Config.agent_level + 1):
            stats[level] += batch[level]

        if step % tenth == 0 and step > 0:
            print('Batch', step, 'of', num_batches)

    for level in range(Config.agent_level + 1):
        print('')
        print('')
        print('-' * 10)
        print('Level', level)
        print(pd.Series(stats[level]).describe())
        print('95th percentile', np.percentile(stats[level], 95))


if __name__ == '__main__':
    calculate()
