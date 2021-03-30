from src.config import Config
from src.pre_processing.dataset import Dataset
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

        # Strip if not doing chapter or book level
        if Config.agent_level < Config.levels['CHAPTER']:
            data = re.sub('Chapter .*\n', '', data)

        data = data.split(' ')
        # print(' '.join(data[:5]))
        return ' '.join(data[:5])
        # data = data.split('. ')
        # return data[0]
        return data[:25]
