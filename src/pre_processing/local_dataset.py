from src.pre_processing.dataset import Dataset
import glob


# noinspection PyAbstractClass
class LocalDataset(Dataset):
    def __init__(self, folder, max_num=None, **kwargs):
        super().__init__(**kwargs)
        self.data = glob.glob(folder)
        if max_num is not None:
            self.data = self.data[:max_num]

    def _read_file(self, file):
        with open(file, encoding='utf-8') as f:
            data = f.read()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file = self.data[index]
        return self._read_file(file)

    def __iter__(self):
        for file in self.data:
            yield self._read_file(file)
