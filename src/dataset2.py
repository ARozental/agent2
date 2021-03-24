from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import glob
import re
import torch
from itertools import cycle
from src.pre_processing import TreeTokenizer



class Dataset2(torch.utils.data.IterableDataset):
  # A pytorch dataset class for holding data for a text classification task.
  def __init__(self, folder_name,batch_size=2,level=2):
    self.file_names = cycle(glob.glob(folder_name + "/*.txt"))
    self.i = self.iter1(self.file_names)
    self.junk = cycle([7])
    self.batch_size = batch_size
    self.tree_tokenizer = TreeTokenizer()

  def read_text(self,file_name):
    f = open(file_name, "r")
    return f.read()

  def iter1(self,data):
    for _ in self.junk:
      out = []
      for i in range(self.batch_size):
        out.append(self.read_text(next(self.file_names)))
      #yield out
      yield self.tree_tokenizer.batch_texts_to_trees(out)

  def __iter__(self):
    return self.i

dd=Dataset2("../datasets/dummy_dataset",batch_size=2)
# can't use dataloader because: default_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found
print(next(dd.i))
print(next(dd.i))
print(next(dd.i))

