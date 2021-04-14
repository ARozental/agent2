import numpy as np
import math
import random
import torch
import os
import torch.nn as nn
import math
import torch.nn.functional as F
import hashlib

def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def change_dict(d, fn):
    # with regular or default dict
    for k, v in d.items():
        if type(d) != type(d[k]):
            d[k] = fn(v)
        else:
            d[k] = change_dict(d[k], fn)
    return d


# Source: https://stackoverflow.com/a/21767522/556935
def iter_even_split(items, batch_size):
    """
    generates balanced baskets from iterable, contiguous contents
    provide item_count if providing a iterator that doesn't support len()
    """
    item_count = len(items)
    max_baskets = math.ceil(len(items) / batch_size)
    baskets = min(item_count, max_baskets)
    items = iter(items)
    floor = item_count // baskets
    ceiling = floor + 1
    stepdown = item_count % baskets
    for x_i in range(baskets):
        length = ceiling if x_i < stepdown else floor
        yield [items.__next__() for _ in range(length)]


def split_nodes_to_batches(nodes,max_batch_size):

  return []


def attention(q, k, v, d_k, mask=None, dropout=None):
  """ for pndb only"""
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == True, -1e9)

  scores = F.softmax(scores, dim=-1)

  if dropout is not None:
    scores = dropout(scores)

  output = torch.matmul(scores, v)

  return output

def md5(s):
  return hashlib.md5(s.encode('utf-8')).hexdigest()
