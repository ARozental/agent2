from src.config import Config
import numpy as np
import random
import torch
import os
import math
import torch.nn.functional as F
import hashlib
import collections


def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def metsumm(stepno=''):
    if not Config.use_tpu:
        return

    import torch_xla.debug.metrics as met
    x = met.metrics_report().split('\n')
    for i, line in enumerate(x):
        if 'CompileTime' in line or 'aten::' in line:
            key = line.split()[-1]
            value = x[i + 1].split()[-1]
            print('step {}, key {}, value {}'.format(stepno, key, value))
            print('')


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
    if Config.use_tpu:  # Ignore if using the TPU
        yield items
    else:
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


def split_nodes_to_batches(nodes, max_batch_size):
    return []


def attention(q, k, v,d_k, real_positions=None, dropout=None):
    """ for pndb only"""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if real_positions is not None:
      scores += torch.log(real_positions.unsqueeze(1))

    scores = F.softmax(scores, dim=-1)

    if dropout is not None:
      scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def md5(s):
    return hashlib.md5(s.encode('utf-8')).hexdigest()[0:5]


# WTF no one on line knows how to do it?
def earth_movers_distance(l, p):
    v = torch.cumsum(l - p, -1) - (l - p)
    return (v * torch.sign(v)).sum(-1)


def inverse_loss(loss):
    "0.6931471805599453 is loss for 50:50 => this function is not doing what it should, probably"
    return -torch.max(loss - 0.6931471805599453, loss * 0)


def cap_loss(loss):
    "cap loss and effectivly kill gradient to prevent the classifier from winning completely"
    return torch.max(loss, (loss * 0) + 0.05)


def merge_dicts(d1, d2):
    ""
    res = {}
    for level in d1.keys():
        res[level] = {k: d1[level].get(k, 0) + d2[level].get(k, 0) for k in d1[level].keys()}
    return res


def group_by_root(nodes):
  ks = set([n.root_md5 for n in nodes])
  res = {k: [] for k in ks}
  for n in nodes:
    res[n.root_md5].append(n)
  return res

def distinct(lst):
  s = set([])
  output = []
  for x in lst:
    if x not in s:
      s.add(x)
      output.append(x)
  return output
