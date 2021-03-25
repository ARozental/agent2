# Find the number of levels automatically
import  random
import  os
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

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
    #with regular or default dict
    for k, v in d.items():
        if type(d) != type(d[k]):
            d[k] = fn(v)
        else:
            d[k] = change_dict(d[k], fn)
    return d


def find_level(inputs):
    current = inputs[0]
    level = 0
    while isinstance(current, list):
        current = current[0]
        level += 1

    return level


def attention(q, k, v, d_k, mask=None, dropout=None):
  scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
  # print("s",scores)
  # print("mask",mask)

  if mask is not None:
    mask = mask.unsqueeze(1)
    scores = scores.masked_fill(mask == True, -1e9)

  scores = F.softmax(scores, dim=-1)
  # print("scores", scores)

  if dropout is not None:
    scores = dropout(scores)

  output = torch.matmul(scores, v)

  return output