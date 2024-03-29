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


def prob_to_logit(p):
  p = torch.clip(p,min=0.000001,max=0.999999)
  return torch.log(p / (1 - p))



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


def attention(q, k, v, d_k, real_positions=None, dropout=None):
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


# WTF no one on stack overflow knows how to do it?
def earth_movers_distance(l, p):
    v = torch.cumsum(l - p, -1) - (l - p)
    return (v * torch.sign(v)).sum(-1)


def earth_movers_distance2(l, p):
    #loss per position
    sizes = torch.arange(l.size()[-1], device=l.device).unsqueeze(0) - torch.argmax(l)
    sizes = sizes * torch.sign(sizes)
    dist = l - p
    dist = dist * torch.sign(dist)
    res = (sizes * dist).sum(-1)
    return res


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


# todo: make smarter, current level 1 solution can create 2 small buckets that can be merged
def node_batch_to_small_batches(node_batch, level):
    max_size = Config.node_sizes[level]
    if level == 1:
        node_batchs = list(group_by_root(node_batch).values())
    else:
        node_batchs = [node_batch]
    temp_res = []
    res = []
    while node_batchs:
        batch = node_batchs.pop()
        if len(batch) == max_size:
            res.append(batch)
        elif len(batch) > max_size:
            res.append(batch[:max_size])
            node_batchs.append(batch[max_size:])
        elif len(temp_res) + len(batch) <= max_size:
            temp_res.extend(batch)
        else:
            res.append(temp_res)
            temp_res = batch
    if temp_res:
        res.append(temp_res)
    return res


def distinct(lst):
    s = set([])
    output = []
    for x in lst:
        if x not in s:
            s.add(x)
            output.append(x)
    return output


def make_noise(t, noise):
    noise = torch.min(noise, noise / noise) * t.norm(dim=[-1]).mean()  # capped at 1
    changed_examples = torch.rand(t.shape[0], 1, device=t.device).round()
    n = torch.normal(torch.mean(t).data, torch.std(t).data, size=t.shape, device=t.device)
    return t + noise * changed_examples * n / 20.0 #because with 1.0 it is wayyy too noisy; does reconstruction1 without eos0??? #it is true??? => kee p train and see


def apply_recursive(func, obj):
    if isinstance(obj, dict):  # if dict, apply to each key
        return {k: apply_recursive(func, v) for k, v in obj.items()}
    elif isinstance(obj, list):  # if list, apply to each element
        return [apply_recursive(func, elem) for elem in obj]
    else:
        return func(obj)


def prepare_inputs(inputs, squeeze=False, to_device=True):
    for parent_key, values in inputs.items():
        for key, value in values.items():
            if squeeze:
                inputs[parent_key][key] = value.squeeze(0)
            if Config.use_cuda and to_device:
                inputs[parent_key][key] = inputs[parent_key][key].to(Config.device)
    return inputs


def recycle_weights(new_untrained_model, old_trained_model):
  new_untrained_model['loss_weights'] = '{"0": {"m": 0.001, "rm": 0.0001}, "1": {"m": 0.01, "rm": 0.001, "e": 0.3, "re":0.6}, "c": 0.2, "r": 0.01, "e": 0.01, "j": 1e-08, "d": 0.03, "rc": 0.0, "re": 0.02, "rj": 1.001e-07}'
  new_untrained_model['loss_weights'] = '{"0": {"m": 0.00005, "d": 0.0, "c": 0.05, "r": 0.03, "e": 0.0005, "re": 0.005},' \
                                        ' "1": {"m": 0.02, "d": 0.08, "c": 0.002, "r": 0.05, "e": 0.0005, "re": 0.005},' \
                                        ' "rj": 0.0,"j": 0.0,"rm": 0.0,"rc": 0.0}'
  for k in new_untrained_model['model'].keys():
    if k in old_trained_model['model'].keys():
      if old_trained_model['model'][k].shape == new_untrained_model['model'][k].shape:
        new_untrained_model['model'][k] = old_trained_model['model'][k]
      else:
        # print(k,old_trained_model['model'][k].shape,new_untrained_model['model'][k].shape)
        if len(new_untrained_model['model'][k].shape) == 1:
          good_weights = old_trained_model['model'][k][
                         :(min(old_trained_model['model'][k].shape[0], new_untrained_model['model'][k].shape[0]))]
          other_weights = new_untrained_model['model'][k] / 10  # make the random weights small but not 0
          new_weights = torch.cat([good_weights, other_weights])[:new_untrained_model['model'][k].shape[0]]
          new_untrained_model['model'][k] = new_weights
        elif len(new_untrained_model['model'][k].shape) == 2:
          s0 = (min(old_trained_model['model'][k].shape[0], new_untrained_model['model'][k].shape[0]))
          s1 = (min(old_trained_model['model'][k].shape[1], new_untrained_model['model'][k].shape[1]))
          good_weights = old_trained_model['model'][k][:s0, :s1]
          other_weights = new_untrained_model['model'][k] / 10  # make the random weights small but not 0
          new_weights = torch.cat([good_weights, other_weights[:, :s1]])[:new_untrained_model['model'][k].shape[0]]
          if new_weights.shape != new_untrained_model['model'][k].shape:
            raise "2 different shapes bigger and smaller"

          new_untrained_model['model'][k] = new_weights
        else:
          raise "Y U 3d"
    else:
      if "bias" in k:
        new_untrained_model['model'][k] = new_untrained_model['model'][k] - 1
      elif "norm" in k:
        pass
      else:
        new_untrained_model['model'][k] = new_untrained_model['model'][k] / 10
