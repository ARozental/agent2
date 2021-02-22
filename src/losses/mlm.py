import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math
import numpy as np


def prob_mask_like(t, prob):
    return torch.zeros_like(t).float().uniform_(0, 1) < prob


def mask_with_tokens(t, token_ids):
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


def get_mask_subset_with_prob(mask, prob):
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)

    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]

    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


def tok_to_rand(c, same=0.1, rand_char=0.1):
    r = np.random.rand()
    if r < same:
        return c
    elif r < same + rand_char:
        return np.random.randint(0, 80)  # embedding_size
    else:
        return 1  # [MASK]


tok_to_rand = np.vectorize(tok_to_rand)


def make_masked_sequence(seq, input_mask, mask_prob=0.2):
    seq = seq.detach().cpu().numpy()
    input_mask = input_mask.detach().cpu().numpy()
    mlm_mask = np.ceil(np.random.rand(seq.shape[0], seq.shape[1]) - 1.0 + mask_prob)  # position where loss counts
    replacments = tok_to_rand(seq)  # replacment tokens
    masked_seq = (seq * (1 - mlm_mask) + (replacments * mlm_mask)) * input_mask  # new sequence
    return torch.tensor(mlm_mask * input_mask, dtype=torch.int64), torch.tensor(masked_seq,
                                                                                dtype=torch.int64)  # this tensor has (and shouldn't have) a gradient


# Some code copied from: https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
class MLMLoss(nn.Module):
    def __init__(self, model, mask_prob=0.15, replace_prob=0.9, num_tokens=None, random_token_prob=0., mask_token_id=2,
                 pad_token_id=0, mask_ignore_token_ids=[]):
        super().__init__()

        self.model = model

        # mlm related probabilities
        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        # token ids
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, inputs, model_mask):
        mlm_mask, masked_input = make_masked_sequence(inputs, ~model_mask, mask_prob=0.5)
        labels = inputs * mlm_mask
        logits = self.model.mlm(masked_input, model_mask)
        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.pad_token_id
        )

        return mlm_loss

        # no_mask = mask_with_tokens(inputs, self.mask_ignore_token_ids)
        # mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)
        # # get mask indices
        # mask_indices = torch.nonzero(mask, as_tuple=True)
        #
        #
        # # mask input with mask tokens with probability of `replace_prob` (keep tokens the same with probability 1 - replace_prob)
        # masked_input = inputs.clone().detach()
        # #print("masked_input", masked_input.shape)
        #
        # # if random token probability > 0 for mlm
        # if self.random_token_prob > 0:
        #     assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
        #     random_token_prob = prob_mask_like(inputs, self.random_token_prob)
        #     random_tokens = torch.randint(0, self.num_tokens, inputs.shape, device=inputs.device)
        #     random_no_mask = mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
        #     random_token_prob &= ~random_no_mask
        #     random_indices = torch.nonzero(random_token_prob, as_tuple=True)
        #     masked_input[random_indices] = random_tokens[random_indices]
        #
        # # [mask] input
        # replace_prob = prob_mask_like(inputs, self.replace_prob)
        # masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)
        #
        # # set inverse of mask to padding tokens for labels
        # labels = inputs.masked_fill(~mask, self.pad_token_id)
        #
        # logits = self.model(masked_input.transpose(0, 1), model_mask)
        # mlm_loss = F.cross_entropy(
        #     logits.transpose(1, 0).transpose(1, 2),
        #     labels,
        #     ignore_index=self.pad_token_id
        # )
        # print('mlm_loss', mlm_loss)
        #
        # return mlm_loss
