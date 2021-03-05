import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import reduce
import math


# Some code copied from: https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
class MLMLoss(nn.Module):
    def __init__(self, model, mask_prob=0.15, replace_prob=0.9, num_tokens=None, random_token_prob=0., mask_token_id=2,
                 pad_token_id=0, mask_ignore_token_ids=[]):
        """
        Keep tokens the same with probability 1 - replace_prob
        @param model: The current AgentLevel
        @param mask_prob:
        @param replace_prob:
        @param num_tokens:
        @param random_token_prob:
        @param mask_token_id:
        @param pad_token_id:
        @param mask_ignore_token_ids:
        """
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
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])  # TODO - Add the EOS token to this

    @staticmethod
    def prob_mask_like(t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    @staticmethod
    def mask_with_tokens(t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    @staticmethod
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

    def make_masked_input(self, inputs):
        no_mask = self.mask_with_tokens(inputs, self.mask_ignore_token_ids)
        mask = self.get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # Clone to convert the inputs
        masked_input = inputs.clone().detach()

        # if random token probability > 0 for mlm
        if self.random_token_prob > 0:
            # num_tokens keyword must be supplied when instantiating MLM if using random token replacement
            assert self.num_tokens is not None
            random_token_prob = self.prob_mask_like(inputs, self.random_token_prob)
            random_tokens = torch.randint(0, self.num_tokens, inputs.shape, device=inputs.device)
            random_no_mask = self.mask_with_tokens(random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            random_indices = torch.nonzero(random_token_prob, as_tuple=True)
            masked_input[random_indices] = random_tokens[random_indices]

        # [mask] input
        replace_prob = self.prob_mask_like(inputs, self.replace_prob)
        masked_input = masked_input.masked_fill(mask * replace_prob, self.mask_token_id)

        # set inverse of mask to padding tokens for labels
        labels = inputs.masked_fill(~mask, self.pad_token_id)

        return masked_input, labels

    def forward(self, inputs, model_mask):
        masked_input, labels = self.make_masked_input(inputs)

        # Run the model
        encoded = self.model.encoder(masked_input, model_mask)
        encoded = encoded.transpose(0, 1)
        output = self.model.encoder_transform(encoded)
        emb_weight = torch.transpose(self.model.encoder.embedding.weight, 0, 1).unsqueeze(0)
        logits = torch.matmul(output, emb_weight)  # [batch, seq_length, num_tokens]
        logits = logits.transpose(1, 0)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.pad_token_id
        )

        return mlm_loss
