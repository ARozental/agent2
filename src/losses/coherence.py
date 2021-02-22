import torch.nn as nn
import torch


class CoherenceLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pad_token_id = 0

        self.loss = nn.MSELoss()
        self.replace_prob = 0.5

    # TODO - Don't replace the indices of the special token
    # TODO - Clean this up
    def forward(self, inputs, mask):
        unique_values = inputs.unique()
        unique_values = unique_values[unique_values != self.pad_token_id]

        do_replace = (torch.rand((inputs.size(0), 1)) > self.replace_prob)

        target_probs = torch.rand((inputs.size(0), 1)) * do_replace
        prob_each = target_probs.repeat((1, inputs.size(1)))
        replace_each = do_replace.repeat((1, inputs.size(1)))

        token_prob = torch.rand((inputs.size(0), inputs.size(1)))

        token_replace = token_prob > prob_each
        token_replace = token_replace * replace_each

        random_tokens = torch.randint(0, unique_values.size(0), size=inputs.size())

        replaced_inputs = (inputs * ~token_replace) + (token_replace * random_tokens)

        # The mask should handle the padding tokens (even if they were replaced)
        preds = self.model.coherence(replaced_inputs, mask)

        return self.loss(preds * replace_each, target_probs)
