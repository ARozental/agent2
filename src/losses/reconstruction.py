import torch.nn.functional as F
import torch.nn as nn


class ReconstructionLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.pad_token_id = 0

    # TODO - Fix this and refactor some more
    def forward(self, inputs, mask):
        logits = self.model.reconstruct(inputs, mask)

        return F.cross_entropy(
            logits.transpose(1, 2),
            inputs,
            ignore_index=self.pad_token_id
        )
