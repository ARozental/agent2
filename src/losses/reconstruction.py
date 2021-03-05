import torch.nn.functional as F
import torch.nn as nn
import torch


class ReconstructionLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model  # Model is the current AgentLevel
        self.pad_token_id = 0

    def forward(self, inputs, mask):
        vector = self.model.encode(inputs, mask)  # This calls the encoder and the compressor
        decompressed = self.model.decompressor(vector)
        output = self.model.decoder(tgt=decompressed, memory=decompressed)
        emb_weight = torch.transpose(self.model.encoder.embedding.weight, 0, 1).unsqueeze(0)
        logits = torch.matmul(output, emb_weight)

        return F.cross_entropy(
            logits.transpose(1, 2),
            inputs,
            ignore_index=self.pad_token_id
        )
