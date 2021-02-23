from src.agent import Level
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, config, num_tokens, max_seq_length):
        super().__init__()

        self.levels = nn.ModuleList()
        for i, level_config in enumerate(config):
            if i < len(config) - 1:
                parent_embed = config[i + 1]['embed_size']
            else:
                parent_embed = level_config['embed_size'] * 2  # Figure out what to do for the last level
            self.levels.append(Level(num_tokens=num_tokens, max_seq_length=max_seq_length, parent_embed=parent_embed,
                                     **level_config))

    def forward(self, src, mask):
        raise NotImplementedError

    def decode(self, src, mask):
        raise NotImplementedError
