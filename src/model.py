from src.losses import MLMLoss, CoherenceLoss, ReconstructionLoss
from src.agent import Level
import torch.nn as nn
import torch


class Model(nn.Module):
    def __init__(self, config, num_tokens, max_seq_length):
        super().__init__()

        self.levels = nn.ModuleList()
        self.losses = []
        for i, level_config in enumerate(config):
            if i < len(config) - 1:
                parent_embed = config[i + 1]['embed_size']
            else:
                parent_embed = level_config['embed_size'] * 2  # Figure out what to do for the last level
            if i == 0:
                level_config['is_base'] = True
            level = Level(num_tokens=num_tokens, max_seq_length=max_seq_length[i], parent_embed=parent_embed,
                          **level_config)

            self.levels.append(level)
            self.losses.append({
                'mlm': MLMLoss(level,
                               pad_token_id=0,
                               mask_token_id=1,
                               mask_prob=0.5,
                               random_token_prob=0.1,
                               num_tokens=81),
                'coherence': CoherenceLoss(level),
                'reconstruct': ReconstructionLoss(level),
            })

    def fit(self, inputs, mask, level=None):
        if level is None:
            level = len(inputs.size()) - 2

        print('level', level)
        if level > 0:
            print(inputs.reshape((4, 5)))
            vectors, loss_m, loss_c, loss_r = self.fit(inputs.reshape((4, 5)), mask.reshape((4, 5)), level=level - 1)
            print(vectors.size())
            inputs = vectors.reshape((2, 2, 160))
            # inputs[:, :] = vectors
            # print(vectors)
            # exit()
        else:
            loss_m = []
            loss_c = []
            loss_r = []

        mlm_loss = self.losses[level]['mlm'](inputs, mask)
        coherence_loss = self.losses[level]['coherence'](inputs, mask)
        reconstruct_loss = self.losses[level]['reconstruct'](inputs, mask)

        print('mlm_loss', mlm_loss.item())
        print('coherence_loss', coherence_loss.item())
        print('reconstruct_loss', reconstruct_loss.item())

        loss_m.append(mlm_loss)
        loss_c.append(coherence_loss)
        loss_r.append(reconstruct_loss)

        return self.levels[level].encode(inputs, mask), loss_m, loss_c, loss_r

    def forward(self, src, mask):
        raise NotImplementedError

    def decode(self, src, mask):
        raise NotImplementedError
