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
                               num_tokens=num_tokens),
                'coherence': CoherenceLoss(level),
                'reconstruct': ReconstructionLoss(level),
            })

    def fit(self, inputs, mask, level=None):
        if level is None:
            level = len(inputs.size()) - 2

        if level > 0:
            original_shape = inputs.size()
            shape = (inputs.size(0) * inputs.size(1), inputs.size(2))
            vectors, loss_m, loss_c, loss_r = self.fit(inputs.reshape(shape), mask.reshape(shape), level=level - 1)

            # TODO - Add EoS token at the end of each

            # For now just replace vectors to be indices and treat normally
            import numpy as np
            vectors = vectors.detach().numpy()
            unique_vectors = np.unique(vectors, axis=0)

            self.levels[level].set_embedding(torch.tensor(unique_vectors))

            inputs = np.array([np.argwhere((vec == unique_vectors).all(1))[0][0] for vec in vectors])
            inputs = inputs.reshape((original_shape[0], original_shape[1]))
            inputs += 4  # Make room for the pad, mask, etc tokens

            inputs = torch.tensor(inputs)
            mask = mask.all(-1)

            self.losses[level]['mlm'].num_tokens = len(unique_vectors) + 4
        else:
            loss_m = []
            loss_c = []
            loss_r = []

        print('Level', level)
        mlm_loss = self.losses[level]['mlm'](inputs, mask)
        coherence_loss = self.losses[level]['coherence'](inputs, mask)
        reconstruct_loss = self.losses[level]['reconstruct'](inputs, mask)

        print('mlm_loss', mlm_loss.item())
        print('coherence_loss', coherence_loss.item())
        print('reconstruct_loss', reconstruct_loss.item())

        loss_m.append(mlm_loss)
        loss_c.append(coherence_loss)
        loss_r.append(reconstruct_loss)

        self.levels[level].eval()
        with torch.no_grad():
            vectors = self.levels[level].encode(inputs, mask)
        self.levels[level].train()

        return vectors, loss_m, loss_c, loss_r

    def forward(self, src, mask):
        raise NotImplementedError

    def decode(self, inputs, mask, level=None):
        if level is None:
            level = len(inputs.size()) - 2

        if level > 0:
            original_shape = inputs.size()
            shape = (inputs.size(0) * inputs.size(1), inputs.size(2))
            result = self.decode(inputs.reshape(shape), mask.reshape(shape), level=level - 1)
            result = result.reshape(original_shape)
            return result

        return self.levels[level].decode(inputs, mask)
