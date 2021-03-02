from src.losses import MLMLoss, CoherenceLoss, ReconstructionLoss
from src.agent import AgentLevel
from typing import Iterator
from torch.nn import Parameter
import numpy as np
import torch.nn as nn
import torch


class AgentModel(nn.Module):
    def __init__(self, config, num_tokens, max_seq_length):
        super().__init__()

        self.levels = nn.ModuleList()
        self.max_seq_length = max_seq_length
        self.losses = []
        for i, level_config in enumerate(config):
            if i < len(config) - 1:
                parent_embed = config[i + 1]['embed_size']
            else:
                parent_embed = level_config['embed_size'] * 2  # Figure out what to do for the last level
            mlm_config = level_config.pop('mlm', {})
            level = AgentLevel(level_num=i, num_tokens=num_tokens, max_seq_length=max_seq_length[i],
                               parent_embed=parent_embed, **level_config)

            self.levels.append(level)
            self.losses.append({
                'mlm': MLMLoss(level, pad_token_id=0, mask_token_id=1, **mlm_config, num_tokens=num_tokens),
                'coherence': CoherenceLoss(level),
                'reconstruct': ReconstructionLoss(level),
            })

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for name, param in self.named_parameters(recurse=recurse):
            if 'embedding' in name and not name.startswith('levels.0.'):  # Ignore embedding above base level
                continue

            yield param

    def convert_vectors_to_indices(self, vectors, level):
        vectors = vectors.detach().numpy()
        unique_vectors = np.unique(vectors, axis=0)

        self.levels[level].set_embedding(torch.tensor(unique_vectors))

        inputs = np.array([np.argwhere((vec == unique_vectors).all(1))[0][0] for vec in vectors])
        inputs += 4  # Make room for the pad, mask, etc tokens

        self.losses[level]['mlm'].num_tokens = len(unique_vectors) + 4

        return inputs

    def create_inputs_mask(self, inputs, level):
        # Add EoS Token
        # TODO - Reference the EoS token straight from the tokenizer so that it will be dynamic
        inputs = [seq + [2] for seq in inputs]

        mask = [[0] * len(seq) + [1] * (self.max_seq_length[level] - len(seq)) for seq in inputs]
        inputs = [seq + [0] * (self.max_seq_length[level] - len(seq)) for seq in inputs]
        inputs = torch.tensor(inputs)
        mask = torch.tensor(mask)

        return inputs, mask

    def fit(self, inputs, level=None):
        if level is None:
            current = inputs[0]
            level = -1
            while isinstance(current, list):
                current = current[0]
                level += 1

        if level > 0:
            lengths = [len(seq) for seq in inputs]
            inputs = [item for seq in inputs for item in seq]
            vectors, loss_m, loss_c, loss_r = self.fit(inputs, level=level - 1)

            inputs = self.convert_vectors_to_indices(vectors, level)
            inputs = np.split(inputs, lengths)

            # TODO - Identify why this happens (there is an empty sequence in there for some reason)
            inputs = [seq.tolist() for seq in inputs if len(seq) > 0]
        else:
            loss_m = []
            loss_c = []
            loss_r = []

        inputs, mask = self.create_inputs_mask(inputs, level)

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

        vectors = self.levels[level].encode(inputs, mask)

        return vectors, loss_m, loss_c, loss_r

    def forward(self, src, mask):
        raise NotImplementedError

    def encode(self, inputs, level=None):
        if level is None:
            current = inputs[0]
            level = -1
            while isinstance(current, list):
                current = current[0]
                level += 1

        if level > 0:
            lengths = [len(seq) for seq in inputs]
            inputs = [item for seq in inputs for item in seq]
            vectors = self.encode(inputs, level=level - 1)

            inputs = self.convert_vectors_to_indices(vectors, level)
            inputs = np.split(inputs, lengths)

            # TODO - Identify why this happens (there is an empty sequence in there for some reason)
            inputs = [seq.tolist() for seq in inputs if len(seq) > 0]

        inputs, mask = self.create_inputs_mask(inputs, level)
        return self.levels[level].encode(inputs, mask)

    # return_word_vectors is temporary now while debugging
    def debug_decode(self, vectors, level=None, return_word_vectors=False):
        if level is None:
            level = len(vectors.size()) - 2

        if level == 0:
            # Don't reshape when doing word level eval
            # TODO - Make this dynamic in the future and not so rigid
            need_reshape = len(vectors.size()) > 2
            if need_reshape:
                original_shape = vectors.size()
                shape = (vectors.size(0) * vectors.size(1), vectors.size(2))
                vectors = vectors.reshape(shape)
        decoded = self.levels[level].debug_decode(vectors)
        if level == 0:
            if need_reshape:
                decoded = decoded.reshape((original_shape[0], original_shape[1], decoded.size(1)))
            return decoded

        if level == 1 and return_word_vectors:
            return decoded

        return self.debug_decode(decoded, level=level - 1)
