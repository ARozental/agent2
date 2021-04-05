import torch.nn.functional as F
import torch.nn as nn
import torch
from src.config import Config

# it makes all non EoS positions go and be the opposite of EoS => fixed by: dot = torch.max(dot, torch.zeros(dot.shape))
bce_loss = nn.BCEWithLogitsLoss(reduction='none')
mce_loss = nn.CrossEntropyLoss(reduction='none')
dot_act = nn.ELU()


def calc_reconstruction_loss(agent_level, matrices, vectors, mask, eos_positions, embeddings, labels):
    """

    Parameters
    ----------
    agent_level
    vectors
    mask
    embeddings: [seq_length, vec_size]
    labels

    Returns
    -------

    """
    # matrices, mask, labels => [batch,seq_length,vec_size]

    real_positions = (1 - mask.float()).unsqueeze(-1)

    eos_vector = agent_level.eos_vector.unsqueeze(0).unsqueeze(0)
    decompressed = agent_level.decompressor(vectors)
    post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)
    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm()).sum(dim=-1,
                                                                                                       keepdim=True)

    eos_labels = torch.argmax(eos_positions, dim=1)

    # dot = torch.max(dot, torch.zeros(dot.shape))  # no need for vectors to learn to become anti eos, max kills learning if init is bad elu in classifier1 works
    cdot = agent_level.eos_classifier1(dot).squeeze(-1)
    eos_losses1 = bce_loss(cdot, eos_positions).mean(-1)  # needed because of texts with full size and no EoS

    # multiply losses where no eos exist by 0 otherwise by 1 because argmax for all zeroes is 0
    eos_losses2 = mce_loss(cdot, eos_labels) * torch.sign(torch.count_nonzero(eos_positions, dim=1))

    eos_losses = eos_losses1 + eos_losses2

    # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
    reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
    reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))

    if agent_level.level == 0:
        eos_losses = eos_losses  # todo: move this * 20 to hyper parameters for loss object, level 0 needs E and R but little M and no D
        logits = logits + agent_level.token_bias

    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch, words]
    ).mean(-1)

    return reconstruction_diff, eos_losses, reconstruction_losses
