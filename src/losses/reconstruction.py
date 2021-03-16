import torch.nn.functional as F
import torch.nn as nn
import torch
from src.config import Config


def calc_reconstruction_loss(agent_level, vectors, mask,eos_positions, embeddings, labels):
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
    # matrices,mask,labels => [batch,seq_length,vec_size]

    batch, seq_length = mask.shape
    decompressed = agent_level.decompressor(vectors)
    # print("d",decompressed.shape,mask.shape)
    post_decoder = agent_level.decoder(decompressed, mask,eos_positions)  # [batch, seq_length, vec_size]

    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)
    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch, words]
    ).mean(-1)

    return reconstruction_losses
