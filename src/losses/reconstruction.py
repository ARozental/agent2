import torch.nn.functional as F
import torch.nn as nn
import torch
from src.config import Config



bce_loss = nn.BCEWithLogitsLoss(reduction='none') #it makes all non EoS positions go and be the opposite of EoS
def calc_reconstruction_loss(agent_level, matrices, vectors, mask,eos_positions, embeddings, labels,epoch=7):
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

    # if agent_level.level ==2:
    #     print(eos_positions) => tensor([[0., 0., 1., 0., 0., 0.], [0., 1., 0., 0., 0., 0.]]) => ok


    #real_positions = (1-mask.float()-eos_positions).unsqueeze(-1)
    real_positions = (1-mask.float()).unsqueeze(-1)
    eos_vector = agent_level.eos_vector.unsqueeze(0).unsqueeze(0)

    decompressed = agent_level.decompressor(vectors)

    post_decoder = agent_level.decoder(decompressed, mask,eos_positions)  # [batch, seq_length, vec_size]

    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)

    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm()).sum(dim=-1,
                                                                                                       keepdim=True)
    dot = torch.max(dot, torch.zeros(dot.shape))  # no need for vectors to learn to become anti eos
    eos_losses = bce_loss(agent_level.eos_classifier1(dot).squeeze(-1), eos_positions).mean(-1)
    reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2])) / ((matrices * real_positions).norm(dim=[1,2]))  # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
    if agent_level.level==0:
        eos_losses = 20 * eos_losses
        logits = logits+agent_level.token_bias

    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch, words]
    ).mean(-1)

    return reconstruction_diff,eos_losses,reconstruction_losses
