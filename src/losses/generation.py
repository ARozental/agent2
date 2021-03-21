import torch.nn as nn
import torch
from src.config import Config


bce_loss = nn.BCEWithLogitsLoss()
def calc_generation_loss(agent_level,vectors,matrices):
    batch, vec_size = vectors.shape
    fake_vecs = agent_level.generator.forward(vectors)  # make fake vecs of the same shape
    labels = torch.cat([torch.ones(batch), torch.zeros(batch)], dim=0)
    vecs = torch.cat([vectors, fake_vecs], dim=0)
    disc_loss = agent_level.discriminator.get_loss(vecs, labels)


    # fake_decompressed = agent_level.decompressor(vectors)
    # fake_matrices = agent_level.decoder(fake_decompressed, fake_mask,fake_eos_positions)  #todo: needs mask and EoS positions for fake to work
    # mats = torch.cat([matrices, fake_matrices], dim=0)
    # agent_level.cnn_discriminator.get_loss(mats,fake_vecs)


    coherence = agent_level.coherence_checker(fake_vecs).squeeze()
    coherence_g_loss = (coherence - torch.zeros(batch)).norm() / ((Config.vector_sizes[agent_level.level + 1]) ** 0.5)
    # also get coherence for fake children??

    return coherence_g_loss, disc_loss
