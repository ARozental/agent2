import torch.nn as nn
import torch
from src.config import Config


bce_loss = nn.BCEWithLogitsLoss()
def calc_generation_loss(agent_level,vectors,matrices,mask):
    batch, vec_size = vectors.shape
    fake_vecs = agent_level.generator.forward(vectors)  # make fake vecs of the same shape
    labels = torch.cat([torch.ones(batch), torch.zeros(batch)], dim=0)
    vecs = torch.cat([vectors, fake_vecs], dim=0)
    disc_loss = agent_level.discriminator.get_loss(vecs, labels)

    _, fake_matrices, fake_mask = agent_level.vecs_to_children_vecs2(vectors)
    fake_mask = (1 - fake_mask.float()).unsqueeze(-1)
    cnn_mask = (1 - mask.float()).unsqueeze(-1)
    fake_matrices*=fake_mask #0 on all pad positions to make life easy for the CNN
    cnn_matrices = matrices * cnn_mask

    mats = torch.cat([cnn_matrices, fake_matrices], dim=0)
    cnn_disc_loss =agent_level.cnn_discriminator.get_loss(mats,labels)


    coherence = agent_level.coherence_checker(fake_vecs).squeeze()
    coherence_g_loss = (coherence - torch.zeros(batch)).norm() / ((Config.vector_sizes[agent_level.level + 1]) ** 0.5)
    # also get coherence for fake children??

    return coherence_g_loss, disc_loss
