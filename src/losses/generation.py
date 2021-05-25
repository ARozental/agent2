import torch.nn as nn
import torch
from src.config import Config

BCE_LOSS = nn.BCEWithLogitsLoss()


def calc_generation_loss(agent_level, vectors, matrices, real_positions):
    batch, vec_size = vectors.shape
    fake_vecs = agent_level.generator.forward(vectors)  # make fake vecs of the same shape
    labels = torch.cat([torch.ones(batch, device=Config.device), torch.zeros(batch, device=Config.device)], dim=0)
    vecs = torch.cat([vectors, fake_vecs], dim=0)
    disc_loss = agent_level.discriminator.get_loss(vecs, labels)

    _, _, fake_matrices, fake_real_positions = agent_level.vecs_to_children_vecs(vectors)
    fake_real_positions = fake_real_positions.unsqueeze(-1)
    cnn_real_positions = real_positions.unsqueeze(-1)
    fake_matrices *= fake_real_positions  # 0 on all pad positions to make life easy for the CNN
    cnn_matrices = matrices * cnn_real_positions

    mats = torch.cat([cnn_matrices, fake_matrices], dim=0)
    cnn_disc_loss = agent_level.cnn_discriminator.get_loss(mats, labels)

    coherence_scores, coherence_probs = agent_level.coherence_checker(fake_vecs).squeeze()
    coherence_g_loss = (coherence_scores - torch.zeros(batch, device=Config.device)).norm() / (
                (Config.vector_sizes[agent_level.level + 1]) ** 0.5)
    # also get coherence for fake children??

    return coherence_g_loss, cnn_disc_loss + disc_loss
