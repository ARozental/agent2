import torch
import torch.nn.functional as F
from src.config import Config


def calc_mlm_loss(agent_level, matrices, mask, eos_positions, embeddings, labels):
    # matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    batch, seq_length, vec_size = matrices.shape

    # 1 => keep original 0, calc mlm,Config.mlm_rate
    keep_positions = (torch.rand(batch, seq_length, 1).to(Config.device) + Config.mlm_rate).floor()
    mlm_positions = 1 - keep_positions

    # 1 => replace with <mask>
    mask_positions = (torch.rand(batch, seq_length, 1).to(Config.device) + 0.8).floor() * mlm_positions

    # 1 => replace with original, 0 replace with random
    special_mlm_positions = torch.rand(batch, seq_length, 1).to(Config.device)
    random_replace_positions = mlm_positions * (1 - mask_positions) * (1 - special_mlm_positions)
    replace_with_original_positions = mlm_positions * (1 - mask_positions) * special_mlm_positions

    mask_vec_replacements = agent_level.mask_vector.repeat(batch * seq_length).view(batch, seq_length, vec_size)

    # todo: make sure the pad token is not here, also no join for levels 0 and 1
    random_indexes = torch.fmod(torch.randperm(batch * seq_length).to(Config.device), embeddings.shape[0])
    random_vec_replacements = torch.index_select(embeddings, 0, random_indexes).view(batch, seq_length, vec_size)

    pre_encoder = keep_positions * matrices + mask_positions * mask_vec_replacements
    pre_encoder += random_replace_positions * random_vec_replacements + replace_with_original_positions * matrices

    post_encoder = agent_level.encoder(pre_encoder, mask, eos_positions)
    transformed = agent_level.encoder_transform(post_encoder)
    logits = torch.matmul(transformed, torch.transpose(embeddings, 0, 1))  # [batch,max_length,embedding_size)

    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    mlm_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch,words]
    ).mean(-1)

    return mlm_losses
