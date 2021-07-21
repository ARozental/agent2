from src.config import Config
import torch.nn.functional as F
import torch
import math


def calc_mlm_loss(agent_level, matrices, real_positions, eos_positions, embeddings, labels, num_dummy=0,
                  dummy_logit_bias=None):
    # matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    # to use when calc_clear_mlm_loss is not active, has a "same word" component

    # todo: have the mask one version be more efficient => just choose one vector

    batch, seq_length, vec_size = matrices.shape

    # Choose 1 to mask MLM
    mlm_indices = torch.max(torch.rand((batch, seq_length), device=matrices.device) * real_positions, dim=-1).indices
    mlm_positions = torch.zeros((batch, seq_length), device=matrices.device)
    mlm_positions[torch.arange(batch, device=matrices.device), mlm_indices] = 1
    mlm_positions = mlm_positions.unsqueeze(-1)
    keep_positions = 1 - mlm_positions

    # Prob MLM;   1 => keep original 0, calc mlm,Config.mlm_rate
    # keep_positions = (torch.rand(batch, seq_length, 1, device=Config.device) + Config.mlm_rate).floor()
    # mlm_positions = 1 - keep_positions

    # 1 => replace with <mask>
    mask_positions = (torch.rand(batch, seq_length, 1, device=matrices.device) + 0.9).floor() * mlm_positions

    # 1 => replace with original, 0 replace with random   #valid when there is no calc_clear_mlm_loss active
    random_replace_positions = mlm_positions * (1 - mask_positions)

    mask_vec_replacements = agent_level.mask_vector.repeat(batch * seq_length).view(batch, seq_length, vec_size)

    # todo: make sure the pad token is not here, also no join for levels 0 and 1
    # random_indexes = torch.fmod(torch.randperm(batch * seq_length).to(Config.device), embeddings.shape[0])
    num_indices = (embeddings.size(0) - num_dummy)  # Number of real indices to use
    random_indexes = (torch.rand(batch * seq_length, device=matrices.device) * num_indices).floor().long()
    random_vec_replacements = torch.index_select(embeddings, 0, random_indexes).view(batch, seq_length, vec_size)

    pre_encoder = keep_positions * matrices + mask_positions * mask_vec_replacements
    pre_encoder += random_replace_positions * random_vec_replacements

    post_encoder = agent_level.encoder(pre_encoder, real_positions, eos_positions)
    transformed = agent_level.encoder_transform(post_encoder)
    logits = torch.matmul(transformed, torch.transpose(embeddings, 0, 1))  # [batch,max_length,embedding_size)

    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    if Config.use_tpu and dummy_logit_bias is not None:
        logits = logits - dummy_logit_bias

    mlm_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch,words]
    )
    # mlm_losses = mlm_losses.mean(-1) => this is a bug, as we count average all positions for the loss (even not selected ones
    mlm_losses = (mlm_losses * real_positions * mlm_positions.squeeze(-1)).sum(
        -1)  # this fix is only valid in the choose 1 MLM position version otherwise replace "sum" with mean_on_non_zeros

    # 4.4 is ln(len(char_embedding)) == ln(81)
    mlm_losses = mlm_losses * (4.4 / math.log(embeddings.size(0) - num_dummy))
    # mlm_losses = torch.min(torch.stack([(mlm_losses/mlm_losses)*Config.max_typo_loss,mlm_losses],dim=0),dim=0)[0] #can't explode on typo


    #rmlm/keep_all_mlm
    reencoded_matrices = agent_level.encoder(matrices, real_positions, eos_positions)
    transformed = agent_level.encoder_transform(reencoded_matrices)
    logits = torch.matmul(transformed, torch.transpose(embeddings, 0, 1))
    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    rmlm_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch,words]
    )
    rmlm_losses = rmlm_losses.sum(-1) / real_positions.sum(-1)
    rmlm_losses = rmlm_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)



    # mlm_diff
    # real_positions = real_positions.unsqueeze(-1)
    # mlm_diff = (((matrices - transformed) * real_positions).norm(dim=[1, 2]))
    # mlm_diff = mlm_diff / ((matrices * real_positions).norm(dim=[1, 2]))
    # mlm_diff = (mlm_diff * (4.4 / math.log(embeddings.shape[0]))) #/ 100
    #mlm_diff = torch.zeros(batch, device=Config.device)
    #rmlm_diff = mlm_diff

    return mlm_losses, rmlm_losses


def calc_rmlm_loss(agent_level, post_decoder, real_positions_for_mask, eos_positions, real_positions, matrices, embeddings, labels):
    reencoded_matrices = agent_level.encoder(post_decoder, real_positions_for_mask, eos_positions)
    batch, seq_length, vec_size = reencoded_matrices.shape

    transformed = agent_level.encoder_transform(reencoded_matrices)
    logits = torch.matmul(transformed, torch.transpose(embeddings, 0, 1))  # [batch,max_length,embedding_size)

    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    mlm_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch,words]
    )  # .mean(-1)
    mlm_losses = mlm_losses.sum(-1) / real_positions.sum(-1)

    # todo?? have mlm_diff here?
    mlm_losses = mlm_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
    # mlm_losses = torch.min(torch.stack([(mlm_losses/mlm_losses)*Config.max_typo_loss,mlm_losses],dim=0),dim=0)[0] #can't explode on typo

    # mlm_diff
    # real_positions = real_positions.unsqueeze(-1)
    # mlm_diff = (((matrices - transformed) * real_positions).norm(dim=[1, 2]))
    # mlm_diff = mlm_diff / ((matrices * real_positions).norm(dim=[1, 2]))

    # no mmlm_diff
    mlm_diff = torch.zeros(batch, device=reencoded_matrices.device)

    return mlm_losses, mlm_diff
