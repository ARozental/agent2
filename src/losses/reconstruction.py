from src.config import Config
import torch.nn.functional as F
import torch


def calc_reconstruction_loss(agent_level, matrices, decompressed, mask, eos_positions, embeddings, labels):
    # matrices, mask, labels => [batch,seq_length,vec_size]

    post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)

    real_positions = (1 - mask.float()).unsqueeze(-1)
    # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
    reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
    reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))

    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch, words]
    ).mean(-1)
    return reconstruction_diff, reconstruction_losses


def calc_reconstruction_loss_with_pndb(agent_level, matrices, decompressed, mask, eos_positions, embeddings, labels,pndb,A1,A2):
  # matrices, mask, labels => [batch,seq_length,vec_size]
  if Config.use_pndb2:
    decompressed = pndb.get_data_from_A_matrix(A2,decompressed, pndb_type=2)

  post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
  if Config.use_pndb1:
    post_decoder = pndb.get_data_from_A_matrix(A1,post_decoder, pndb_type=1)

  logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)

  real_positions = (1 - mask.float()).unsqueeze(-1)
  # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
  reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
  reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))

  if agent_level.level == 0:
    logits = logits + agent_level.token_bias

  reconstruction_losses = F.cross_entropy(
    logits.transpose(1, 2),
    labels,
    ignore_index=Config.pad_token_id,
    reduction='none'  # Gives mlm loss from each of [batch, words]
  ).mean(-1)

  reconstruction_losses = reconstruction_losses * (4.4 / embeddings.shape[0])  # 4.4 is ln(len(char_embedding)) == ln(81)
  reconstruction_diff = reconstruction_diff / 100

  return reconstruction_diff, reconstruction_losses
