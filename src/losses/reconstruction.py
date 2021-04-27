from src.config import Config
import torch.nn.functional as F
import torch
import math
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_rmlm_loss
from src.losses.coherence import calc_rc_loss

def calc_reconstruction_loss(agent_level, matrices, decompressed, mask, eos_positions,join_positions, embeddings, labels):
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

    reconstruction_losses = reconstruction_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
    #reconstruction_losses = torch.min(torch.stack([(reconstruction_losses/reconstruction_losses)*Config.max_typo_loss,reconstruction_losses],dim=0),dim=0)[0] #can't explode on typo
    reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.shape[0])))

    re_loss = calc_eos_loss(agent_level, post_decoder, eos_positions)

    reencoded_matrices = agent_level.encoder(post_decoder, mask, eos_positions)
    rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings,labels)
    rc_loss = calc_rc_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings)


    if Config.join_texts and agent_level.level >0:
      rj_loss = calc_join_loss(agent_level, post_decoder, join_positions)
    else:
      rj_loss = torch.tensor([0.0] * matrices.size(0)).to(Config.device)

    return reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss,rm_diff_loss


#todo have rc here later
def calc_reconstruction_loss_with_pndb(agent_level, matrices, decompressed, mask, eos_positions,join_positions, embeddings, labels,pndb,A1,A2):
  # matrices, mask, labels => [batch,seq_length,vec_size]
  if Config.use_pndb2:
    decompressed = pndb.get_data_from_A2_matrix(A2,decompressed)

  post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
  if Config.use_pndb1:
    post_decoder = pndb.get_data_from_A_matrix(A1,post_decoder)

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

  reconstruction_losses = reconstruction_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
  #reconstruction_losses = torch.min(torch.stack([(reconstruction_losses / reconstruction_losses) * Config.max_typo_loss, reconstruction_losses], dim=0),dim=0)[0]  # can't explode on typo
  reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.shape[0])))
  re_loss = calc_eos_loss(agent_level, post_decoder, eos_positions)

  reencoded_matrices = agent_level.encoder(post_decoder, mask, eos_positions)
  rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings, labels) #no mask keep the decoded vectors and predict originals by encoding
  rc_loss = calc_rc_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings)

  if Config.join_texts and agent_level.level > 0:
    rj_loss = calc_join_loss(agent_level, post_decoder, join_positions)
  else:
    rj_loss = torch.tensor([0.0] * matrices.size(0)).to(Config.device)
  reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss, rm_diff_loss = reconstruction_diff.to(Config.device), reconstruction_losses.to(Config.device), rc_loss.to(Config.device), re_loss.to(Config.device), rj_loss.to(Config.device), rm_loss.to(Config.device),rm_diff_loss.to(Config.device)

  return reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss,rm_diff_loss
