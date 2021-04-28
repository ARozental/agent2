from src.config import Config
import torch.nn.functional as F
import torch
import math
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_rmlm_loss
from src.losses.coherence import calc_rc_loss, calc_lower_rc_loss

def make_reconstruction_loss_fn(level):
  def do_pndb1(pndb,A1,post_decoder):
    return pndb.get_data_from_A_matrix(A1, post_decoder)
  def no_pndb1(pndb,A1,post_decoder):
    return post_decoder
  def do_pndb2(pndb,A2,decompressed):
    return pndb.get_data_from_A2_matrix(A2, decompressed)
  def no_pndb2(pndb,A2,decompressed):
    return decompressed

  def make_A1(pndb,matrices, mask):
    return pndb.create_A_matrix(matrices, mask),None
  def make_A2(pndb,matrices, mask):
    return None,pndb.create_A2_matrix(matrices, mask)
  def no_A(pndb,matrices, mask):
    return None,None
  def both_A(pndb,matrices, mask):
    return pndb.create_A_matrix(matrices, mask),pndb.create_A2_matrix(matrices, mask)

  def create_As_fn():
    if level != 1:
      return no_A
    elif (Config.use_pndb1 is not None) and (Config.use_pndb2 is not None):
      return both_A
    elif Config.use_pndb1 is not None:
      return make_A1
    elif Config.use_pndb2 is not None:
      return make_A2
    else:
      return no_A

  As_fn = create_As_fn()

  def create_pndb1_fn():
    if level == 1 and (Config.use_pndb1 is not None):
      return do_pndb1
    else:
      return no_pndb1

  def create_pndb2_fn():
    if level == 1 and (Config.use_pndb2 is not None):
      return do_pndb2
    else:
      return no_pndb2
  pndb1_fn = create_pndb1_fn()
  pndb2_fn = create_pndb2_fn()

  def do_token_bias(agent_level,logits):
    return logits + agent_level.token_bias
  def no_token_bias(agent_level,logits):
    return logits

  def create_token_bias_fn():
    if level == 0:
      return do_token_bias
    else:
      return no_token_bias

  token_bias_fn = create_token_bias_fn()

  def do_lower_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder):
    return calc_lower_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder)
  def no_lower_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder):
    return calc_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder)

  def create_rc_loss_fn():
    if level > 0:
      return do_lower_rc_loss
    else:
      return calc_rc_loss

  rc_loss_fn = create_rc_loss_fn()


  def do_join_loss(agent_level, post_decoder, join_positions):
    return calc_join_loss(agent_level, post_decoder, join_positions)
  def no_join_loss(agent_level, post_decoder, join_positions):
    return torch.tensor([0.0] * post_decoder.size(0)).to(Config.device)

  def create_join_loss_fn():
    if Config.join_texts and level > 0:
      return do_join_loss
    else:
      return no_join_loss

  join_loss_fn = create_join_loss_fn()


  def calc_reconstruction_loss_fn(agent_level, matrices, decompressed, mask, eos_positions, join_positions,
                                         embeddings, labels, pndb):
    A1,A2 = As_fn(pndb,matrices, mask)
    # matrices, mask, labels => [batch,seq_length,vec_size]
    decompressed = pndb2_fn(pndb,A2, decompressed)


    post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
    post_decoder = pndb1_fn(pndb,A1, post_decoder)

    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)

    real_positions = (1 - mask.float()).unsqueeze(-1)
    # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
    reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
    reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))

    logits = token_bias_fn(agent_level,logits)

    reconstruction_losses = F.cross_entropy(
      logits.transpose(1, 2),
      labels,
      ignore_index=Config.pad_token_id,
      reduction='none'  # Gives mlm loss from each of [batch, words]
    ).mean(-1)

    reconstruction_losses = reconstruction_losses * (
    4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
    # reconstruction_losses = torch.min(torch.stack([(reconstruction_losses / reconstruction_losses) * Config.max_typo_loss, reconstruction_losses], dim=0),dim=0)[0]  # can't explode on typo
    reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.shape[0])))
    re_loss = calc_eos_loss(agent_level, post_decoder, eos_positions)

    reencoded_matrices = agent_level.encoder(post_decoder, mask, eos_positions)
    rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings,
                                           labels)  # no mask keep the decoded vectors and predict originals by encoding

    rc_loss = rc_loss_fn(agent_level, reencoded_matrices, mask, agent_level.previous_level, post_decoder)

    rj_loss = join_loss_fn(agent_level, post_decoder, join_positions)

    return reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss, rm_diff_loss
  return calc_reconstruction_loss_fn

#
# def calc_reconstruction_loss(agent_level, matrices, decompressed, mask, eos_positions,join_positions, embeddings, labels):
#     # matrices, mask, labels => [batch,seq_length,vec_size]
#
#     post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
#     logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)
#
#     real_positions = (1 - mask.float()).unsqueeze(-1)
#     # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
#     reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
#     reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))
#
#     if agent_level.level == 0:
#         logits = logits + agent_level.token_bias
#
#     reconstruction_losses = F.cross_entropy(
#         logits.transpose(1, 2),
#         labels,
#         ignore_index=Config.pad_token_id,
#         reduction='none'  # Gives mlm loss from each of [batch, words]
#     ).mean(-1)
#
#     reconstruction_losses = reconstruction_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
#     #reconstruction_losses = torch.min(torch.stack([(reconstruction_losses/reconstruction_losses)*Config.max_typo_loss,reconstruction_losses],dim=0),dim=0)[0] #can't explode on typo
#     reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.shape[0])))
#
#     re_loss = calc_eos_loss(agent_level, post_decoder, eos_positions)
#
#     reencoded_matrices = agent_level.encoder(post_decoder, mask, eos_positions)
#     rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings,labels)
#     rc_loss = calc_rc_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings)
#
#
#     if Config.join_texts and agent_level.level >0:
#       rj_loss = calc_join_loss(agent_level, post_decoder, join_positions)
#     else:
#       rj_loss = torch.tensor([0.0] * matrices.size(0)).to(Config.device)
#
#     #todo: figure out wtf it is needed here
#     reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss, rm_diff_loss = reconstruction_diff.to(Config.device), reconstruction_losses.to(Config.device), rc_loss.to(Config.device), re_loss.to(Config.device), rj_loss.to(Config.device), rm_loss.to(Config.device), rm_diff_loss.to(Config.device)
#
#     return reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss,rm_diff_loss
# #todo have rc here later
# def calc_reconstruction_loss_with_pndb(agent_level, matrices, decompressed, mask, eos_positions,join_positions, embeddings, labels,pndb,A1,A2):
#   # matrices, mask, labels => [batch,seq_length,vec_size]
#   if Config.use_pndb2:
#     decompressed = pndb.get_data_from_A2_matrix(A2,decompressed)
#
#   post_decoder = agent_level.decoder(decompressed, mask, eos_positions)  # [batch, seq_length, vec_size]
#   if Config.use_pndb1:
#     post_decoder = pndb.get_data_from_A_matrix(A1,post_decoder)
#
#   logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)
#
#   real_positions = (1 - mask.float()).unsqueeze(-1)
#   # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later
#   reconstruction_diff = (((matrices - post_decoder) * real_positions).norm(dim=[1, 2]))
#   reconstruction_diff = reconstruction_diff / ((matrices * real_positions).norm(dim=[1, 2]))
#
#   if agent_level.level == 0:
#     logits = logits + agent_level.token_bias
#
#   reconstruction_losses = F.cross_entropy(
#     logits.transpose(1, 2),
#     labels,
#     ignore_index=Config.pad_token_id,
#     reduction='none'  # Gives mlm loss from each of [batch, words]
#   ).mean(-1)
#
#   reconstruction_losses = reconstruction_losses * (4.4 / math.log(embeddings.shape[0]))  # 4.4 is ln(len(char_embedding)) == ln(81)
#   #reconstruction_losses = torch.min(torch.stack([(reconstruction_losses / reconstruction_losses) * Config.max_typo_loss, reconstruction_losses], dim=0),dim=0)[0]  # can't explode on typo
#   reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.shape[0])))
#   re_loss = calc_eos_loss(agent_level, post_decoder, eos_positions)
#
#   reencoded_matrices = agent_level.encoder(post_decoder, mask, eos_positions)
#   rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings, labels) #no mask keep the decoded vectors and predict originals by encoding
#   rc_loss = calc_rc_loss(agent_level, reencoded_matrices, mask, eos_positions, embeddings)
#
#   if Config.join_texts and agent_level.level > 0:
#     rj_loss = calc_join_loss(agent_level, post_decoder, join_positions)
#   else:
#     rj_loss = torch.tensor([0.0] * matrices.size(0)).to(Config.device)
#
#   # todo: figure out wtf it is needed here
#   reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss, rm_diff_loss = reconstruction_diff.to(Config.device), reconstruction_losses.to(Config.device), rc_loss.to(Config.device), re_loss.to(Config.device), rj_loss.to(Config.device), rm_loss.to(Config.device),rm_diff_loss.to(Config.device)
#
#   return reconstruction_diff, reconstruction_losses, rc_loss, re_loss, rj_loss, rm_loss,rm_diff_loss
