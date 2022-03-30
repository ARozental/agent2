from src.config import Config
import torch.nn.functional as F
import torch
import math
from src.losses.eos import calc_eos_loss
from src.losses.join import calc_join_loss
from src.losses.mlm import calc_rmlm_loss
from src.losses.coherence import calc_rc_loss, calc_lower_rc_loss, make_fake_normal_vectors


def calc_reconstruction_loss(agent_level, matrices, vectors, reencoded_matrices, real_positions, eos_positions, join_positions,
                             embeddings,
                             labels, pndb,A1s, pndb_lookup_ids, num_dummy=0, dummy_logit_bias=None):
    # matrices, mask, labels => [batch,seq_length,vec_size]


    #todo: decompressed-last_compressed here?
    decompressed = agent_level.decompressor(vectors)
    regularization_diff = decompressed-reencoded_matrices
    regularization_diff = (regularization_diff * real_positions.unsqueeze(-1)).norm(dim=[2]).mean(dim=1)
    regularization_diff = 2*regularization_diff / (((decompressed * real_positions.unsqueeze(-1)).norm(dim=[2]) +
                                                    (reencoded_matrices * real_positions.unsqueeze(-1)).norm(dim=[2])).mean(dim=1))




    # if Config.use_pndb2 is not None and agent_level.level == 1:
    #     decompressed = pndb.old_get_data_from_A_matrix(pndb.create_A_matrix(matrices, real_positions), decompressed)

    # overrides real_positions with the best the decompressor can do
    eos_loss, projected_eos_positions = calc_eos_loss(agent_level, decompressed, eos_positions)
    real_positions_for_mask = (1 - torch.cumsum(projected_eos_positions, dim=1))

    # [batch,seq_length,vec_size]
    post_decoder = agent_level.decoder(decompressed, real_positions_for_mask, eos_positions)

    if Config.use_pndb1 is not None and agent_level.level == 1:
        #post_decoder = pndb.old_get_data_from_A_matrix(pndb.create_A_matrix(matrices, real_positions), post_decoder)
        post_decoder, gate_values = pndb.get_data_from_A_matrix(A1s, pndb_lookup_ids, post_decoder,real_positions_for_mask)
        real_sum = real_positions.sum(dim=-1)
        gate_values = gate_values.squeeze(-1) * real_positions
        e_x = gate_values.sum(dim=-1) / real_sum #E(x)
        v_x = ((gate_values-e_x.unsqueeze(-1))**2).sum(dim=-1) / real_sum
        memory_loss = e_x - e_x*(v_x**0.5) #better to make a hard decision here so minus var #Popoviciu's inequality => var<= 1/4
    else:
        memory_loss = torch.zeros(post_decoder.size(0), device=post_decoder.device)


    logits = torch.matmul(post_decoder, torch.transpose(embeddings, 0, 1))  # [batch, max_length, embedding_size)

    # works :) with *10?, maybe we won't need the *10 when there is a real dataset, verify the norm doesn't go crazy because of this line later

    # norm(p=1) is not supported on the TPU
    #reconstruction_diff = ((matrices - post_decoder) * real_positions.unsqueeze(-1)).norm(dim=[1, 2])
    reconstruction_diff = matrices - post_decoder
    reconstruction_diff = reconstruction_diff #* (eos_positions.unsqueeze(-1)*2+1) #bug because eos is doubly counted in numerator but not denominator
    reconstruction_diff = (reconstruction_diff * real_positions.unsqueeze(-1)).norm(dim=[2]).mean(dim=1)
    reconstruction_diff = 2*reconstruction_diff / (((matrices * real_positions.unsqueeze(-1)).norm(dim=[2]) +
                                                    (post_decoder * real_positions.unsqueeze(-1)).norm(dim=[2])).mean(dim=1))

    if agent_level.level == 0:
        logits = logits + agent_level.token_bias

    if Config.use_tpu and dummy_logit_bias is not None:
        logits = logits - dummy_logit_bias

    reconstruction_losses = F.cross_entropy(
        logits.transpose(1, 2),
        labels,
        ignore_index=Config.pad_token_id,
        reduction='none'  # Gives mlm loss from each of [batch, words]
    )  # .mean(-1) => this was a bug
    reconstruction_losses = reconstruction_losses * (eos_positions*2+1)
    reconstruction_losses = reconstruction_losses.sum(-1) / real_positions.sum(-1)

    # 4.4 is ln(len(char_embedding)) == ln(81)
    reconstruction_losses = reconstruction_losses * (4.4 / math.log(embeddings.size(0) - num_dummy))
    # reconstruction_losses = torch.min(torch.stack([(reconstruction_losses / reconstruction_losses) * Config.max_typo_loss, reconstruction_losses], dim=0),dim=0)[0]  # can't explode on typo
    reconstruction_diff = (reconstruction_diff * (4.4 / math.log(embeddings.size(0) - num_dummy)))
    #todo: add same size and same angle??? maybe only cosign similarity?

    re_loss, _ = calc_eos_loss(agent_level, post_decoder, eos_positions)

    _=None #do these 2 lines save on memory? reencoded_matrices moved inside calc_rmlm_loss
    #reencoded_matrices = agent_level.encoder(post_decoder, real_positions_for_mask, eos_positions)

    #rm_loss, rm_diff_loss = calc_rmlm_loss(agent_level, post_decoder, real_positions_for_mask, eos_positions, real_positions, matrices, embeddings,labels)  # no mask keep the decoded vectors and predict originals by encoding


    if agent_level.level > 0:
        rc_loss= calc_lower_rc_loss(real_positions,
                                               agent_level.previous_level,
                                               post_decoder)
    else:
        rc_loss = eos_loss * 0

    #no rc/rcd loss
    #rcd_loss = torch.zeros(batch * 2, device=Config.device)
    #rc_loss = torch.zeros(batch * seq_length, device=Config.device)


    # if Config.join_texts and agent_level.level > 0:
    #     rj_loss = calc_join_loss(agent_level, post_decoder, join_positions)
    # else:
    #     rj_loss = torch.zeros(post_decoder.size(0), device=post_decoder.device)
    rj_loss = regularization_diff+memory_loss
    return reconstruction_diff, reconstruction_losses,eos_loss, re_loss, rj_loss,rc_loss
