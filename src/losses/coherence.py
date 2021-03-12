import torch.nn as nn
import torch



def calc_coherence_loss(agent_level,matrices,mask,embeddings):
    #matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    batch, seq_length, vec_size = matrices.shape
    changed_examples = torch.rand(batch,1).round() #50% of examples don't change at all, move to config?
    change_probs = torch.rand(batch,1)
    changed_tokens = torch.add(torch.rand(batch,seq_length),change_probs).floor()
    non_mask = 1 - mask.int()
    labels = (changed_tokens*changed_examples*non_mask).sum(-1) / non_mask.sum(-1) #number of changed real tokens / num real tokens [batch]

    random_indexes = torch.fmod(torch.randperm(batch * seq_length), embeddings.shape[0])
    random_vec_replacments = torch.index_select(embeddings, 0,random_indexes).view(batch, seq_length, vec_size) #todo: make sure the pad token is not here, also no join for levels 0 and 1

    pre_encoder = (1-changed_examples).unsqueeze(-1) * matrices + changed_examples.unsqueeze(-1) * random_vec_replacments
    vectors_for_coherence = agent_level.compressor(agent_level.encoder(pre_encoder, mask))
    res = agent_level.coherence_checker(vectors_for_coherence).squeeze(-1)
    coherence_losses = (res-labels)**2

    return coherence_losses

