from src.config import Config
import torch


def calc_coherence_loss(agent_level, matrices, mask, eos_positions, embeddings):
    # matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    batch, seq_length, vec_size = matrices.shape

    # 50% of examples don't change at all, move to config?
    changed_examples = torch.rand(batch, 1).to(Config.device).round()

    change_probs = torch.rand(batch, 1).to(Config.device) * Config.max_coherence_noise
    changed_tokens = torch.add(torch.rand(batch, seq_length).to(Config.device), change_probs).floor()
    non_mask = 1 - mask.int()

    # number of changed real tokens / num real tokens [batch]
    labels = (changed_tokens * changed_examples * non_mask).sum(-1) / non_mask.sum(-1)

    # todo: make sure the pad token is not here, also no join for levels 0 and 1 otherwise pad learns
    #random_indexes = torch.fmod(torch.randperm(batch * seq_length).to(Config.device), embeddings.shape[0])
    random_indexes = (torch.rand(batch * seq_length).to(Config.device) * embeddings.shape[0]).floor().int()
    random_vec_replacements = torch.index_select(embeddings, 0, random_indexes)
    random_vec_replacements = random_vec_replacements.view(batch, seq_length, vec_size)

    pre_encoder = (1 - changed_examples).unsqueeze(-1) * matrices
    pre_encoder += changed_examples.unsqueeze(-1) * changed_tokens.unsqueeze(-1) * random_vec_replacements
    pre_encoder += changed_examples.unsqueeze(-1) * (1-changed_tokens).unsqueeze(-1) * matrices


    vectors_for_coherence = agent_level.compressor(agent_level.encoder(pre_encoder, mask, eos_positions), mask)
    res = agent_level.coherence_checker(vectors_for_coherence).squeeze(-1)
    coherence_losses = (res - labels) ** 2

    return coherence_losses
