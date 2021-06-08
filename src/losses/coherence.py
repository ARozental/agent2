from src.config import Config
import torch
import torch.nn as nn

bce_loss = nn.BCELoss(reduction='none')
ce_loss = nn.CrossEntropyLoss(reduction='none')


def make_fake_normal_vectors(vectors):
    batch, hidden = vectors.shape
    m = vectors.var(0, keepdims=True)
    v = vectors.mean(0, keepdims=True)
    fake = torch.rand(batch, hidden, device=Config.device)
    fake = (fake * v) + m
    return fake


def calc_coherence_loss(agent_level, matrices, real_positions, eos_positions, embeddings, num_dummy=0):
    # matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    batch, seq_length, vec_size = matrices.shape

    # 50% of examples don't change at all, move to config?
    #changed_examples = torch.rand(batch, 1, device=Config.device).round()
    #changed_examples = torch.zeros(batch, 1, device=Config.device)
    changed_examples = agent_level.zeros1d[:batch]

    #change_probs = torch.rand(batch, 1, device=Config.device) * Config.max_coherence_noise
    #changed_tokens = torch.add(torch.rand(batch, seq_length, device=Config.device), change_probs).floor()

    #changed_tokens = torch.zeros(batch, seq_length, device=Config.device) #todo: delete no rand
    changed_tokens = agent_level.zeros2d[:batch]

    # number of changed real tokens / num real tokens [batch]
    labels = (changed_tokens * changed_examples * real_positions).sum(-1) / real_positions.sum(-1)  # ~40% are changed

    # todo: make sure the pad token is not here, also no join for levels 0 and 1 otherwise pad learns
    # random_indexes = torch.fmod(torch.randperm(batch * seq_length).to(Config.device), embeddings.shape[0])
    #num_indices = (embeddings.size(0) - num_dummy)  # Number of real indices to use
    #random_indexes = (torch.rand(batch * seq_length, device=Config.device) * num_indices).floor().long()
    # random_indexes = (torch.zeros(batch * seq_length, device=Config.device) * num_indices).floor().long() #todo: delete no rand
    # random_vec_replacements = torch.index_select(embeddings, 0, random_indexes)
    # random_vec_replacements = random_vec_replacements.view(batch, seq_length, vec_size)

    # pre_encoder = (1 - changed_examples).unsqueeze(-1) * matrices
    # pre_encoder += changed_examples.unsqueeze(-1) * changed_tokens.unsqueeze(-1) * random_vec_replacements
    # pre_encoder += changed_examples.unsqueeze(-1) * (1 - changed_tokens).unsqueeze(-1) * matrices

    vectors_for_coherence = agent_level.compressor(agent_level.encoder(matrices, real_positions, eos_positions),
                                                   real_positions)
    scores, probs, class_predictions = agent_level.coherence_checker(vectors_for_coherence)
    coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)

    # # fake stuff
    # fake_vectors = make_fake_normal_vectors(vectors_for_coherence)
    # _, _, noise_class_predictions = agent_level.coherence_checker(fake_vectors)
    # predictions = torch.cat([noise_class_predictions, class_predictions])
    # predictions_labels = torch.cat([torch.zeros(batch, dtype=torch.long, device=Config.device),
    #                                 torch.ones(batch, dtype=torch.long, device=Config.device) ])
    # total_cd_loss = ce_loss(predictions, predictions_labels).sum()  # rcd is reconstruction coherence discrimination

    total_cd_loss = torch.tensor(0.0, device=Config.device)

    return coherence_losses, total_cd_loss


def calc_rc_loss(agent_level, reencoded_matrices, real_positions, lower_agent_level, post_decoder, matrices):
    batch, seq_length, vec_size = post_decoder.shape
    # labels = torch.zeros(batch, device=Config.device)
    # vectors_for_coherence = agent_level.compressor(reencoded_matrices, real_positions)
    # scores, probs,class_predictions = agent_level.coherence_checker(vectors_for_coherence)
    # coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)

    rcd_loss = torch.zeros(batch * 2, device=Config.device)
    coherence_losses = torch.zeros(batch * seq_length, device=Config.device)
    return coherence_losses, rcd_loss


def calc_lower_rc_loss(agent_level, reencoded_matrices, real_positions, lower_agent_level, matrices, post_decoder):
    batch, seq_length, vec_size = post_decoder.shape
    vectors_for_coherence = post_decoder.view(-1,
                                              vec_size)  # todo fix bug we also take the would be masked vecotrs here
    labels = torch.ones(batch * seq_length, device=Config.device)
    scores, probs, class_predictions = lower_agent_level.coherence_checker.lower_forward(vectors_for_coherence,
                                                                                         torch.cat([
                                                                                                       matrices * real_positions.unsqueeze(
                                                                                                           -1),
                                                                                                       post_decoder * real_positions.unsqueeze(
                                                                                                           -1)]))
    coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)
    real_positions = real_positions.view(-1)
    coherence_losses = coherence_losses * real_positions

    predictions_labels = torch.cat([torch.zeros(batch, device=Config.device), torch.ones(batch, device=Config.device)])

    rcd_loss = bce_loss(class_predictions.squeeze(-1), predictions_labels)

    return coherence_losses, rcd_loss
