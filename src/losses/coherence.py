from src.config import Config
import torch
import torch.nn as nn
bce_loss = nn.BCELoss(reduction='none')
ce_loss = nn.CrossEntropyLoss(reduction='none')


def make_fake_normal_vectors(vectors):
  batch,hidden = vectors.shape
  m = vectors.var(0,keepdims=True)
  v = vectors.mean(0,keepdims=True)
  fake = torch.rand(batch, hidden).to(Config.device)
  fake = (fake*v)+m
  return fake



def calc_coherence_loss(agent_level, matrices, mask, eos_positions, embeddings):
    # matrices,mask,labels => [batch,seq_length,vec_size], embeddings => [seq_length,vec_size]
    batch, seq_length, vec_size = matrices.shape

    # 50% of examples don't change at all, move to config?
    changed_examples = torch.rand(batch, 1).to(Config.device).round()

    change_probs = torch.rand(batch, 1).to(Config.device) * Config.max_coherence_noise
    changed_tokens = torch.add(torch.rand(batch, seq_length).to(Config.device), change_probs).floor()
    non_mask = 1 - mask.int()

    # number of changed real tokens / num real tokens [batch]
    labels = (changed_tokens * changed_examples * non_mask).sum(-1) / non_mask.sum(-1) #~40% are changed

    # todo: make sure the pad token is not here, also no join for levels 0 and 1 otherwise pad learns
    #random_indexes = torch.fmod(torch.randperm(batch * seq_length).to(Config.device), embeddings.shape[0])
    random_indexes = (torch.rand(batch * seq_length).to(Config.device) * embeddings.shape[0]).floor().long()
    random_vec_replacements = torch.index_select(embeddings, 0, random_indexes)
    random_vec_replacements = random_vec_replacements.view(batch, seq_length, vec_size)

    pre_encoder = (1 - changed_examples).unsqueeze(-1) * matrices
    pre_encoder += changed_examples.unsqueeze(-1) * changed_tokens.unsqueeze(-1) * random_vec_replacements
    pre_encoder += changed_examples.unsqueeze(-1) * (1-changed_tokens).unsqueeze(-1) * matrices


    vectors_for_coherence = agent_level.compressor(agent_level.encoder(pre_encoder, mask, eos_positions), mask)
    scores, probs, class_predictions = agent_level.coherence_checker(vectors_for_coherence)
    coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1),labels.ceil()) * 0.05)

    # fake stuff
    total_cd_loss = torch.tensor(0.0).to(Config.device)
    # fake_vectors = make_fake_normal_vectors(vectors_for_coherence)
    # _, _, noise_class_predictions = agent_level.coherence_checker(fake_vectors)
    # predictions = torch.cat([noise_class_predictions, class_predictions])
    # predictions_labels = torch.cat([torch.zeros(batch, dtype=torch.long).to(Config.device),
    #                                 torch.ones(batch, dtype=torch.long).to(Config.device) ])
    # total_cd_loss = ce_loss(predictions, predictions_labels).sum()  # rcd is reconstruction coherence discrimination

    return coherence_losses,total_cd_loss

def calc_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder):
  batch, seq_length, vec_size = reencoded_matrices.shape
  labels = torch.zeros(batch).to(Config.device)
  vectors_for_coherence = agent_level.compressor(reencoded_matrices, mask)
  scores, probs,class_predictions = agent_level.coherence_checker(vectors_for_coherence)
  coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)


  #fake stuff
  total_rcd_loss = torch.tensor(0.0).to(Config.device)
  # fake_vectors = make_fake_normal_vectors(vectors_for_coherence)
  # _, _, noise_class_predictions = agent_level.coherence_checker(fake_vectors)
  # predictions = torch.cat([noise_class_predictions, class_predictions ])
  # predictions_labels = torch.cat([torch.zeros(batch,dtype=torch.long).to(Config.device), torch.ones(batch,dtype=torch.long).to(Config.device) * 2])
  # total_rcd_loss = ce_loss(predictions,predictions_labels).sum() #rcd is reconstruction coherence discrimination


  return coherence_losses,total_rcd_loss

# def calc_lower_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder):
#   batch, seq_length, vec_size = post_decoder.shape
#   vectors_for_coherence = post_decoder.view(-1,vec_size)
#   labels = torch.zeros(batch*seq_length).to(Config.device)
#   scores,probs,class_predictions = lower_agent_level.coherence_checker(vectors_for_coherence)
#   coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)
#   return coherence_losses,class_predictions


def calc_lower_rc_loss(agent_level, reencoded_matrices, mask, lower_agent_level, post_decoder):
  batch, seq_length, vec_size = post_decoder.shape
  non_mask = (1 - mask.int()).view(-1)
  vectors_for_coherence = post_decoder.view(-1,vec_size) #todo fix bug we also take the would be masked vecotrs here
  labels = torch.zeros(batch*seq_length).to(Config.device)
  scores,probs,class_predictions = lower_agent_level.coherence_checker(vectors_for_coherence)
  coherence_losses = (scores.squeeze(-1) - labels) ** 2 + (bce_loss(probs.squeeze(-1), labels.ceil()) * 0.05)
  coherence_losses = coherence_losses*non_mask

  #fake stuff
  total_rcd_loss=torch.tensor(0.0).to(Config.device)
  # fake_vectors = make_fake_normal_vectors(vectors_for_coherence)
  # _, _, noise_class_predictions = lower_agent_level.coherence_checker(fake_vectors)
  # predictions = torch.cat([noise_class_predictions, class_predictions ])
  # predictions_labels = torch.cat([torch.zeros(batch*seq_length,dtype=torch.long).to(Config.device), torch.ones(batch*seq_length,dtype=torch.long).to(Config.device) * 2])
  # total_rcd_loss = ce_loss(predictions,predictions_labels)#.sum() #rcd is reconstruction coherence discrimination
  # total_rcd_loss = total_rcd_loss * torch.cat([non_mask,non_mask])
  # total_rcd_loss = total_rcd_loss.sum() / seq_length
  return coherence_losses,total_rcd_loss