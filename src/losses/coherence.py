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




#
#
# class CoherenceLoss(nn.Module):
#     def __init__(self, model):
#         super().__init__()
#         self.model = model
#         self.pad_token_id = 0
#
#         self.loss = nn.MSELoss()
#         self.replace_prob = 0.5
#
#     # TODO - Don't replace the indices of the special token
#     # TODO - Clean this up
#     def forward(self, inputs, mask):
#         unique_values = inputs.unique()
#         unique_values = unique_values[unique_values != self.pad_token_id]
#
#         do_replace = (torch.rand((inputs.size(0), 1)) > self.replace_prob)
#
#         target_probs = torch.rand((inputs.size(0), 1)) * do_replace
#         prob_each = target_probs.repeat((1, inputs.size(1)))
#         replace_each = do_replace.repeat((1, inputs.size(1)))
#
#         token_prob = torch.rand((inputs.size(0), inputs.size(1)))
#
#         token_replace = token_prob > prob_each
#         token_replace = token_replace * replace_each
#
#         random_tokens = torch.randint(0, unique_values.size(0), size=inputs.size())
#
#         replaced_inputs = (inputs * ~token_replace) + (token_replace * random_tokens)
#
#         # The mask should handle the padding tokens (even if they were replaced)
#         encoded = self.model.encoder(replaced_inputs, mask)
#         vector = self.model.compressor(encoded)
#         preds = self.model.coherence_checker(vector)
#
#         # Ignore rows that are all padded
#         all_padded = (inputs != self.pad_token_id).sum(1)
#         ignore_padding = (all_padded > 0)
#         target_probs = target_probs[ignore_padding]
#         replace_each = replace_each[ignore_padding]
#         preds = preds[ignore_padding]
#
#         return self.loss(preds * replace_each, target_probs)
