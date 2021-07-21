#from src.utils import earth_movers_distance
import torch.nn as nn
import torch
#import torch.nn.functional as F
#from src.config import Config

# It makes all non EoS positions go and be the opposite of EoS => fixed by: dot = torch.max(dot, torch.zeros(dot.shape))
bce_loss = nn.BCEWithLogitsLoss(reduction='none')
mce_loss = nn.CrossEntropyLoss(reduction='none')


def decompressed_to_cdot(agent_level, decompressed):
    eos_vector = agent_level.eos_vector.unsqueeze(0).unsqueeze(0)
    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm())
    dot = dot.sum(dim=-1, keepdim=True)
    cdot = agent_level.eos_classifier1(dot).squeeze(-1)
    return cdot


def cdot_to_probs(cdot):
    return torch.stack([
        torch.sigmoid(cdot),
        torch.softmax(cdot, -1) * 0.9999  # *0.999 for exploding gradient on edge condition
    ]).min(0)[0]


def calc_eos_loss(agent_level, decompressed, eos_positions):
    cdot = decompressed_to_cdot(agent_level, decompressed)
    # Convert to float to avoid a case where they are int's and the loss breaks
    eos_mask = cdot_to_probs(cdot)

    #loss_mul = earth_sizes(eos_positions, eos_mask) #/ decompressed.shape[-1]
    sizes = torch.arange(eos_positions.size()[-1]).unsqueeze(0) - torch.argmax(eos_positions)
    sizes = sizes * torch.sign(sizes) / decompressed.shape[1] * 4
    cdot = cdot+sizes #we make cdot (the logits) worse here the further away from the label they are to force the network to give really low probabilities to everything too far from the label

    loss1 = bce_loss(cdot.float(), eos_positions.float()).mean(-1)  # needed because of texts with full size and no EoS
    eos_labels = torch.argmax(eos_positions, dim=1)

    # multiply losses where no eos exist by 0 otherwise by 1 because argmax for all zeroes is 0
    loss2 = mce_loss(cdot, eos_labels) * torch.clamp(torch.sum(eos_positions, dim=1), 0, 1) #clamp for 0 loss on 0 EoSs
    total_loss = loss1 + loss2 #+ loss3
    # total_loss = torch.min(torch.stack([(total_loss/total_loss)*Config.max_typo_loss,total_loss],dim=0),dim=0)[0] #can't explode on typo

    return total_loss, eos_mask


# def calc_eos_loss(agent_level, decompressed, eos_positions):
#   cdot = decompressed_to_cdot(agent_level, decompressed)
#   # Convert to float to avoid a case where they are int's and the loss breaks
#   loss1 = bce_loss(cdot.float(), eos_positions.float()).mean(-1)  # needed because of texts with full size and no EoS
#
#   eos_labels = torch.argmax(eos_positions, dim=1)
#
#   # multiply losses where no eos exist by 0 otherwise by 1 because argmax for all zeroes is 0
#   loss2 = mce_loss(cdot, eos_labels) * torch.clamp(torch.sum(eos_positions, dim=1), 0, 1)
#
#   probs = torch.softmax(cdot, -1)
#   eos_mask = cdot_to_probs(cdot)
#
#   loss3 = earth_movers_distance(eos_positions, eos_mask) / decompressed.shape[-1]
#
#   total_loss = loss1 + loss2 + loss3
#   #total_loss = torch.min(torch.stack([(total_loss/total_loss)*Config.max_typo_loss,total_loss],dim=0),dim=0)[0] #can't explode on typo
#
#   return total_loss, eos_mask

