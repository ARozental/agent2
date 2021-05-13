import torch.nn as nn
import torch

# It makes all non EoS positions go and be the opposite of EoS => fixed by: dot = torch.max(dot, torch.zeros(dot.shape))
bce_loss = nn.BCEWithLogitsLoss(reduction='none')
mce_loss = nn.CrossEntropyLoss(reduction='none')


def calc_eos_loss(agent_level, decompressed, eos_positions):
    eos_vector = agent_level.eos_vector.unsqueeze(0).unsqueeze(0)
    eos_labels = torch.argmax(eos_positions, dim=1)

    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm())
    dot = dot.sum(dim=-1, keepdim=True)

    cdot = agent_level.eos_classifier1(dot).squeeze(-1)
    # Convert to float to avoid a case where they are int's and the loss breaks
    loss1 = bce_loss(cdot.float(), eos_positions.float()).mean(-1)  # needed because of texts with full size and no EoS

    # multiply losses where no eos exist by 0 otherwise by 1 because argmax for all zeroes is 0
    loss2 = mce_loss(cdot, eos_labels) * torch.clamp(torch.sum(eos_positions, dim=1), 0, 1)

    total_loss = loss1 + loss2
    # total_loss = torch.min(torch.stack([(total_loss/total_loss)*Config.max_typo_loss,total_loss],dim=0),dim=0)[0] #can't explode on typo

    return total_loss
