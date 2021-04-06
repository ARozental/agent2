import torch.nn as nn
import torch

# It makes all non EoS positions go and be the opposite of EoS => fixed by: dot = torch.max(dot, torch.zeros(dot.shape))
bce_loss = nn.BCEWithLogitsLoss(reduction='none')
mce_loss = nn.CrossEntropyLoss(reduction='none')
dot_act = nn.ELU()


def calc_eos_loss(agent_level, decompressed, eos_positions):
    eos_vector = agent_level.eos_vector.unsqueeze(0).unsqueeze(0)
    eos_labels = torch.argmax(eos_positions, dim=1)

    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * eos_vector / eos_vector.norm()).sum(dim=-1,
                                                                                                       keepdim=True)

    cdot = agent_level.eos_classifier1(dot).squeeze(-1)
    loss1 = bce_loss(cdot, eos_positions).mean(-1)  # needed because of texts with full size and no EoS

    # multiply losses where no eos exist by 0 otherwise by 1 because argmax for all zeroes is 0
    loss2 = mce_loss(cdot, eos_labels) * torch.sign(torch.count_nonzero(eos_positions, dim=1))

    total_loss = loss1 + loss2

    # TODO: move this * 20 to hyper parameters for loss object, level 0 needs E and R but little M and no D
    if agent_level.level == 0:
        total_loss = total_loss

    return total_loss
