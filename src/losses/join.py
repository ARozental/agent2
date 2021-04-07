import torch.nn as nn

bce_loss = nn.BCEWithLogitsLoss(reduction='none')


def calc_join_loss(agent_level, decompressed, join_positions):
    join_vector = agent_level.join_vector.unsqueeze(0).unsqueeze(0)

    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * join_vector / join_vector.norm())
    dot = dot.sum(dim=-1, keepdim=True)
    cdot = agent_level.join_classifier(dot).squeeze(-1)

    loss = bce_loss(cdot, join_positions).mean(-1)  # needed because of texts with full size and no EoS

    return loss
