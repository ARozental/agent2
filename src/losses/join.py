import torch.nn as nn
import torch
from src.config import Config

bce_loss = nn.BCEWithLogitsLoss(reduction='none')


def calc_join_loss(agent_level, decompressed, join_positions):
    join_vector = agent_level.join_vector.unsqueeze(0).unsqueeze(0)

    dot = (decompressed / decompressed.norm(dim=2, keepdim=True) * join_vector / join_vector.norm())
    dot = dot.sum(dim=-1, keepdim=True)
    cdot = agent_level.join_classifier(dot).squeeze(-1)

    # Convert to float to avoid a case where they are int's and the loss breaks
    loss = bce_loss(cdot.float(), join_positions.float()).mean(-1)  # needed because of texts with full size and no EoS
    loss = torch.min(torch.stack([(loss/loss)*Config.max_typo_loss,loss],dim=0),dim=0)[0] #can't explode on typo


    return loss
