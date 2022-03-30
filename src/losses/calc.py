from src.config import Config
from src.utils import inverse_loss, cap_loss


def loss_object_to_main_loss(loss_object):
    loss = 0.0
    for i, level in loss_object.items():
        for name, value in level.items():
            if name!= 'rc':
                if i in Config.loss_weights and name in Config.loss_weights[i]:
                    loss += value * Config.loss_weights[i][name]
                elif name in Config.loss_weights:
                    loss += value * Config.loss_weights[name]
                else:
                    raise ValueError(f'A loss weight for "{name}" needs to be defined in Config.loss_weights')

    return loss


def loss_object_to_reconstruction_weights_loss(obj):
    loss = 0.0
    for l in obj.keys():
        loss += obj[l]['rc'] * (Config.loss_weights[l]['rc'])
        # loss += obj[l]['rm'] * (-Config.main_rm)
        # loss += obj[l]['rmd'] * (-Config.main_rmd)
    return loss


def loss_object_to_extra_coherence_weights_loss(obj):
    loss = obj[0]['r'] * 0.0
    for l in obj.keys():
        loss += cap_loss(obj[l]['rcd']) * 1
    return loss
