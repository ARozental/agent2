from src.config import Config
from src.utils import inverse_loss, cap_loss


def loss_object_to_main_loss(obj):
    loss = 0.0
    for l in obj.keys():
        loss += obj[l]['m'] * 0.1
        # loss += obj[l]['md'] * 0.1 #off from code
        loss += obj[l]['c'] * 2.0
        loss += obj[l]['r'] * 0.1
        loss += obj[l]['e'] * 0.1
        loss += obj[l]['j'] * 0.001  # do we even need it??
        loss += obj[l]['d'] * Config.main_d  # moved here as a test

        loss += obj[l]['rc'] * 0.4
        loss += obj[l]['re'] * 0.1
        loss += obj[l]['rj'] * 0.01
        #loss += obj[l]['rmd'] * Config.main_rmd

        # if l > 0:
        #     loss += - obj[l]['rcd'] * Config.main_rcd  # negative on the main weights
        loss += obj[l]['rm'] * Config.main_rm

    return loss


def loss_object_to_reconstruction_weights_loss(obj):
    loss =  0.0
    for l in obj.keys():
        loss += obj[l]['rm'] * (-Config.main_rm)
        loss += obj[l]['rmd'] * (-Config.main_rmd)
    return loss


def loss_object_to_extra_coherence_weights_loss(obj):
    loss = obj[0]['r'] * 0.0
    for l in obj.keys():
        loss += cap_loss(obj[l]['rcd']) * 1
    return loss
