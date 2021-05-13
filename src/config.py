import torch
import json
import sys


class Config:
    sequence_lengths = [16, 16, 6, 3, 4]  # [10,12,6,20,20]
    # vector_sizes = [8, 10, 12, 14, 16, 18]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    vector_sizes = [32, 64, 96, 96, 128, 156]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    num_heads = [4, 8, 2, 2, 2, 2]  # [2,3,4,5] #for transformers
    fnn_sizes = vector_sizes  # [8, 10, 12, 14, 16, 18]  # [2,3,4,5] #for fnn in transformers
    num_transformer_layers = [2, 2, 2, 2, 2, 2]  # [2,2,2,2]
    mlm_rate = 0.15  # 0.15 like BERT
    batch_size = 8  # How many books/articles/etc per batch.
    batch_sizes = [4096, 4096, 1024, 1000, 1000]  # How many nodes to process at a time at each level
    batch_sizes_dynamic = [[4096, 9216, 15360], [512, 1024, 3072]]  # Used for the TPU (TODO - Make these true dynamic)
    dynamic_batch_sizes = True  # Used for the TPU
    mini_batch_size = 256

    drop_rate = 0.0

    pad_token_id = 1  # hard coded; will break logic if changed!!!
    eos_token_id = 2  # hard coded; will break logic if changed!!!
    join_token_id = 3  # hard coded; will break logic if changed!!!
    join_texts = True

    max_coherence_noise = 0.8

    # PNDB - None is off; integer for number of questions
    use_pndb1 = None
    use_pndb2 = None

    # smoothing
    # max_typo_loss = 10.0
    grad_clip_value = 0.99
    optimizer = "Adam"
    lr = 0.0002
    momentum = 0.9

    skip_batches = None  # How many batches to skip (additional on top of the checkpoint)
    use_checkpoint = None  # Load saved model and dataset step from a checkpoint

    log_experiment = False  # Log into tensorboard?
    log_every = 100  # Log the reconstructed text every x epochs/batches
    save_every = None  # Save the model every x epochs/batches; None never saves
    model_folder = "test"  # Where inside of the "models" folder to save the model to
    exp_folder = None  # Folder name in "runs" to log into. None defaults to tensorboard default
    viz_file = None  # CSV file where to save viz results in

    # An easy way to remember the indices of each level
    levels = {
        'WORD': 0,
        'SENTENCE': 1,
        'PARAGRAPH': 2,
        'CHAPTER': 3,
        'BOOK': 4,
    }

    agent_level = levels['SENTENCE']  # most complex vector agent can create 2=paragraph

    # Run configuration below (keeping device here makes it easier to use throughout all of the code)
    use_cuda = True
    use_tpu = False
    gpu_num = 0
    device = None  # Will be set in setup()

    @staticmethod
    def setup_device():
        if Config.use_cuda and torch.cuda.is_available():
            Config.device = torch.device('cuda', Config.gpu_num)
        else:
            Config.device = torch.device('cpu')


def loss_object_to_main_loss(obj):
    loss = 0.0
    for level in obj.keys():
        loss += obj[level]['m'] * 1.0
        loss += obj[level]['md'] * 0.1
        loss += obj[level]['c'] * 10.0
        loss += obj[level]['r'] * 1.0
        loss += obj[level]['d'] * 0.2
        loss += obj[level]['e'] * 0.1
        loss += obj[level]['j'] * 0.1
        loss += obj[level]['rm'] * 0.3

        # loss += obj[level]['rc'] * 10.0
        # loss += obj[level]['re'] * 0.1
        # loss += obj[level]['rj'] * 0.1
        # loss += obj[level]['rmd']* 0.0 #off from code

        loss += obj[level]['cd'] * -0.03  # negative on the main weights
        loss += obj[level]['rcd'] * -0.03  # negative on the main weights

    return loss


def loss_object_to_reconstruction_weights_loss(obj):
    loss = 0.0
    for level in obj.keys():
        loss += obj[level]['rc'] * 10.0
        loss += obj[level]['re'] * 0.1
        loss += obj[level]['rj'] * 0.1
        # loss += obj[level]['rmd']* 0.0 #off from code
    return loss


def loss_object_to_extra_coherence_weights_loss(obj):
    loss = 0.0
    for level in obj.keys():
        loss += obj[level]['cd'] * 0.2
        loss += obj[level]['rcd'] * 0.2
    return loss
