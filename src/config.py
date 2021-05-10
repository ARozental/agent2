import torch
import json
import sys
from src.utils import inverse_loss, cap_loss

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
    lr = 0.0005
    momentum = 0.9
    half_life_steps = 150000
    grad_acc_steps = 2

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
    gpu_num = 0
    device = None  # Will be set in setup()

    @staticmethod
    def setup_device():
        if Config.use_cuda and torch.cuda.is_available():
            Config.device = torch.device('cuda', Config.gpu_num)
        else:
            Config.device = torch.device('cpu')


    cnn_padding = 2 # kernal=2*padding+1
    reconstruction_d = 0.0
    main_d= 0.03
    main_rcd = 0.03
    main_rm = 0.1
    main_rmd = 0.03



def loss_object_to_main_loss(obj):
  loss = obj[0]['r'] * 0.0
  for l in obj.keys():
    loss += obj[l]['m']  * 0.1
    #loss += obj[l]['md'] * 0.1 #off from code
    loss += obj[l]['c']  * 1.0
    loss += obj[l]['r']  * 0.1
    loss += obj[l]['e']  * 0.1
    loss += obj[l]['j']  * 0.001   #do we even need it??
    loss += obj[l]['d']  * Config.main_d #moved here as a test



    loss += obj[l]['rc'] * 1.0
    loss += obj[l]['re'] * 0.1
    loss += obj[l]['rj'] * 0.01
    loss += obj[l]['rmd']* Config.main_rmd

    if l>0:
      loss += inverse_loss(obj[l]['rcd']) * Config.main_rcd   #negative on the main weights
    loss += obj[l]['rm'] * Config.main_rm



  return loss

def loss_object_to_reconstruction_weights_loss(obj):
  loss = obj[0]['r'] * 0.0
  for l in obj.keys():
    loss += obj[l]['rm'] * (-Config.main_rm)
    loss += obj[l]['rmd'] * (-Config.main_rmd)


  return loss

def loss_object_to_extra_coherence_weights_loss(obj):
  loss = obj[0]['r'] * 0.0
  for l in obj.keys():
    loss += cap_loss(obj[l]['rcd']) * 1
  return loss