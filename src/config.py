import torch
import json
import sys


class Config:
    levels = {'WORD': 0,'SENTENCE': 1,'PARAGRAPH': 2,'CHAPTER': 3,'BOOK': 4,}
    agent_level = levels['SENTENCE']  # most complex vector agent can create 2=paragraph

    sequence_lengths = [16, 16, 6, 3, 4]  # [10,12,6,20,20]
    # vector_sizes = [8, 10, 12, 14, 16, 18]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    vector_sizes = [32, 64, 96, 96, 128, 156]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    num_heads = [4, 8, 2, 2, 2, 2]  # [2,3,4,5] #for transformers
    fnn_sizes = vector_sizes  # [8, 10, 12, 14, 16, 18]  # [2,3,4,5] #for fnn in transformers
    num_transformer_layers = [2, 2, 2, 2, 2, 2]  # [2,2,2,2]
    mlm_rate = 0.15  # 0.15 like BERT
    batch_size = 128  # How many books/articles/etc per batch.
    node_sizes = [4096, 512, 4096, 1000, 1000]  # How many nodes to process at a time at each level => todo: change, each here limits the other
    node_sizes_max = [8192, 1024]  # Used for the TPU; only used when "dynamic_node_sizes" is True
    dynamic_node_sizes = False  # Used for the TPU to make it do 25%/50%/75%
    mini_batch_size = 1024 #max number of max_agent_level document, not working as intender but has an effect: final number is ~1.7 times higers, can be higher than batch size like when we get wiki articles as input but only doing up to level 1 (sentneces). should be at least as high as corresponding node size

    drop_rate = 0.0
    noise = 0.1
    max_word_embedding_size = 12000

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
    grad_acc_steps = 1

    skip_batches = None  # How many batches to skip (additional on top of the checkpoint)
    use_checkpoint = None  # Load saved model and dataset step from a checkpoint

    storage_location = None  # Where the root storage is (put "gs://" into here for Google Cloud)
    log_experiment = False  # Log into tensorboard?
    log_every = 100  # Log the reconstructed text every x epochs/batches
    save_every = None  # Save the model every x epochs/batches; None never saves
    model_folder = "test"  # Where inside of the "models" folder to save the model to
    exp_folder = None  # Folder name in "runs" to log into. None defaults to tensorboard default
    viz_file = None  # CSV file where to save viz results in

    force_resume = True


    # Run configuration below (keeping device here makes it easier to use throughout all of the code)
    use_cuda = True
    use_tpu = False
    use_all_tpu_cores = False
    debug_tpu = False
    profile_tpu = False
    gpu_num = 0
    device = None  # Will be set in setup()

    use_dummy_dataset = False

    @staticmethod
    def setup_device():
        if Config.use_tpu:
            return  # Don't do anything because needs to be done inline to support all cores
        elif Config.use_cuda and torch.cuda.is_available():
            Config.device = torch.device('cuda', Config.gpu_num)
        else:
            Config.device = torch.device('cpu')

    @staticmethod
    def grad_acc_fn(step):
        return 1
        # if step<300000:
        #     return 1
        # elif step<500000:
        #     return 2
        # elif step < 700000:
        #   return 4
        # else:
        #   return 8

    cnn_padding = 2  # kernal=2*padding+1
    reconstruction_d = 0.0
    main_d = 0.03
    main_rcd = 0.01
    main_rm = 0.1
    main_rmd = 0.03
