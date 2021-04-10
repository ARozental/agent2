import torch


class Config:
    sequence_lengths = [7, 9, 6, 3, 4]  # [10,12,6,20,20]
    # vector_sizes = [8, 10, 12, 14, 16, 18]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    vector_sizes = [32, 48, 64, 96, 128, 156]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    num_heads = [2, 2, 2, 2, 2, 2]  # [2,3,4,5] #for transformers
    fnn_sizes = [8, 10, 12, 14, 16, 18]  # [2,3,4,5] #for fnn in transformers
    vocab_sizes = [80, 20, 10, 8, 8, 8]  # [1000,21,6,5
    num_transformer_layers = [2, 2, 2, 2, 2, 2]  # [2,2,2,2]
    dtype = 'float32'
    mlm_rate = 0.15  # 0.15 like BERT
    batch_size = 2  # todo: use it to create the actual dataset, it is also hardcoded there
    drop_rate = 0.0

    pad_token_id = 1  # hard coded; will break logic if changed!!!
    eos_token_id = 2  # hard coded; will break logic if changed!!!
    join_token_id = 3  # hard coded; will break logic if changed!!!
    join_texts = False

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
    USE_CUDA = True
    device = torch.device('cuda' if torch.cuda.is_available() and USE_CUDA else 'cpu')
