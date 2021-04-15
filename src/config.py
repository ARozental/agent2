import torch
import json
import sys


class Config:
    sequence_lengths = [16, 16, 6, 3, 4]  # [10,12,6,20,20]
    # vector_sizes = [8, 10, 12, 14, 16, 18]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    vector_sizes = [32, 48, 96, 96, 128, 156]  # [4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    num_heads = [2, 6, 2, 2, 2, 2]  # [2,3,4,5] #for transformers
    fnn_sizes = vector_sizes #[8, 10, 12, 14, 16, 18]  # [2,3,4,5] #for fnn in transformers
    num_transformer_layers = [2, 2, 2, 2, 2, 2]  # [2,2,2,2]
    mlm_rate = 0.15  # 0.15 like BERT
    batch_size = 2  # How many books/articles/etc per batch.
    batch_sizes = [3000, 3000, 1000, 1000, 1000]  # How many nodes to process at a time at each level
    drop_rate = 0.0

    pad_token_id = 1  # hard coded; will break logic if changed!!!
    eos_token_id = 2  # hard coded; will break logic if changed!!!
    join_token_id = 3  # hard coded; will break logic if changed!!!
    join_texts = True

    # pndb
    use_pndb1 = True
    use_pndb2 = False
    pndb_questions = 16

    #smoothing
    #max_eos_loss = 7.0 #doesn't fix anything on its own
    grad_clip_value = 1.0

    model_folder = "test"  # Where inside of the "models" folder to place this current run

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


def load(filename):
    with open('configs/' + filename + '.json') as f:
        data = json.load(f)

    for key, value in data.items():
        setattr(Config, key, value)


if len(sys.argv) > 1:
    load(sys.argv[1])
