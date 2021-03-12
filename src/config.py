class Config():
    sequence_lengths = [7,9,6,3,4] #[10,12,6,20,20]
    vector_sizes = [8,10,12,14,16,18] #[4,6,8,10] #letters,words,sentences,paragraphs,chapters,book
    num_heads = [2,2,2,2,2,2]#[2,3,4,5] #for transformers
    fnn_sizes = [8,10,12,14,16,18] #[2,3,4,5] #for fnn in transformers
    vocab_sizes = [80,20,10,8,8,8]#[1000,21,6,5
    num_transformer_layers = [2,2,2,2,2,2]#[2,2,2,2]
    dtype = 'float32'
    mlm_rate = 0.15 #0.15 like BERT
    batch_size = 2 #todo: use it to create the actual dataset, it is also hardcoded there
    drop_rate = 0.15
    batch_size = batch_size

    pad_token_id = 1 #hard coded; will break logic if changed!!!
    eos_token_id = 2 #hard coded; will break logic if changed!!!
    join_token_id = 3 #hard coded; will break logic if changed!!!
    join_texts = False
    agent_level = 2 #most complex vector agent can create 2=paragraph



# MODEL_CONFIG = [
#     {
#         # Word Level
#         'embed_size': 80,
#         'encoder': {
#             'num_hidden': 80,
#             'num_layers': 2,
#             'num_head': 2,
#             'dropout': 0.00,
#         },
#         'decoder': {
#             'num_hidden': 80,
#             'num_layers': 2,
#             'num_head': 2,
#             'dropout': 0.00,
#         },
#         'mlm': {
#             'mask_prob': 0.5,
#             'random_token_prob': 0.1,
#         },
#     },
#     {
#         # Sentence Level
#         'embed_size': 160,
#         'encoder': {
#             'num_hidden': 160,
#             'num_layers': 2,
#             'num_head': 2,
#             'dropout': 0.00,
#         },
#         'decoder': {
#             'num_hidden': 160,
#             'num_layers': 2,
#             'num_head': 2,
#             'dropout': 0.00,
#         },
#         'mlm': {
#             'mask_prob': 0.5,
#             'random_token_prob': 0.1,
#         },
#     },
# ]
