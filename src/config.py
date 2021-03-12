# TODO - Change this to a class like the old Config was
MODEL_CONFIG = [
    {
        # Word Level
        'embed_size': 80,
        'max_seq_length': 10,
        'encoder': {
            'num_hidden': 80,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.00,
        },
        'decoder': {
            'num_hidden': 80,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.00,
        },
        'mlm': {
            'mask_prob': 0.5,
            'random_token_prob': 0.1,
        },
    },
    {
        # Sentence Level
        'embed_size': 160,
        'max_seq_length': 6,
        'encoder': {
            'num_hidden': 160,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.00,
        },
        'decoder': {
            'num_hidden': 160,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.00,
        },
        'mlm': {
            'mask_prob': 0.5,
            'random_token_prob': 0.1,
        },
    },
]
