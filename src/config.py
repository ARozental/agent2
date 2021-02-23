# TODO - Change this to a class like the old Config was
MODEL_CONFIG = [
    {
        # Word Level
        'embed_size': 80,
        'encoder': {
            'num_hidden': 80,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.01,
        },
        'decoder': {
            'num_hidden': 80,
            'num_layers': 2,
            'num_head': 2,
            'dropout': 0.01,
        },
    },
    # {
    #     # Sentence Level
    #     'embed_size': 160,
    #     'encoder': {
    #         'num_hidden': 80,
    #         'num_layers': 2,
    #         'num_head': 2,
    #         'dropout': 0.01,
    #     },
    #     'decoder': {
    #         'num_hidden': 80,
    #         'num_layers': 2,
    #         'num_head': 2,
    #         'dropout': 0.01,
    #     },
    # }
]
