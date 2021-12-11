from src.logger import Logger
from nltk.tag import pos_tag


def log_pndb(expected, activations, global_step=None):
    activations = activations.flatten()

    proper_indices = [i for i, (_, pos) in enumerate(pos_tag(expected.split())) if pos in ['NNP', 'NNPS','CD']]
    non_indices = list(set(range(activations.shape[0])) - set(proper_indices))

    Logger.log_pndb({
        'all': activations.mean(),
        'proper': activations[proper_indices].mean(),
        'non': activations[non_indices].mean(),
    }, step=global_step)
