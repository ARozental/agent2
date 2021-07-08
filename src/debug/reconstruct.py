from src.config import Config
from src.logger import Logger
from src.pre_processing import TreeTokenizer
import torch


def reconstruct_text(batch_tree, model, embedding_matrix, first_A1s, first_pndb_lookup_ids, global_step=None, exit_on_match=False):
    "we should have an embedding matrix for each level for debugging here, currently it is just for the word level"
    nodes = batch_tree.batch_root.children
    expected = [TreeTokenizer.deep_detokenize(node.build_struct(return_eos=True)[0], Config.agent_level)
                for node in nodes]

    with torch.no_grad():
        reconstructed = [model.full_decode(batch_tree.level_nodes[i][:5], first_A1s, first_pndb_lookup_ids[0:5],embedding_matrix) for i in
                     range(Config.agent_level + 1)]

    reconstructed = [[TreeTokenizer.deep_detokenize(node[0], i) for node in items] for i, items in
                     enumerate(reconstructed)]
    for i, text in enumerate(reconstructed):
        print('Level', i, text)
        Logger.log_reconstructed(text, i, step=global_step)
        # for j, item in enumerate(text):
        #    Logger.log_viz(batch.level_nodes[i][j], text[j], i, step=global_step)
        if i == len(reconstructed) - 1:  # Upper most level
            are_equal = [t == e for t, e in zip(text, expected)]
            if False not in are_equal:
                print('MATCHED')

                if exit_on_match:
                    exit()
