from src.config import Config
from src.logger import Logger
from src.pre_processing import TreeTokenizer
import torch


def reconstruct_text(batch_tree, model, embedding_matrix, first_A1s, first_pndb_lookup_ids, global_step=None, exit_on_match=False):
    nodes = batch_tree.batch_root.children
    expected = [TreeTokenizer.deep_detokenize(node.build_struct(return_eos=True)[0], Config.agent_level)
                for node in nodes]

    with torch.no_grad():
        #todo: make more efficient, we don't need 2 runs here
        reconstructed = [model.full_decode(batch_tree.level_nodes[i][:10], first_A1s, first_pndb_lookup_ids[0:10],embedding_matrix) for i in
                     range(Config.agent_level + 1)]
        reconstructed_e = [model.full_decode(batch_tree.level_nodes[i][:10], first_A1s, first_pndb_lookup_ids[0:10],embedding_matrix,from_embedding = True) for i in
                     range(Config.agent_level + 1)]

    reconstructed = [[TreeTokenizer.deep_detokenize(node[0], i) for node in items] for i, items in
                     enumerate(reconstructed)]
    reconstructed_e = [[TreeTokenizer.deep_detokenize(node[0], i) for node in items] for i, items in
                     enumerate(reconstructed_e)]
    for i, text in enumerate(reconstructed):
        print('Level', i, reconstructed[i])
        print('Level_e', i, reconstructed_e[i])
        Logger.log_reconstructed(reconstructed[i], i, step=global_step)
        Logger.log_reconstructed_e(reconstructed_e[i], i, step=global_step)
        # for j, item in enumerate(text):
        #    Logger.log_viz(batch.level_nodes[i][j], text[j], i, step=global_step)
        if i == len(reconstructed) - 1:  # Upper most level
            are_equal = [t == e for t, e in zip(text, expected)]
            if False not in are_equal:
                print('MATCHED')

                if exit_on_match:
                    exit()
