from src.checkpoints import Checkpoints
from src.commands import Commands
from src.debug.reconstruct import reconstruct_text
from src.pre_processing import TreeTokenizer, Splitters
from src.debug.prompt import get_sentence
from src.debug.profiler import Profiler as xp
from src.model import AgentModel
from src.storage import Storage
from src.utils import seed_torch
from src.config import Config

Commands.parse_arguments()
xp.setup()

seed_torch(0)


def debug():
    Config.setup_device()
    Config.force_resume = 'y'
    TreeTokenizer.split_functions = [Splitters.sentence_to_words]

    model = AgentModel()
    model.to(Config.device)

    Storage.setup()
    Checkpoints.setup()
    Checkpoints.load(model)

    sentence = get_sentence()

    batch, inputs = TreeTokenizer.batch_texts_to_trees([sentence])
    for parent_key, values in inputs.items():
        for key, value in values.items():
            # inputs[parent_key][key] = value.squeeze(0)
            if Config.use_cuda:
                inputs[parent_key][key] = inputs[parent_key][key].to(Config.device)
    g_loss, disc_loss, main_loss, loss_object, first_A1s, first_pndb_lookup_ids = model.forward(batch, inputs,
                                                                                                debug=True, xm=None)

    print('Main Loss:', main_loss)

    reconstruct_text(batch, model, first_A1s, first_pndb_lookup_ids, exit_on_match=False)


if __name__ == '__main__':
    debug()
