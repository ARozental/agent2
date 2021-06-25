from src.commands import Commands
from src.config import Config
from src.datasets import TestExample
from src.pre_processing import TreeTokenizer, worker_init_fn
from src.storage import Storage
from src.model import AgentModel
from src.debug.profiler import Profiler as xp
from torch.utils.data.dataloader import DataLoader
import torch
import torch.optim.lr_scheduler

Commands.parse_arguments()
xp.setup()
Config.setup_device()


# Need to wrap in a function for the child workers
def test_model():
    model_file_location = "/Users/alonrozental/IdeaProjects/agent2/models/s256_pndb128_m2"
    Config.device = torch.device('cpu')
    model = AgentModel()
    #model.to(Config.device)
    Storage.setup()
    with Storage.fs.open(model_file_location, 'rb') as f:
      model.load_state_dict(torch.load(f,map_location=torch.device('cpu')))  # Load model weights
    Config.batch_size = 1
    dataset = TestExample("4 5 6")
    dataloader = DataLoader(
      dataset,
      batch_size=Config.batch_size,
      collate_fn=TreeTokenizer.batch_texts_to_trees,
      worker_init_fn=worker_init_fn,
      num_workers=1,
      persistent_workers=True  # This is helpful when num_workers > 0
    )
    example = [x for x in dataloader][0]
    g_loss, disc_loss, main_loss, loss_object = model.forward(example,debug=True,generate=True)
    #print(loss_object)
    nodes = example.batch_root.children
    expected = [TreeTokenizer.deep_detokenize(node.build_struct(return_eos=True)[0], Config.agent_level)
                for node in nodes]
    reconstructed = [model.full_decode(example.level_nodes[i][:5]) for i in range(Config.agent_level + 1)]

    reconstructed = [[TreeTokenizer.deep_detokenize(node[0], i) for node in items] for i, items in enumerate(reconstructed)]
    for i, text in enumerate(reconstructed):
      print('Level', i,"r", text)
      print('Level', i,"g", model.generate_texts(i, 1)[0])
      print('Level', i,"g", model.generate_texts(i, 1)[0])





if __name__ == '__main__':
    test_model()
