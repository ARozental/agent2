# Find the number of levels automatically
import  random
import  os
import numpy as np
import torch
def seed_torch(seed=777):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


seed_torch()

def find_level(inputs):
    current = inputs[0]
    level = 0
    while isinstance(current, list):
        current = current[0]
        level += 1

    return level
