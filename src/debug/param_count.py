from collections import OrderedDict
from src.config import Config

from prettytable import PrettyTable


def count_parameters(model, trainable=True):
    params = {i: {} for i in range(Config.agent_level + 1)}
    for name, param in model.named_parameters():
        if trainable and not param.requires_grad:
            continue

        if not name.startswith('agent_levels.'):
            continue

        parts = name.split('.')
        level_index = int(parts[1])
        if len(parts) > 3:
            group_name = parts[2]
        else:
            group_name = 'other'

        if group_name not in params[level_index]:
            params[level_index][group_name] = {'count': 0, 'size': 0}

        params[level_index][group_name]['count'] += 1
        params[level_index][group_name]['size'] += param.numel()

    print_parameters(params)

    exit()


def print_parameters(params):
    # Sort the dictionary based on key
    for i in range(Config.agent_level + 1):
        params[i] = OrderedDict(sorted(params[i].items(), key=lambda x: x[0].lower()))

    total_params = 0
    total_size = 0
    print('')
    print('Trainable Parameters')
    for i in range(Config.agent_level + 1):
        print(f'Agent Level: {i}')
        table = PrettyTable(['Group', 'Parameters', 'Size'])
        table.align = 'c'
        current_params = 0
        current_size = 0
        for key, values in params[i].items():
            table.add_row([key, f"{values['count']:,}", f"{values['size']:,}"])
            current_params += values['count']
            current_size += values['size']
        print(table)
        print(f'Parameters: {current_params:,}')
        print(f'Size: {current_size:,}')
        print('')

        total_params += current_params
        total_size += current_size

    print(f'Total Parameters: {total_params:,}')
    print(f'Total Size: {total_size:,}')
