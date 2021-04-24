from src.config import Config
import torch
import json
import os


class Checkpoints:
    MODELS = os.path.join('models')
    MODEL_FOLDER = os.path.join(MODELS, Config.model_folder)

    @classmethod
    def setup(cls):
        if Config.save_every is None:
            return

        if not os.path.exists(cls.MODEL_FOLDER):
            os.makedirs(cls.MODEL_FOLDER)

        config_file = os.path.join(cls.MODEL_FOLDER, 'config.json')

        if os.path.exists(config_file):  # The config.json already exists
            return

        # Write config to the folder
        config = {}
        for attr in dir(Config):
            if attr.startswith('__') or callable(getattr(Config, attr)):
                continue

            if attr == 'device':
                continue

            config[attr] = getattr(Config, attr)

        with open(config_file, 'w') as f:
            json.dump(config, f)

    @classmethod
    def save(cls, model, epoch, step):
        if Config.save_every is None:
            return

        if step > 0 and step % Config.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cls.MODEL_FOLDER, str(epoch) + '.' + str(step)))

    @classmethod
    def load(cls, model):
        if Config.use_checkpoint is None:
            return

        file = os.path.join(cls.MODELS, Config.use_checkpoint)
        if not os.path.exists(file):
            raise ValueError('The corresponding model folder for loading a checkpoint does not exist.')

        model.load_state_dict(torch.load(file))  # Load model weights

        # Calculate stopping point of checkpoint (Epoch is position 0 but can ignore it)
        step = int(Config.use_checkpoint.split('.')[1]) + 1
        Config.skip_batches = (Config.skip_batches or 0) + step
