from src.config import Config
import torch
import json
import os


class Checkpoints:
    MODEL_FOLDER = os.path.join('models', Config.model_folder)

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
    def save(cls, step, model):
        if Config.save_every is None:
            return

        if step > 0 and step % Config.save_every == 0:
            torch.save(model.state_dict(), os.path.join(cls.MODEL_FOLDER, str(step)))

    @classmethod
    def load(cls):
        # TODO - load model weights

        # TODO - resume at dataset
        raise NotImplementedError
