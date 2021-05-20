from src.config import Config
from src.storage import Storage
from fsspec.implementations.local import LocalFileSystem
import gcsfs
import torch
import json
import os


class Checkpoints:
    fs = None
    MODELS = None
    MODEL_FOLDER = None

    @classmethod
    def setup(cls):
        models = ['models']
        if Config.storage_location is not None:
            models.insert(0, Config.storage_location)
        cls.MODELS = os.path.join(*models)

        if Config.storage_location is not None:
            cls.fs = gcsfs.GCSFileSystem()
        else:
            cls.fs = LocalFileSystem()

        cls.MODEL_FOLDER = os.path.join(cls.MODELS, Config.model_folder)
        if Config.save_every is None:
            return

        if not cls.fs.exists(cls.MODEL_FOLDER):
            cls.fs.makedirs(cls.MODEL_FOLDER)

        config_file = os.path.join(cls.MODEL_FOLDER, 'config.json')

        if cls.fs.exists(config_file):  # The config.json already exists
            return

        # Write config to the folder
        config = {}
        for attr in dir(Config):
            if attr.startswith('__') or callable(getattr(Config, attr)):
                continue

            if attr == 'device':
                continue

            config[attr] = getattr(Config, attr)

        with cls.fs.open(config_file, 'w') as f:
            json.dump(config, f)

    @classmethod
    def save(cls, model, epoch, step):
        if Config.save_every is None:
            return

        if step > 0 and step % Config.save_every == 0:
            with cls.fs.open(os.path.join(cls.MODEL_FOLDER, str(epoch) + '.' + str(step)), 'w') as f:
                torch.save(model.state_dict(), f)

    @classmethod
    def find_existing_model(cls):
        if not cls.fs.exists(cls.MODEL_FOLDER):
            return None

        checkpoint_file = None
        checkpoint_epoch = -1
        checkpoint_step = -1
        for file in cls.fs.listdir(cls.MODEL_FOLDER):
            filename = os.path.basename(file['name'])
            if filename == 'config.json':
                continue

            if '.' not in filename:
                continue

            parts = filename.split('.')
            try:
                epoch, step = int(parts[0]), int(parts[1])
            except ValueError:
                continue

            if epoch > checkpoint_epoch:
                checkpoint_epoch = epoch
                checkpoint_step = step
                checkpoint_file = filename
            elif epoch == checkpoint_epoch and step > checkpoint_step:
                checkpoint_epoch = epoch
                checkpoint_step = step
                checkpoint_file = filename

        return checkpoint_file

    @classmethod
    def load(cls, model):
        checkpoint_file = Config.use_checkpoint
        if checkpoint_file is None:
            # See if a model already exists in the folder
            existing_model = cls.find_existing_model()
            if existing_model is not None:
                should_resume = input('An existing version exists in the folder `' + str(cls.MODEL_FOLDER) + '`.  ' +
                                      'Would you like to resume from the latest file `' + str(existing_model) + '`? ' +
                                      '[y/n] ')
                if should_resume.lower() in ['y', 'yes']:
                    checkpoint_file = os.path.join(Config.model_folder, existing_model)

        if checkpoint_file is None:
            return

        file = os.path.join(cls.MODELS, checkpoint_file)
        if not cls.fs.exists(file):
            raise ValueError('The corresponding model folder for loading a checkpoint does not exist.')

        with cls.fs.open(file) as f:
            model.load_state_dict(torch.load(f))  # Load model weights

        # Calculate stopping point of checkpoint (Epoch is position 0 but can ignore it)
        step = int(checkpoint_file.split('.')[1]) + 1
        Config.skip_batches = (Config.skip_batches or 0) + step
