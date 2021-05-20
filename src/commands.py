from src.config import Config
import argparse
import json


class Commands:
    @staticmethod
    def parse_arguments():
        parser = argparse.ArgumentParser(
            usage='python train.py [options]',
            description='Train agent model'
        )

        parser.add_argument('-c', '--config', type=str,
                            help='the name of the json file in `configs/` to load')

        parser.add_argument('--log', action='store_true', default=False,
                            help='Turn on "log_experiment" for Tensorboard')

        parser.add_argument('--exp_folder', type=str, default=None,
                            help='Value for "exp_folder"')

        parser.add_argument('--skip', type=int, default=None,
                            help='value for the `skip_batches` config')

        parser.add_argument('--gpu', type=int, default=0,
                            help='the CUDA GPU to place the model on')

        parser.add_argument('--tpu', action='store_true', default=False,
                            help='Whether to use a single TPU core')

        parser.add_argument('--tpu-all', action='store_true', default=False,
                            help='Whether to use all TPU cores')

        parser.add_argument('--debug-tpu', action='store_true', default=False,
                            help='Whether or not to print out TPU metrics')

        parser.add_argument('--dummy', action='store_true', default=False,
                            help='Whether to use the dummy dataset')

        parser.add_argument('--gcs', action='store_true', default=False,
                            help='Use the GCS Bucket')

        args = parser.parse_args()

        if args.config is not None:
            Commands.load_config(args.config)
        if args.exp_folder is not None:
            Config.exp_folder = args.exp_folder
        Config.gpu_num = args.gpu
        Config.use_tpu = args.tpu or args.tpu_all
        Config.tpu_all = args.tpu_all
        Config.debug_tpu = args.debug_tpu
        if args.dummy:
            Config.use_dummy_dataset = True
        if args.skip is not None:
            Config.skip_batches = args.skip

        if args.log:
            Config.log_experiment = True

        if args.gcs:
            Config.storage_location = 'gs://agent_output/'

    @staticmethod
    def load_config(filename):
        with open('configs/' + filename + '.json') as f:
            data = json.load(f)

        for key, value in data.items():
            setattr(Config, key, value)
