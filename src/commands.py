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

        parser.add_argument('--gpu', type=int, default=0,
                            help='the CUDA GPU to place the model on')

        args = parser.parse_args()

        Config.gpu_num = args.gpu
        if args.config is not None:
            Commands.load_config(args.config)

    @staticmethod
    def load_config(filename):
        with open('configs/' + filename + '.json') as f:
            data = json.load(f)

        for key, value in data.items():
            setattr(Config, key, value)
