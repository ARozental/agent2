# README

## Command Line Execution

```
usage: python train.py [options]

Train agent model

optional arguments:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        the name of the json file in `configs/` to load
  --gpu GPU             the CUDA GPU to place the model on

```

## Debug Model

Call `python debug.py` with the corresponding config and the last saved model will automatically be used.

Use the `use_checkpoint` argument to load a specific epoch/step.

```
usage: python debug.py [options]

Debug a trained agent model

arguments:
  -c CONFIG, --config CONFIG
                        the name of the json file in `configs/` to load
  --gpu GPU             the CUDA GPU to place the model on
  --model_folder        the location of the saved model
  --use_checkpoint      the filename of a specific epoch/step of a model to load
```