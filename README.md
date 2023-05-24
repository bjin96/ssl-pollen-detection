# Semi-supervised Pollen Grain Detection

This repository contains the semi-supervised implementation of pollen detection described in our paper 
[Airborne pollen grain detection from partially labelled data utilising semi-supervised learning](https://doi.org/10.1016/j.scitotenv.2023.164295).
For the supervised part of the implementation please check out https://github.com/MilesGrey/pollen-detection.

## Installation

Please install the necessary python packages via pip, i.e. like so:

```commandline
pip install -r requirements.txt
```

Furthermore, the datasets should be made available in a directory `datasets` located at the root of the project.
Datasets are made available upon request.

## Training

The code provides a simple command line interface built with [click](https://palletsprojects.com/p/click/) to train 
neural networks. Training can be started by running:

```commandline
python lightning_training.py --experiment_name=<EXPERIMENT_NAME>
```

There are several options to customize the training a full list of options can be shown with command:

```commandline
python lightning_training.py --help
```

The permitted arguments for non-self-explanatory options are detailed below.

| Option                         | Values                                                                                                                         |
|:-------------------------------|:-------------------------------------------------------------------------------------------------------------------------------|
| --backbone                     | 'resnet50', 'efficient_net_v2', 'mobile_net_v3'                                                                                |
| --classification_loss_function | 'cross_entropy', 'focal_loss'                                                                                                  |
| --data_augmentation            | 'vertical_flip', 'horizontal_flip', 'rotation', 'rotation_cutoff', 'crop'                                                      |
| --train_dataset                | 'train_synthesized_2016_augsburg15', 'train_synthesized_2016_2018_augsburg15', 'train_raw_2016_2018_augsburg15'                |
| --validation_dataset           | 'validation_synthesized_2016_augsburg15', 'validation_synthesized_2016_2018_augsburg15', 'validation_raw_2016_2018_augsburg15' |
| --test_dataset                 | 'test_synthesized_2016_augsburg15', 'test_synthesized_2016_2018_augsburg15', 'test_raw_2016_2018_augsburg15'                   |

The `--data_augmentation` option can be used repeatedly to specify multiple data augmentations.

## Evaluation

Similarly to training, you can run the evaluation as follows:

```commandline
python run_evaluation.py --checkpoint_path=<PATH_TO_TRAINED_MODEL> --evaluation_dataset_group=<EVALUATION_DATASET_GROUP>
```

Here,  `--evaluation_dataset_group` can have the following values:

| Option                     | Values                                                                                                    |
|:---------------------------|:----------------------------------------------------------------------------------------------------------|
| --evaluation_dataset_group | 'evaluate_2016augsburg15', 'evaluate_2016+2018augsburg15_raw', 'evaluate_2016+2018augsburg15_synthesised' |


## Citation


```
@article{jin2023,
    title = {Airborne pollen grain detection from partially labelled data utilising semi-supervised learning},
    journal = {Science of The Total Environment},
    pages = {164295},
    year = {2023},
    issn = {0048-9697},
    doi = {https://doi.org/10.1016/j.scitotenv.2023.164295},
    url = {https://www.sciencedirect.com/science/article/pii/S0048969723029169},
    author = {Benjamin Jin and Manuel Milling and Maria Pilar Plaza and Jens O. Brunner and Claudia Traidl-Hoffmann and Björn W. Schuller and Athanasios Damialis},
    keywords = {Aerobiology, Automatic detection, Object detection, Semi-supervised learning, Deep learning, Pollen taxonomy},
}
```