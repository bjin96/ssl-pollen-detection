import pickle
from pathlib import Path
from typing import List

import click
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.data_loading.load_augsburg15 import Augsburg15Dataset, collate_augsburg15_detection
import torch

from src.models.soft_teacher import SoftTeacher


def _get_evaluation_data_loader(evaluation_dataset_name):
    root_directory, image_info_csv, dataset_type = Augsburg15Dataset.DATASET_MAPPING[evaluation_dataset_name]
    dataset = Augsburg15Dataset(
        root_directory=root_directory,
        image_info_csv=image_info_csv,
        transforms=[],
        dataset_type=dataset_type,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )


def _make_validation_predictions(checkpoint_path, evaluation_dataset_name):
    data_loader = _get_evaluation_data_loader(evaluation_dataset_name)

    model = SoftTeacher.load_from_checkpoint(
        checkpoint_path,
        num_classes=Augsburg15Dataset.NUM_CLASSES,
        batch_size=1,
        train_dataset=None,
        validation_dataset=None,
        test_dataset=None,
    )
    model.eval()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    results = []

    for index, sample in enumerate(data_loader):
        image, target = sample

        image = image.to(device)

        result = model(image)
        results.append(
            (result[0]['boxes'].detach().cpu(), result[0]['labels'].detach().cpu(), result[0]['scores'].detach().cpu()))

    with open(checkpoint_path.parent / f'predictions_{evaluation_dataset_name}.pkl', 'wb') as file:
        pickle.dump(results, file)


def _run_test_set(checkpoint_path, evaluation_dataset_name):
    data_loader = _get_evaluation_data_loader(evaluation_dataset_name)

    model = SoftTeacher.load_from_checkpoint(
        checkpoint_path,
        num_classes=Augsburg15Dataset.NUM_CLASSES,
        batch_size=1
    )

    trainer = Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16,
    )
    results = trainer.test(
        model,
        dataloaders=data_loader,
        ckpt_path=checkpoint_path
    )
    with open(checkpoint_path.parent / f'results_{evaluation_dataset_name}.pkl', 'w') as file:
        file.write(str(results))


def _run_evaluation_for_experiment(
        ckpt_path: Path,
        evaluation_datasets: List[str]
):
    ckpt_path = Path(ckpt_path)
    for evaluation_dataset in evaluation_datasets:
        _make_validation_predictions(ckpt_path, evaluation_dataset)
        _run_test_set(ckpt_path, evaluation_dataset)


@click.command()
@click.option(
    '--checkpoint_path',
    required=True,
    multiple=True,
    help='Which checkpoints to use for evaluation.'
)
@click.option(
    '--evaluation_dataset_group',
    default='evaluate_2016augsburg15',
    help='Which datasets to use for evaluation.'
)
def run_evaluation_for_experiments(
        checkpoint_path,
        evaluation_dataset_group: str
):
    if evaluation_dataset_group == 'evaluate_2016augsburg15':
        evaluation_datasets = [
            'validation_synthesized_2016_augsburg15',
            'test_synthesized_2016_augsburg15',
            'test_synthesized_manual_set'
        ]
    elif evaluation_dataset_group == 'evaluate_2016+2018augsburg15_raw':
        evaluation_datasets = [
            'validation_raw_2016_2018_augsburg15',
            'test_raw_2016_2018_augsburg15',
            'test_raw_manual_set'
        ]
    elif evaluation_dataset_group == 'evaluate_2016+2018augsburg15_synthesised':
        evaluation_datasets = [
            'validation_synthesized_2016_2018_augsburg15',
            'test_synthesized_2016_2018_augsburg15',
            'test_synthesized_manual_set'
        ]
    else:
        raise ValueError(f'No such evaluation_dataset_group: {evaluation_dataset_group}')

    for ckpt_path in checkpoint_path:
        _run_evaluation_for_experiment(ckpt_path, evaluation_datasets)


if __name__ == '__main__':
    run_evaluation_for_experiments()
