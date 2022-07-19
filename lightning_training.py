import subprocess
from typing import List

import click
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_loading.load_augsburg15 import Augsburg15Dataset, Augmentation
from src.models.object_detector import ClassificationLoss
from src.models.timm_adapter import Network

from src.models.soft_teacher import SoftTeacher


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


NETWORKS = {
    'resnet50': Network.RESNET_50,
    'efficient_net_v2': Network.EFFICIENT_NET_V2,
    'mobile_net_v3': Network.MOBILE_NET_V3,
}
CLASSIFICATION_LOSS_FUNCTIONS = {
    'cross_entropy': ClassificationLoss.CROSS_ENTROPY,
    'focal_loss': ClassificationLoss.FOCAL,
}

augmentations = {
    'vertical_flip': Augmentation.VERTICAL_FLIP,
    'horizontal_flip': Augmentation.HORIZONTAL_FLIP,
    'rotation': Augmentation.ROTATION,
    'rotation_cutoff': Augmentation.ROTATION_CUTOFF,
    'crop': Augmentation.CROP,
}


@click.command()
@click.option(
    '--experiment_name',
    help='Experiment name used for identifying the logs and checkpoints.',
    required=True,
)
@click.option(
    '--checkpoint_path',
    default=None,
    help='Path to the checkpoint from which to resume training.',
)
@click.option(
    '--batch_size',
    default=4,
    help='Batch size for the training.'
)
@click.option(
    '--backbone',
    default='resnet50',
    help='Which pre-trained model to use as a backbone.'
)
@click.option(
    '--min_image_size',
    default=800,
    help='Minimum size of the resized image that is fed into the model.'
)
@click.option(
    '--max_image_size',
    default=1066,
    help='Maximum size of the resized image that is fed into the model.'
)
@click.option(
    '--freeze_backbone',
    default=False,
    help='Whether to freeze the backbone for the training.'
)
@click.option(
    '--classification_loss_function',
    default='cross_entropy',
    help='Which loss function to use for the classification.'
)
@click.option(
    '--data_augmentation',
    default=('vertical_flip', 'horizontal_flip'),
    multiple=True,
    help='Which data augmentations to use for the training.'
)
@click.option(
    '--unsupervised_loss_weight',
    default=1.0,
    help='Factor for scaling the unsupervised loss.'
)
@click.option(
    '--student_only_epochs',
    default=0,
    help='Number of epochs with supervised training only.'
)
@click.option(
    '--train_dataset',
    default='train_synthesized_2016_augsburg15',
    help='Dataset to use for training.'
)
@click.option(
    '--validation_dataset',
    default='validation_synthesized_2016_augsburg15',
    help='Dataset to use for validation.'
)
@click.option(
    '--test_dataset',
    default='test_synthesized_2016_augsburg15',
    help='Dataset to use for test.'
)
def start_experiment(
        experiment_name: str,
        checkpoint_path: str,
        batch_size: int,
        backbone: str,
        min_image_size: int,
        max_image_size: int,
        freeze_backbone: bool,
        classification_loss_function: str,
        data_augmentation: List[str],
        unsupervised_loss_weight: float,
        student_only_epochs: int,
        train_dataset: str,
        validation_dataset: str,
        test_dataset: str,
):
    print(
        f'Starting experiment {experiment_name=} with: {batch_size=}, {backbone=}, {min_image_size=}, '
        f'{max_image_size=}, {freeze_backbone=}, {classification_loss_function=}, {data_augmentation=}, '
        f'{student_only_epochs=}'
    )

    backbone = NETWORKS[backbone]
    classification_loss_function = CLASSIFICATION_LOSS_FUNCTIONS[classification_loss_function]
    data_augmentations = [augmentations[augmentation] for augmentation in data_augmentation]

    train_dataset = Augsburg15Dataset.create_dataset_from_name(train_dataset, data_augmentations)
    validation_dataset = Augsburg15Dataset.create_dataset_from_name(validation_dataset, [])
    test_dataset = Augsburg15Dataset.create_dataset_from_name(test_dataset, [])

    model = SoftTeacher(
        num_classes=Augsburg15Dataset.NUM_CLASSES,
        batch_size=1,
        learning_rate=0.0001,
        teacher_pseudo_roi_threshold=0.9,
        teacher_pseudo_rpn_threshold=0.9,
        student_inference_threshold=0.05,
        unsupervised_loss_weight=unsupervised_loss_weight,
        backbone=backbone,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        min_image_size=800,
        max_image_size=1066,
        freeze_backbone=False,
        classification_loss_function=classification_loss_function,
        student_only_epochs=student_only_epochs,
    )

    log_directory = 'logs'
    experiment_name = f'{experiment_name}#{get_git_revision_short_hash()}'
    logger = TensorBoardLogger(log_directory, experiment_name)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'{log_directory}/{experiment_name}',
        save_last=True,
        save_top_k=1,
        monitor='validation_map_50',
        mode='max',
    )
    trainer = Trainer(
        max_epochs=40,
        logger=logger,
        callbacks=[checkpoint_callback],
        gpus=1 if torch.cuda.is_available() else 0,
        precision=16 if torch.cuda.is_available() else 32,
        progress_bar_refresh_rate=1000,
        accumulate_grad_batches=batch_size,
    )
    trainer.fit(
        model,
        train_dataloaders=model.train_dataloader(),
        val_dataloaders=model.val_dataloader(),
        ckpt_path=checkpoint_path,
    )


if __name__ == '__main__':
    start_experiment()
