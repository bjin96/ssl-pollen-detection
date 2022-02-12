import subprocess

from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset
from src.models.faster_rcnn import PretrainedEfficientNetV2
from src.models.soft_teacher import SoftTeacher


def get_git_revision_short_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()


if __name__ == '__main__':
    model = SoftTeacher(
        num_classes=Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=16,
        teacher_pseudo_threshold=0.9,
        student_inference_threshold=0.5,
        unsupervised_loss_weight=1.0,
        image_size=Augsburg15DetectionDataset.IMAGE_SIZE
    )

    logger = TensorBoardLogger('logs', f'soft_teacher#{get_git_revision_short_hash()}')
    trainer = Trainer(max_epochs=40, logger=logger)
    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
    # CKPT_PATH = './lightning_logs/soft_teacher_loss_weights#25b59bef/checkpoints/epoch=15-step=11807.ckpt'
    # test = trainer.validate(model, model.val_dataloader(), CKPT_PATH)
    # print()
