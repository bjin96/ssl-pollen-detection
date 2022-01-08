from pytorch_lightning import Trainer

from src.data_loading.load_augsburg15 import Augsburg15DetectionDataset
from src.models.faster_rcnn import PretrainedEfficientNetV2
from src.models.soft_teacher import SoftTeacher

if __name__ == '__main__':
    model = SoftTeacher(
        Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=4
    )

    trainer = Trainer(max_epochs=40)
    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
    # CKPT_PATH = 'lightning_logs/pretrained_timm_default_feature_pyramid/checkpoints/epoch=24-step=36924.ckpt'
    # trainer.test(model, model.test_dataloader())
