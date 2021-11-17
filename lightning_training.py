from pytorch_lightning import Trainer

from data_loading.load_augsburg15 import Augsburg15DetectionDataset
from models.faster_rcnn import PretrainedEfficientNetV2
from models.pretrained_torchvision_models import PretrainedMobileNet

if __name__ == '__main__':
    model = PretrainedEfficientNetV2(
        Augsburg15DetectionDataset.NUM_CLASSES,
        batch_size=4
    )

    trainer = Trainer(gpus=1, max_epochs=40)
    trainer.fit(model, model.train_dataloader(), model.val_dataloader())
