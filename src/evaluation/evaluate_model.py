from pathlib import Path

from pytorch_lightning import Trainer

from src.models.soft_teacher import SoftTeacher


if __name__ == '__main__':
    log_directory = Path(__file__).parents[2] / 'logs'
    ckpt_path = log_directory / 'soft_teacher#352b63b/version_0/checkpoints/epoch=18-step=14021.ckpt'
    model = SoftTeacher.load_from_checkpoint(ckpt_path)
    trainer = Trainer(max_epochs=1)
    test = trainer.validate(model, model.val_dataloader())
