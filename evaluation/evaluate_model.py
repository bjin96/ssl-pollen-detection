import torch
from torch.utils.data import DataLoader

from training.engine import evaluate
from fine_tune_faster_rcnn import get_fasterrcnn_model
from training.transforms import ToTensor
import os
from data_loading.load_augsburg15 import Augsburg15DetectionDataset, collate_augsburg15_detection


def evaluate_saved_model(saved_model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    validation_dataset = Augsburg15DetectionDataset(
        root_directory=os.path.join('../datasets/pollen_only'),
        image_info_csv='pollen15_val_annotations_preprocessed.csv',
        transforms=ToTensor()
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=16,
        collate_fn=collate_augsburg15_detection,
        drop_last=True,
        num_workers=4
    )

    model = get_fasterrcnn_model()
    model.load_state_dict(torch.load(saved_model_path))
    model.to(device)

    evaluate(model, validation_loader, device)


if __name__ == '__main__':
    evaluate_saved_model('../models/model_epoch_25')
