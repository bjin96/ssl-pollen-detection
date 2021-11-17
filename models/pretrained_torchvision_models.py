import model_definition
from model_definition.faster_rcnn import FastRCNNPredictor
from models.object_detector import ObjectDetector


class PretrainedMobileNet(ObjectDetector):

    def define_model(self):
        model = model_definition.faster_rcnn.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        return model
