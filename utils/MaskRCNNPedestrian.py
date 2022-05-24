import pixellib
from pixellib.instance import instance_segmentation

class MaskRCNNPedestrian:
    def __init__(self, weights_path="models/mask_rcnn_coco.h5"):
        self.weights_path = weights_path

    def load_model(self):
        seg = instance_segmentation()
        seg.load_model(self.weights_path)
        return seg