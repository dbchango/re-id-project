import sys
import os

from utils.class_names import class_names
# loading paths
ROOT_DIR = os.getcwd()
ROOT_DIR = os.path.join(ROOT_DIR, "Mask_RCNN")

# import MaskRCNN
sys.path.append(ROOT_DIR) # to find local version of the library
sys.path.append(os.path.join(ROOT_DIR, "Mask_RCNN/samples/coco/"))

import Mask_RCNN.coco

# batch_size = 3

from mrcnn import utils
import mrcnn.model as modellib

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")


# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

from Mask_RCNN.samples.coco import coco



class MaskRCNN:
    def __init__(self, batch_size):
        class InferenceConfig(coco.CocoConfig):

            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = batch_size
        # Download COCO trained weights from Releases if needed
        if not os.path.exists(COCO_MODEL_PATH):
            utils.download_trained_weights(COCO_MODEL_PATH)
        self.config = InferenceConfig()
        self.config.display()

    def loadmodel(self):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=self.config)
        model.load_weights(COCO_MODEL_PATH, by_name=True)
        return model
