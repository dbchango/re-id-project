import cv2
from utils.MaskRCNNPedestrian import MaskRCNNPedestrian
from tensorflow.python.client import device_lib
from utils.MaskRCNNPedestrian import MaskRCNNPedestrian
class_names = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                 'bus', 'train', 'truck', 'boat', 'traffic light',
                 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                 'teddy bear', 'hair drier', 'toothbrush']

if __name__ == '__main__':
    image = cv2.imread("Datasets/images/25691390_f9944f61b5_z.jpg")

    seg = MaskRCNNPedestrian().load_model()
    target_classes = seg.select_target_classes(person=True)
    print(seg.segmentFrame(image, segment_target_classes=target_classes).shape)

    # pame funcion -> salida()
