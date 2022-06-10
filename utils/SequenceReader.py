import os
import cv2
from utils.ReID import DPM
from utils.ReID import lbp
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        r, output = extract_masks(frame)
        if r["rois"] != []:
            cropc, cropt, cropp = DPM(r,frame)
            # Cuadro cabeza
            lbp(cropc)
            # Cuadro torso
            lbp(cropt)
            # Cuadro piernas
            lbp(cropp)
        yield r, output
