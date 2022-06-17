import os
import cv2
import numpy as np
from utils.ReID import mask_area_perimeter
from utils.ReID import DPM
from utils.ReID import lbp_feature
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        r, output = extract_masks(frame)
        if r["rois"] != []:
            silhouette=r['masks'][:,:,0]
            test = np.array(silhouette, dtype='uint8')
            area , perimeter = mask_area_perimeter(test)
            print(area,perimeter)

        if r["rois"] != []:
            cropc, cropt, cropp = DPM(r,frame)
            # Cuadro cabeza
            lbp_feature(cropc)
            print(lbp_feature(cropc))
            # Cuadro torso
            lbp_feature(cropt)
            print(lbp_feature(cropt))
            # # Cuadro piernas
            lbp_feature(cropp)
            print(lbp_feature(cropp))
        yield r, output

