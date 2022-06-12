import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from utils.ReID import dpm
from utils.ReID import lbp, crop_frame


def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        frame_cp = frame.copy()
        r, output = extract_masks(frame)
        if len(r["rois"]) != 0 and len(r["masks"]) != 0:
            for i in range(len(r["rois"])):

                temp_frame = frame_cp.copy()
                #mask = r['masks']
                mask = r["masks"][:, :, i].astype(int)
                temp_frame[:, :, 0] = frame_cp[:, :, 0] * mask
                temp_frame[:, :, 1] = frame_cp[:, :, 1] * mask
                temp_frame[:, :, 2] = frame_cp[:, :, 2] * mask
                h_crop, t_crop, l_crop = dpm(r["rois"][i], temp_frame)

                # Cuadro cabeza
                head_hg = lbp(h_crop)
                # Cuadro torso
                torso_hg = lbp(t_crop)
                # Cuadro piernas
                legs_hg = lbp(l_crop)
        yield r, output
