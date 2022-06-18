import os
import cv2
import numpy as np
from utils.ReID import mask_area_perimeter
from utils.ReID import dpm, calc_lbph, apply_mask
from matplotlib import pyplot as plt
def read_sequence(path, extract_masks):
    for image_path in os.listdir(path):
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        scale = 10
        width = int(frame.shape[1] * scale / 100)
        height = int(frame.shape[0] * scale / 100)
        frame = cv2.resize(frame, (width, height))
        frame_cp = frame.copy()
        r, output = extract_masks(frame)
        silhouette=r['masks'][:,:,0]
        test = np.array(silhouette, dtype='uint8')
        area , perimeter = mask_area_perimeter(test)
        print(area,perimeter)
        if len(r["rois"]) != 0 and len(r["masks"]) != 0:
            for i in range(len(r["rois"])):
                # temp_frame = frame_cp.copy()

                mask = r["masks"][:, :, i].astype(int)
                temp_frame = apply_mask(frame_cp, mask)
                h_crop, t_crop, l_crop = dpm(r["rois"][i], temp_frame)
                plt.imshow(h_crop)
                plt.imshow(t_crop)
                plt.imshow(l_crop)
                plt.show()
                # Cuadro cabeza
                head_hg = calc_lbph(h_crop)
                # Cuadro torso
                torso_hg = calc_lbph(t_crop)
                # Cuadro piernas
                legs_hg = calc_lbph(l_crop)
        yield r, output

