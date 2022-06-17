import cv2

from utils.Camera import Camera
from utils.ReID import extract_masks
import numpy as np

if __name__ == '__main__':
    source_1 = 'Datasets/Propio/cameras/pasillo/cam_1/soft/Pasillo_001_sf.mp4'
    source_2 = 'Datasets/Propio/cameras/pasillo/cam_2/soft/Pasillo_001_sf.mp4'

    for o1, o2 in zip(Camera(src=source_1).read_video(extract_masks), Camera(src=source_2).read_video(extract_masks)):
        o = np.concatenate((o1, o2), axis=1)
        cv2.imshow("Output", o)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
