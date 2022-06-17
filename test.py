import cv2
from matplotlib import pyplot as plt
# from utils.Camera import Camera
from utils.ReID import *
from utils.SequenceReader import read_sequence
# import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    width = 0
    height = 0
    directory_cam1 = 'Datasets/Propio/Video/pasillo/cam_1/pasillo_001'
    directory_cam2 = 'Datasets/Propio/Video/pasillo/cam_2/pasillo_001'
    # reading n streamings

    # for i, j in zip(Camera("Datasets/videos/chaplin.mp4").read_video(extract_masks), Camera("Datasets/videos/VIRAT_S_000002.mp4").read_video(extract_masks)):
    for (r1, o1), (r2, o2) in zip(read_sequence(directory_cam1, extract_masks), read_sequence(directory_cam2, extract_masks)):

        k = np.concatenate((o1, o2), axis=1)
        cv2.imshow("Output", k)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break