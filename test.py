import cv2
from utils.Camera import Camera
from utils.ReID import *
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    width = 0
    height = 0
    # reading two streamings
    for i, j in zip(Camera("Datasets/videos/chaplin.mp4").read_video(extract_masks), Camera("Datasets/videos/VIRAT_S_000002.mp4").read_video(extract_masks)):
        if i.shape[0] > j.shape[0]:
            height = j.shape[0]
            width = j.shape[1]
            i = cv2.resize(i, (width, height))
        else:
            height = i.shape[0]
            width = i.shape[1]
            j = cv2.resize(j, (width, height))
        k = np.concatenate((i, j), axis=1)
        cv2.imshow("Output", k)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break