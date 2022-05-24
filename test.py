import cv2
from tensorflow.python.client import device_lib
from utils.ReID import *

if __name__ == '__main__':
    image = cv2.imread("Datasets/images/25691390_f9944f61b5_z.jpg")
    r, output = extract_masks(image)
    cv2.imshow("", output)
    cv2.waitKey(0)