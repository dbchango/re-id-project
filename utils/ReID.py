from cv2 import CV_16S
from utils.MaskRCNN import MaskRCNN
from skimage.feature import local_binary_pattern
from skimage import feature
from skimage.color import label2rgb
import numpy as np
import matplotlib.pyplot as plt
import cv2

seg = MaskRCNN()


def extract_texture():

    return None # tensor


def compare_textur_info(item1, item2, threshold):
    return None

def apply_mask(rgb_img, mask):
    rgb_img_cp = rgb_img.copy()
    rgb_img_cp[:, :, 1] = rgb_img[:, :, 1] * mask
    rgb_img_cp[:, :, 2] = rgb_img[:, :, 2] * mask
    rgb_img_cp[:, :, 0] = rgb_img[:, :, 0] * mask
    return rgb_img_cp

def extract_masks(frame):
    """
    This function detect and segment pedestrians in an image and returns processed image
    :param frame:
    :return results, processed_image:
    """
    r, output = seg.segment(frame)

    # TODO: add area calculation function (@PAMELA), this function will calculate tha area of
    #  each person mask and add those to previous maskrcnn result

    return r, output


def removing_background(mask, crop):
    frame_bg = mask.astype(int) * crop


def crop_frame(a, b, c, d, frame):
    """
    This function will crop a frame using 4 main coordinates
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param frame:
    :return: frame crop
    """
    return frame[a:b, c:d]


def dpm(roi, frame):
    """
    This function will calculate thee DPM crops (head, torso, legs)
    :param roi:
    :param frame:
    :return: three crops
    """

    height = abs(roi[2] - roi[0])

    h_crop = crop_frame(roi[0], int(roi[0] + height / 3), roi[1], roi[3], frame)
    t_crop = crop_frame(int(roi[0] + height / 3), int(roi[0] + height * (2 / 3)), roi[1], roi[3], frame)
    l_crop = crop_frame(int(roi[0] + height * (2 / 3)), roi[2], roi[1], roi[3], frame)

    return h_crop, t_crop, l_crop


def lbp(frame):
    # img_gray = frame[:, :, 0].astype(float) * 0.3 + frame[:, :, 1].astype(float) * 0.59 + frame[:, :, 2].astype(float) * 0.11
    
    img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    print(type(img_gray))
    
    return feature.local_binary_pattern(img_gray.astype(np.uint8), 4, 1, method="uniform")


def cal_hist(input):
    (hist, _) = np.histogram(input.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def calc_lbph(image):
    lbp_result = lbp(image)
    return cal_hist(lbp_result)  # returning lbp histogram
