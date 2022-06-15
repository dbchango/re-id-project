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


def crop_frame(y1, y2, x1, x2, frame):
    """
    This function will crop a frame using 4 main coordinates
    :param x1:
    :param y1:
    :param x2:
    :param y2:
    :param frame:
    :return: frame crop
    """
    return frame[y1:y2, x1:x2]


def dpm(roi, frame):
    """
    This function will calculate thee DPM crops (head, torso, legs)
    :param roi:
    :param frame:
    :return: three crops
    """
    # b_box coordinates
    x1, y1 = roi[0], roi[1]
    x2, y2 = roi[2], roi[3]

    # head cropping
    xc1 = int(x1 + (x2 - x1) / 3)
    yc = x1
    xc = y1
    hc = xc1
    wc = y2
    h_crop = crop_frame(yc, yc + hc, xc, xc + wc, frame)  # head crop

    # torso cropping
    xt1 = int(x1 + (x2 - x1) * (2 / 3))
    yt = x1 + xc1
    xt = y1
    ht = xt1 - xc1
    wt = y2
    t_crop = crop_frame(yt, yt + ht, xt, xt + wt, frame)

    # legs cropping
    xl1 = int(x1 + (x2 - x1))
    yl = xt1
    xl = y1
    hl = xl1 - xt1
    wl = y2
    l_crop = crop_frame(yl, yl + hl, xl, xl + wl, frame)

    return h_crop, t_crop, l_crop


def lbp(frame):
    img_gray = frame[:, :, 0].astype(float) * 0.3 + frame[:, :, 1].astype(float) * 0.59 + frame[:, :, 2].astype(float) * 0.11
    return feature.local_binary_pattern(img_gray.astype(np.uint8), 4, 1, method="uniform")


def cal_hist(input):
    (hist, _) = np.histogram(input.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def calc_lbph(image):
    lbp_result = lbp(image)
    return cal_hist(lbp_result)  # returning lbp histogram
