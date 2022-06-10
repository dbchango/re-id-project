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

def DPM(r,frame):
    # Cuadro raiz
    x1 = r["rois"][0][0]
    y1 = r["rois"][0][1]
    x2 = r["rois"][0][2]
    y2 = r["rois"][0][3]

    # Cuadro cabeza
    xc1 = int(x1 + (x2 - x1) / 3)
    yc = x1
    xc = y1
    hc = xc1
    wc = y2
    cropc = frame[yc:yc + hc, xc:xc + wc]

    # Cuadro torso
    xt1 = int(x1 + (x2 - x1) * (2 / 3))
    yt = x1 + xc1
    xt = y1
    ht = xt1 - xc1
    wt = y2
    cropt = frame[yt:yt + ht, xt:xt + wt]

    # Cuadro piernas
    xp1 = int(x1 + (x2 - x1))
    yp = xt1
    xp = y1
    hp = xp1 - xt1
    wp = y2
    cropp = frame[yp:yp + hp, xp:xp + wp]

    return cropc,cropt,cropp

def lbp(frame):
    img_gray = frame[:, :, 0].astype(float) * 0.3 + frame[:, :, 1].astype(float) * 0.59 + frame[:, :, 2].astype(float) * 0.11
    lbp = feature.local_binary_pattern(img_gray.astype(np.uint8), 8, 1, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

