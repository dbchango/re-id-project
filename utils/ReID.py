from skimage import feature
import numpy as np
import cv2


def apply_mask(rgb_img, mask):
    rgb_img_cp = rgb_img.copy()
    rgb_img_cp[:, :, 1] = rgb_img[:, :, 1] * mask
    rgb_img_cp[:, :, 2] = rgb_img[:, :, 2] * mask
    rgb_img_cp[:, :, 0] = rgb_img[:, :, 0] * mask
    return rgb_img_cp


def removing_background(mask, crop):
    frame_bg = mask.astype(int) * crop


def crop_frame(a, b, c, d, frame):
    """
    This function will crop a frame using 4 main coordinates
    :param a: start row coordinate
    :param b: end row coordinate
    :param c: start column coordinate
    :param d: end column coordinate
    :param frame: image or frame that will be cropped
    :return: cropped frame
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

    return feature.local_binary_pattern(img_gray.astype(np.uint8), 4, 1, method="uniform")


def cal_hist(input):
    (hist, _) = np.histogram(input.ravel(), bins=np.arange(0, 8 + 3), range=(0, 8 + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def mask_area_perimeter(segmask):
    end_area = []
    end_perimeter=[]
    #Buscas el contorno
    contornos,hierarchy = cv2.findContours(segmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for i in range (len(contornos)):
        cv2.drawContours(segmask, contornos, i, (255,0,0), 3)
    #Buscas el contorno más grande
    lista_areas = []
    #calcular area y perimetro
    for c in contornos:
        perimeter = cv2.arcLength(c,True)
        area = cv2.contourArea(c)
        lista_areas.append(area)
    mas_grande = contornos[lista_areas.index(max(lista_areas))]
    #Representas el contorno más grande
    area = cv2.contourArea(mas_grande)
    #x,y,w,h = cv2.boundingRect(mas_grande)
    end_area.append(area)
    end_perimeter.append(perimeter)
    return area, perimeter


def calc_lbph(image):
    lbp_result = lbp(image)
    return cal_hist(lbp_result)  # returning lbp histogram

