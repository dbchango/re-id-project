from utils.histogram_color_RGB import RGBHistogram
from utils.ReID import *
import os

import numpy as np
from matplotlib import pyplot as plt
from utils.ReID import *
import cv2
import csv
from keras.preprocessing.image import ImageDataGenerator
from utils.MaskRCNN import MaskRCNN
from utils.LocalBinaryPatterns import LocalBinaryPatterns
from PIL import Image
def write_csv(path, header, data):
    """
    This function saves a .csv file with given data
    :param path: The target location to save .csv file
    :param header: Columns names
    :param data: Dataframe data
    :return: None
    """
    with open(path, 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        if header is not None:
            writer.writerow(header)
        writer.writerows(data)

data = []  # Will hold the RGB feature vectors of each image.
target = []  # Will hold the label corresponding to each image.
lbp_2 = LocalBinaryPatterns(3)
descriptor = RGBHistogram([8, 8, 8])
model = MaskRCNN()
header = ['area', 'perimeter', 'class']
parent_root = 'Datasets/base/validation'
target_root = 'Datasets/espe/lbp_histograms/validation'
csv_path = 'Datasets/espe/lbp_histograms/validation/validation.csv'

for person in os.listdir(parent_root):
    person_root = os.path.join(parent_root, person)
    target_person_root = os.path.join(target_root, person)
    print("Working on: ", person_root)
    counter = 0
    histograms = []
    for instance in os.listdir(person_root):
        if instance.endswith('jpg') is False:
            continue
        counter += 1
        name = 'person_{}_{}.jpg'.format(person, counter)
        print('Processing {} ...'.format(name))
        source_path = os.path.join(person_root, instance)
        print("Opening ", source_path)
        image = cv2.imread(source_path)
        image_cp = image.copy()
        r, _ = model.segment(image)
        if len(r['masks']) != 0 and len(r['rois']) != 0:
            # for i in range(len(r["rois"])):
            mask = r["masks"][:, :, 0].astype('uint8')
            mask_cp = mask.copy()
            area, perimeter = mask_area_perimeter(mask_cp)
            class_id = person

            masked_image = apply_mask(image_cp, mask)
            x1, y1 = r["rois"][0][0], r["rois"][0][1]
            x2, y2 = r["rois"][0][2], r["rois"][0][3]
            cropped_frame = crop_frame(x1, x2, y1, y2, masked_image)
            hist = descriptor.describe(cropped_frame)
            #print(hist)
            hist = np.append(hist, person)
            histograms.append(hist)
            result_name = os.path.join(target_person_root, name)
            data.append([area, perimeter, class_id])

    csv_path_pr = target_person_root + '/hist_{}.csv'.format(person)
    write_csv(path=csv_path_pr, header=None, data=histograms)
write_csv(csv_path, header, data)




# image = cv2.imread("Datasets/base/test/3/003_5 21.jpg")
# image_cp = image.copy()
# r, _ = model.segment(image)

# mask = r["masks"][:, :, 0].astype('uint8')
# mask_cp = mask.copy()


# masked_image = apply_mask(image_cp, mask)
# x1, y1 = r["rois"][0][0], r["rois"][0][1]
# x2, y2 = r["rois"][0][2], r["rois"][0][3]
# cropped_frame = crop_frame(x1, x2, y1, y2, masked_image)
# hist = descriptor.describe(cropped_frame)
# print(hist)



