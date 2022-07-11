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

aug_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)

def data_augmentation(path, target_path):
    model = MaskRCNN()
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        target_person_path = os.path.join(target_path, person)
        counter = 0
        for image_sample in os.listdir(person_path):
            image_sample_path = os.path.join(person_path, image_sample)
            image = cv2.imread(image_sample_path)
            image = np.expand_dims(image, 0)
            aug_iter = aug_generator.flow(image)
            aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
            for k in range(len(aug_images)):
                counter += 1
                aug_item = aug_images[k]
                aug_item_cp = aug_item.copy()
                r, _ = model.segment(aug_images[k])
                if len(r['masks']) != 0 and len(r['rois']) != 0:
                    x1, y1 = r["rois"][0][0], r["rois"][0][1]
                    x2, y2 = r["rois"][0][2], r["rois"][0][3]
                    cropped_frame = crop_frame(x1, x2, y1, y2, aug_item_cp)

                    name = 'person_{}_{}.jpg'.format(person, counter)
                    result_name = os.path.join(target_person_path, name)
                    print('Saving {} element'.format(result_name))
                    cv2.imwrite(result_name, cropped_frame)

def save_frame(path, frame):
    """
    This function will save a frame in the given path
    :param path: Location target to save frame
    :param frame: Given frame
    :return: None
    """
    print('Saving file on {}'.format(path))
    cv2.imwrite(path, frame)
    # Image.fromarray(frame).save(path)


def slice_labels(df):
    """
    This function slices histograms data and labels, also transforms labels to categorical
    :param df: Histograms dataframe
    :return: x, y Histograms data and labels
    """
    from keras.utils import to_categorical
    df_top = df.shape[1] - 1
    x = df.iloc[:, :df_top]
    y = df.iloc[:, df_top]
    y = to_categorical(y)
    return x, y


def load_image_dataset(path, size, gray_scale):
    """
    This function will load given path dataset resizing each image
    :param path: Location dataset
    :param size: Target image size
    :return image, labels: Loaded images and corresponding labels in categorical notation
    """
    from keras.utils import to_categorical
    labels = []
    images = []
    for person in sorted(os.listdir(path)):
        person_path = os.path.join(path, person)
        if person.endswith('.csv'):
            continue
        for instance in os.listdir(person_path):
            if instance.endswith('.jpg') is False:
                continue
            instance_path = os.path.join(person_path, instance)
            image = cv2.imread(instance_path)
            image = cv2.resize(image, size)

            if gray_scale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = image.reshape(size[0], size[1], 1)
            images.append(image)
            labels.append(person)
    labels = to_categorical(labels)
    return np.array(images), np.array(labels)


def load_csv_data(path):
    """
    This function will load .csv data that was generated for each person with dataset generators
    :param path: Dataset location path.
    :return df: Extracted data dataframe
    """
    import pandas as pd
    df = pd.DataFrame()
    for person in os.listdir(path):
        person_path = os.path.join(path, person)
        if person.endswith('.csv'):
            continue
        for instance in os.listdir(person_path):
            if instance.endswith('.csv') is False:
                continue
            instance_path = os.path.join(person_path, instance)
            extracted = pd.read_csv(instance_path, header=None)
            df = df.append(extracted)
    return df


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


def generate_masks_datasets(parent_root, target_root):
    model = MaskRCNN()
    for person in os.listdir(parent_root):
        person_root = os.path.join(parent_root, person)
        target_person_root = os.path.join(target_root, person)
        print("Working on: ", person_root)
        counter = 0
        for instance in os.listdir(person_root):
            if instance.endswith('jpg') is False:
                continue
            counter += 1
            name = 'person_{}_{}.jpg'.format(person, counter)
            print('Processing {} ...'.format(name))
            source_path = os.path.join(person_root, instance)
            print("Opening ", source_path)
            image = cv2.imread(source_path)
            # image_cp = image.copy()
            r, _ = model.segment(image)
            if len(r['masks']) != 0 and len(r['rois']) != 0:
                mask = r["masks"][:, :, 0].astype('uint8') * 255
                mask_cp = mask.copy()
                x1, y1 = r["rois"][0][0], r["rois"][0][1]
                x2, y2 = r["rois"][0][2], r["rois"][0][3]
                cropped_frame = crop_frame(x1, x2, y1, y2, mask_cp)
                result_name = os.path.join(target_person_root, name)
                save_frame(result_name, cropped_frame)


def generate_dataset_with_lbp(parent_root, target_root, csv_path):
    """
    This function generate image dataset in function of given parent root.
    :param parent_root: Base dataset path
    :param target_root: Destination location to save LBP images
    :param csv_path: Target path to save features extracted of the entire dataset.
    :return: None
    """
    lbp_2 = LocalBinaryPatterns(3)
    model = MaskRCNN()
    header = ['area', 'perimeter', 'class']
    data = []

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
            # processed_imgs_path = os.path.join(person_root, 'processed')

            print("Opening ", source_path)
            image = cv2.imread(source_path)
            print('Copying  image ...')
            image_cp = image.copy()
            print('Detecting people ...')
            r, _ = model.segment(image)

            if len(r['masks']) != 0 and len(r['rois']) != 0:
                # for i in range(len(r["rois"])):
                mask = r["masks"][:, :, 0].astype('uint8')
                mask_cp = mask.copy()
                print('Calculating mask area and perimeter ...')
                area, perimeter = mask_area_perimeter(mask_cp)
                class_id = person

                masked_image = apply_mask(image_cp, mask)
                x1, y1 = r["rois"][0][0], r["rois"][0][1]
                x2, y2 = r["rois"][0][2], r["rois"][0][3]
                cropped_frame = crop_frame(x1, x2, y1, y2, masked_image)
                hist = lbp_2.describe(cropped_frame)
                hist = np.append(hist, person)

                histograms.append(hist)

                masked_image = lbp_2.lbp(cropped_frame)
                lbp_image = masked_image.astype('uint8')
                lbp_image = cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2RGB)
                result_name = os.path.join(target_person_root, name)

                data.append([area, perimeter, class_id])
                save_frame(result_name, lbp_image)

        csv_path_pr = target_person_root + '/hist_{}.csv'.format(person)
        write_csv(path=csv_path_pr, header=None, data=histograms)

    write_csv(csv_path, header, data)


def generate_masked_dataset(parent_root, target_root, csv_path):
    """
    This function will generate cropped images dataset with background removal
    :param parent_root: Base dataset path.
    :param target_root: Target location to save images.
    :param csv_path: Target location to save extracted feature dataset .csv (area, perimeter)
    :return: None
    """
    model = MaskRCNN()
    header = ['area', 'perimeter', 'class']
    data = []

    for person in os.listdir(parent_root):
        person_root = os.path.join(parent_root, person)
        target_person_root = os.path.join(target_root, person)
        print("Working on: ", person_root)
        counter = 0
        for instance in os.listdir(person_root):
            if instance.endswith('jpg') is False:
                continue
            counter += 1
            name = 'person_{}_{}.jpg'.format(person, counter)
            print('Processing {} ...'.format(name))
            source_path = os.path.join(person_root, instance)
            # processed_imgs_path = os.path.join(person_root, 'processed')

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

                result_name = os.path.join(target_person_root, name)

                data.append([area, perimeter, class_id])
                save_frame(result_name, cropped_frame)

    write_csv(csv_path, header, data)


def process_dataset_with_lbp(parent_root, target_root):
    """
    This function process image dataset and generate an lbp dataset based on a given base dataset.
    :param parent_root: Base dataset path.
    :param target_root: Target path to save generated dataset
    :return:
    """
    model = MaskRCNN()
    for person in os.listdir(parent_root):

        person_root = os.path.join(parent_root, person)
        target_person_root = os.path.join(target_root, person)
        print("Working on: ", person_root)
        counter = 0
        for instance in os.listdir(person_root):
            if instance == 'processed':
                continue
            counter += 1
            name = 'person_{}_{}.jpg'.format(person, counter)
            print('Processing {} ...'.format(name))
            source_path = os.path.join(person_root, instance)
            processed_imgs_path = os.path.join(person_root, 'processed')
            if os.path.exists(processed_imgs_path) is False:
                print("creating ", processed_imgs_path)
                os.mkdir(processed_imgs_path)

            print("Opening ", source_path)
            image = cv2.imread(source_path, cv2.COLOR_BGR2RGB)
            if os.path.exists(source_path) is False:
                break
            image_cp = image.copy()
            print(image_cp.shape)
            r, _ = model.segment(image)
            if len(r['masks']) != 0 and len(r['rois']) != 0:
                for i in range(len(r["rois"])):
                    mask = r["masks"][:, :, i].astype(int)
                    masked_image = apply_mask(image_cp, mask)
                    x1, y1 = r["rois"][i][0], r["rois"][i][1]
                    x2, y2 = r["rois"][i][2], r["rois"][i][3]
                    print('x1: {}, y1: {}, x2: {}, y2: {}'.format(x1, y1, x2
                                                                  , y2))
                    cropped_frame = crop_frame(x1, x2, y1, y2, masked_image)
                    cropped_frame = lbp(cropped_frame)
                    cropped_frame = cv2.cvtColor(cropped_frame.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)

                    result_name = os.path.join(target_person_root, name)
                    print("Saving on: {}".format(result_name))
                    cv2.imwrite(result_name, cropped_frame)

                    cropped_frame = np.expand_dims(cropped_frame, 0)
                    aug_iter = aug_generator.flow(cropped_frame)
                    aug_images = [next(aug_iter)[0].astype(np.uint8) for i in range(10)]
                    for k in range(len(aug_images)):
                        counter += 1

                        name = 'person_{}_{}.jpg'.format(person, counter)
                        result_name = os.path.join(target_person_root, name)
                        print('Saving {} element'.format(result_name))
                        cv2.imwrite(result_name, aug_images[k])


def generate_dpm_dataset(parent_root, target_root):
    model = MaskRCNN()
    """
    This function will generate a dataset filtered with LBP, each processed pedestrian capture will be cropped in three images. 
    :param parent_root: Base dataset path.
    :param target_root: Target path to save processed dataset.
    :return: 
    """
    for person in os.listdir(parent_root):
        person_root = os.path.join(parent_root, person)
        target_person_root = os.path.join(target_root, person)
        print('Working on: {} route'.format(person_root))
        counter = 0
        for instance in os.listdir(person_root):
            counter += 1
            name = 'person_{}_{}'.format(person, counter)
            print('Processing {} instance ...'.format(name))
            source_path = os.path.join(person_root, instance)
            instance_root = os.path.join(target_person_root, name)
            if os.path.exists(instance_root) is False:
                print("Creating {} directory".format(instance_root))
                os.mkdir(instance_root)
            print('Opening {} file'.format(source_path))
            image = cv2.imread(source_path, cv2.COLOR_BGR2RGB)
            if os.path.exists(source_path) is False:
                break
            image_cp = image.copy()
            r, _ = model.segment(image)
            if len(r['masks']) != 0 and len(r['rois']) != 0:
                for i in range(len(r["rois"])):
                    mask = r["masks"][:, :, i].astype(int)
                    masked_image = apply_mask(image_cp, mask)
                    h_crop, t_crop, l_crop = dpm(r["rois"][i], masked_image)
                    head_lbp = lbp(h_crop)
                    torso_lbp = lbp(t_crop)
                    legs_lbp = lbp(l_crop)

                    head_lbp = cv2.cvtColor(head_lbp.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)
                    torso_lbp = cv2.cvtColor(torso_lbp.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)
                    legs_lbp = cv2.cvtColor(legs_lbp.astype('uint8') * 255, cv2.COLOR_GRAY2RGB)

                    print('Saving {} dpm '.format(name))
                    head_path = os.path.join(instance_root, 'head.jpg')
                    torso_path = os.path.join(instance_root, 'torso.jpg')
                    legs_path = os.path.join(instance_root, 'legs.jpg')

                    print('Saving files: \n{}\n{}\n{}'.format(head_path, torso_path, legs_path))

                    cv2.imwrite(head_path, head_lbp)
                    cv2.imwrite(torso_path, torso_lbp)
                    cv2.imwrite(legs_path, legs_lbp)

                    print('Successfully created files')

