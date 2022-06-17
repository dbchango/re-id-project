import os
from utils.ReID import extract_masks, dpm, lbp, apply_mask, crop_frame
import cv2
from keras.preprocessing.image import ImageDataGenerator

aug_generator = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, zoom_range=0.1, horizontal_flip=True)


def process_dataset_with_lbp(parent_root, target_root):

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
            r, _ = extract_masks(image)
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
            r, _ = extract_masks(image)
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

