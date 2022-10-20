import os
from utils.ReID import lbp
from utils.processing import save_frame, crete_dir
import logging
import cv2

# route path
base_path = ''
outout_path = ''


def filter_image(image, saving_path):
    logging.info(f'Filtering image')
    lbp_image = lbp(image)
    logging.info(f'Saving image image')
    save_frame(saving_path, lbp_image)
    return filtered_image


if __name__ == '__main__':
    for person in os.listdir(base_path):
        person_route_path = os.path.join(base_path, person)
        person_output_route_path = os.path.join(outout_path, person)
        crete_dir(person_output_route_path)
        for video in os.listdir(person_route_path):
            video_route_path = os.path.join(person_route_path, video)
            video_output_path = os.path.join(person_output_route_path, video)
            crete_dir(video_output_path)
            for image in os.listdir(video_route_path):
                image_path = os.path.join(video_route_path, image)
                image_output_path = os.path.join(video_output_path, image)
                image = cv2.imread(image_path)
                filtered_image = filter_image(image, image_output_path)
                cv2.imshow('LBP', filtered_image)
