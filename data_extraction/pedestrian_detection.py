import os
import cv2
from utils.MaskRCNN import MaskRCNN
from utils.processing import save_frame, crete_dir
import logging
from utils.ReID import apply_mask, crop_frame

output_path = '../Datasets/Base de datos - extracted bboxes'
binary_masks_path = '../Datasets/Base de datos - extracted binary masks'
model = MaskRCNN()
log = logging.getLogger(__name__)

def read_video(source, output_person_path, output_masks):
    cap = cv2.VideoCapture(source)
    routes = str.split(source, '\\')
    file_name = routes[len(routes) - 1]
    file_name = file_name.split('.')
    file_name = file_name[0]
    counter = 0
    folder_container_path = os.path.join(output_person_path, file_name)
    folder_container_masks_path = os.path.join(output_masks, file_name)
    crete_dir(folder_container_path)
    crete_dir(folder_container_masks_path)
    while cap.isOpened():
        retrieved, frame = cap.read()
        if frame is None:
            break

        scale = 30
        width = int(frame.shape[1] * scale / 100)
        height = int(frame.shape[0] * scale / 100)
        frame = cv2.resize(frame, (width, height))
        frame2 = frame.copy()
        r, _ = model.segment(frame)
        if len(r["rois"]) != 0 and len(r["masks"]) != 0:  # will process only the first detection
            frame_name = f'{file_name}_{counter}.jpg'
            mask = r["masks"][:, :, 0].astype(int)
            masked_image = apply_mask(frame2, mask)
            x1, y1 = r["rois"][0][0], r["rois"][0][1]
            x2, y2 = r["rois"][0][2], r["rois"][0][3]
            masked_image = crop_frame(x1, x2, y1, y2, masked_image).astype('uint8')
            binary_mask = crop_frame(x1, x2, y1, y2, mask).astype('uint8') * 255
            # binary_mask = binary_mask.reshape(40, 40, 1)
            cv2.imshow('RoI', masked_image)
            cv2.imshow('Binary Mask', binary_mask)
            saving_path = os.path.join(folder_container_path, frame_name)
            saving_mask_path = os.path.join(folder_container_masks_path, frame_name)
            save_frame(saving_path, masked_image)
            save_frame(saving_mask_path, binary_mask)
            counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"Close video: {file_name}")


if __name__ == '__main__':

    folder_route = '../Datasets/Base de datos de Videos'
    for person_route in os.listdir(folder_route):
        person_folder_route = os.path.join(folder_route, person_route)
        output_person_route = os.path.join(output_path, person_route)
        output_masks_route = os.path.join(binary_masks_path, person_route)
        crete_dir(output_person_route)
        crete_dir(output_masks_route)
        for video in os.listdir(person_folder_route):
            resource_path = os.path.join(person_folder_route, video)
            logging.info(f'Vide named: {resource_path} will be processed.')
            read_video(resource_path, output_person_route, output_masks_route)
