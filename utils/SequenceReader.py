import os
import cv2
from utils.ReID import crop_frame, apply_mask
from utils.LocalBinaryPatterns import LocalBinaryPatterns
from utils.metrics.Timer import Timer
from processing import write_csv
import winsound


def read_sequence(path, extract_masks, id_model, target_csv_path):
    lbp_2 = LocalBinaryPatterns(3)
    logs = []
    logs_header = ['count', 'class', 'accuracy', 'detection time', 'identification time', 'processing time',
                   'pre-processing time']
    counter = 0
    detector_timer = Timer()
    identificator_timer = Timer()
    pre_processing_timer = Timer()
    processing_timer = Timer()

    for image_path in os.listdir(path):
        if image_path.endswith('.jpg') is False:
            continue
        processing_timer.start()  # (start) - precessing time
        root = os.path.join(path, image_path)
        frame = cv2.imread(root)
        scale = 200
        width = int(frame.shape[1] * scale / 100)
        height = int(frame.shape[0] * scale / 100)
        frame = cv2.resize(frame, (width, height))

        frame_cp = frame.copy()
        detector_timer.start()
        r, output = extract_masks(frame)
        detector_timer.end()
        if len(r["rois"]) != 0 and len(r["masks"]) != 0:

            # temp_frame = frame_cp.copy()
            pre_processing_timer.start()  # (start) - pre-processing time
            mask = r["masks"][:, :, 0].astype('uint8') * 255
            mask_cp = mask.copy()
            masked_image = apply_mask(frame_cp, mask)
            x1, y1 = r["rois"][0][0], r["rois"][0][1]
            x2, y2 = r["rois"][0][2], r["rois"][0][3]

            # cropping texture and image with bounding box coordinates
            mask_cp = crop_frame(x1, x2, y1, y2, mask_cp).astype('uint8')
            cropped_frame = crop_frame(x1, x2, y1, y2, masked_image).astype('uint8')

            # image filtering using LBP method
            # lbp_image = lbp_2.lbp(cropped_frame)

            # resizing images
            mask_cp = cv2.resize(mask_cp, (40, 40))
            # lbp_image = cv2.resize(lbp_image, (40, 40))
            cropped_frame = cv2.resize(cropped_frame, (40, 40))

            # reshaping - because it will specify channels number
            # mask_cp = mask_cp.reshape(40, 40, 1)
            # lbp_image = lbp_image.reshape(40, 40, 1)
            mask_cp = cv2.cvtColor(mask_cp, cv2.COLOR_GRAY2RGB)
            cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
            pre_processing_timer.end()  # (end) - pre-processing time

            identificator_timer.start()  # (start) - identification time
            # predicted_name, accuracy = id_model.identify([[lbp_image], [mask_cp]])
            predicted_name, accuracy = id_model.identify([[cropped_frame], [mask_cp]]) # color - silhouette
            # predicted_name, accuracy = id_model.identify([[mask_cp]])
            # predicted_name, accuracy = id_model.identify([[cropped_frame]]) # color
            # predicted_name, accuracy = id_model.identify([[cropped_frame]]) # color
            # predicted_name, accuracy = id_model.identify([[lbp_image/255]])
            identificator_timer.end()  # (end) - identification time

            # labeling instance detection on frame
            bbox_height = abs(x1 - x2)
            cv2.putText(frame_cp, f'{predicted_name}', (y1, x1), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=bbox_height / 300, color=(0, 255, 0), thickness=1)  # class name
            cv2.putText(frame_cp, 'acc: {:.2f}'.format(accuracy), (y2, x1 + 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=bbox_height / 325, color=(120, 255, 0), thickness=1)  # identification accuracy
            frame_cp = cv2.rectangle(frame_cp, (y1, x1), (y2, x2), (255, 0, 0), 1)

            processing_timer.end()  # (end) - processing time
            logs.append([counter, predicted_name, accuracy, detector_timer.calculate_time(),
                         identificator_timer.calculate_time(), processing_timer.calculate_time(),
                         pre_processing_timer.calculate_time()])  # appending logs for identification
            counter += 1
        yield frame_cp
    winsound.Beep(440, 550)
    print(f'Will save log file')
    write_csv(target_csv_path, logs_header, logs)  # writing logs