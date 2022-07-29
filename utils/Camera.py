import cv2
from utils.ReID import dpm, lbp, apply_mask, crop_frame
from utils.LocalBinaryPatterns import LocalBinaryPatterns
from processing import write_csv
import matplotlib.pyplot as plt
from utils.metrics.Timer import Timer
import winsound


class Camera:
    def __init__(self, src=0, id=None):
        self.id = id
        self.src = src
        self.cap = cv2.VideoCapture(self.src)


    def read_video(self, extract_masks, id_model, target_csv_path):
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f'[INFO]: {fps} fps')
        print(f"[INFO]: camera {self.id} initialized")
        logs = []
        logs_header = ['count', 'class', 'accuracy', 'detection time', 'identification time', 'processing time', 'pre-processing time']
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        lbp_2 = LocalBinaryPatterns(3)
        counter = 0
        detector_timer = Timer()
        identificator_timer = Timer()
        pre_processing_timer = Timer()
        processing_timer = Timer()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            print(f'[INFO] reading camera {self.id} frame')
            if ret:
            # (start) frame processing flow
                processing_timer.start()
                # image scale
                scale = 30
                width = int(frame.shape[1] * scale / 100)
                height = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (width, height))
                frame_cp = frame.copy()  # copied frame to avoid

                detector_timer.start()
                r, _ = extract_masks(frame)
                detector_timer.end()
                if len(r["rois"]) != 0 and len(r["masks"]) != 0:  # will process only the first detection

                # (start) pre-processing flow
                    pre_processing_timer.start()
                    mask = r["masks"][:, :, 0].astype(int) * 255
                    mask_cp = mask.copy()
                    masked_image = apply_mask(frame_cp, mask)
                    # bb_box coordinates
                    x1, y1 = r["rois"][0][0], r["rois"][0][1]
                    x2, y2 = r["rois"][0][2], r["rois"][0][3]

                    # cropping texture and image with bounding box coordinates
                    # mask_cp = crop_frame(x1, x2, y1, y2, mask_cp).astype('uint8') / 255  # normalization
                    mask_cp = crop_frame(x1, x2, y1, y2, mask_cp).astype('uint8')  # normalization
                    cropped_frame = crop_frame(x1, x2, y1, y2, masked_image).astype('uint8')
                    cropped_frame = cv2.resize(cropped_frame, (40, 40))
                    mask_cp = cv2.resize(mask_cp, (40, 40))
                    # color transformation
                    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                    mask_cp = cv2.cvtColor(mask_cp, cv2.COLOR_GRAY2RGB)

                    # image filtering using LBP method
                    # lbp_image = lbp_2.lbp(cropped_frame)   # normalization

                    # resizing images
                    # mask_cp = cv2.resize(mask_cp, (40, 40))
                    # cropped_frame = cv2.resize(cropped_frame, (40, 40))
                    # lbp_image = cv2.resize(lbp_image, (40, 40))

                    # reshaping - because it will specify channels number
                    # mask_cp = mask_cp.reshape(40, 40, 1)
                    # cropped_frame = cropped_frame.reshape(40, 40, 1)
                    # lbp_image = lbp_image.reshape(40, 40, 1)
                    pre_processing_timer.end()
                # (end) pre-processing flow

                # (start) identification flow
                    identificator_timer.start()
                    # predicted_name, accuracy = id_model.identify([[mask_cp], [lbp_image]])
                    # predicted_name, accuracy = id_model.identify([[mask_cp]])
                    # predicted_name, accuracy = id_model.identify([[lbp_image/255]])
                    # predicted_name, accuracy = id_model.identify([[cropped_frame]])  # color


                    # predicted_name, accuracy = id_model.identify([[mask_cp]])  # silhouette
                    predicted_name, accuracy = id_model.identify([[cropped_frame], [mask_cp]])  # combined
                    identificator_timer.end()
                # (end) identification flow

                    # labeling instance detection on frame
                    bbox_height = abs(x1 - x2)
                    cv2.putText(frame_cp, f'{predicted_name}', (y1, x1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=bbox_height/300, color=(0, 255, 0),thickness=1)  # class name
                    cv2.putText(frame_cp, 'acc: {:.2f}'.format(accuracy), (y2, x1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=bbox_height/325, color=(120, 255, 0),thickness=1)  # identification accuracy
                    frame_cp = cv2.rectangle(frame_cp, (y1, x1), (y2, x2), (255, 0, 0), 1)  # b_box drawing
                    processing_timer.end()
                # (end) frame processing flow
                    logs.append([counter, predicted_name, accuracy, detector_timer.calculate_time(), identificator_timer.calculate_time(), processing_timer.calculate_time(), pre_processing_timer.calculate_time()])  # appending logs for identification
                    counter += 1
                yield frame_cp

            else:
                break
        winsound.Beep(440, 550)
        print(f'Will save log file {self.id}')
        write_csv(target_csv_path, logs_header, logs)  # writing logs
        self.cap.release()
