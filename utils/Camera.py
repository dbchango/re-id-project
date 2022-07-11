import cv2
from utils.ReID import dpm, lbp, apply_mask, crop_frame
from utils.LocalBinaryPatterns import LocalBinaryPatterns
from processing import write_csv
import numpy as np



class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks, id_model, target_csv_path):
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        lbp_2 = LocalBinaryPatterns(3)
        detections = []

        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                scale = 30
                width = int(frame.shape[1] * scale / 100)
                height = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (width, height))
                frame_cp = frame.copy()
                r, _ = extract_masks(frame)
                if len(r["rois"]) != 0 and len(r["masks"]) != 0:
                    # for i in range(len(r["rois"])):
                    mask = r["masks"][:, :, 0].astype(int) * 255
                    mask_cp = mask.copy()
                    masked_image = apply_mask(frame_cp, mask)
                    x1, y1 = r["rois"][0][0], r["rois"][0][1]
                    x2, y2 = r["rois"][0][2], r["rois"][0][3]
                    mask_cp = crop_frame(x1,x2, y1,y2, mask_cp).astype('uint8')

                    cropped_frame = crop_frame(x1, x2, y1, y2, masked_image).astype('uint8')
                    lbp_image = lbp_2.lbp(cropped_frame) / 255

                    mask_cp = cv2.resize(mask_cp, (40, 40))

                    lbp_image = cv2.resize(lbp_image, (40, 40))

                    mask_cp = mask_cp.reshape(40, 40, 1)
                    lbp_image = lbp_image.reshape(40, 40, 1)

                    predicted_name, accuracy = id_model.identify([[lbp_image], [mask_cp]])
                    detections.append([predicted_name, accuracy])
                    bbox_height = abs(x1 - x2)
                    cv2.putText(frame_cp, f'{predicted_name}', (y1, x1), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=bbox_height/300, color=(0, 255, 0),thickness=1)
                    cv2.putText(frame_cp, 'acc: {:.2f}'.format(accuracy), (y2, x1+10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=bbox_height/325, color=(120, 255, 0),thickness=1)
                    frame_cp = cv2.rectangle(frame_cp, (y1, x1), (y2, x2), (255, 0, 0), 1)

                yield frame_cp

            else:
                break
        write_csv(target_csv_path, None, detections)
        self.cap.release()
