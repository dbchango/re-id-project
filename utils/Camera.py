import cv2
from utils.ReID import dpm, lbp, apply_mask, crop_frame
from utils.LocalBinaryPatterns import LocalBinaryPatterns

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks, id_model):
        self.cap.set(cv2.CAP_PROP_FPS, 25)
        lbp_2 = LocalBinaryPatterns(3)
        while self.cap.isOpened():
            ret, frame = self.cap.read()

            if ret:
                scale = 20
                width = int(frame.shape[1] * scale / 100)
                height = int(frame.shape[0] * scale / 100)
                frame = cv2.resize(frame, (width, height))
                frame_cp = frame.copy()
                r, _ = extract_masks(frame)
                if len(r["rois"]) != 0 and len(r["masks"]) != 0:
                    # for i in range(len(r["rois"])):
                    mask = r["masks"][:, :, 0].astype(int)
                    masked_image = apply_mask(frame_cp, mask)
                    x1, y1 = r["rois"][0][0], r["rois"][0][1]
                    x2, y2 = r["rois"][0][2], r["rois"][0][3]
                    cropped_frame = crop_frame(x1, x2, y1, y2, masked_image)
                    lbp_image = lbp_2.lbp(cropped_frame)
                    lbp_image = cv2.resize(lbp_image, (40, 40))
                    lbp_image = lbp_image.astype('uint8')
                    lbp_image = lbp_image.reshape(1, 40, 40, 1)
                    prediction_name = id_model.identify(lbp_image)

                yield frame_cp

            else:
                break
        self.cap.release()
