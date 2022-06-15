import cv2
from utils.ReID import dpm, lbp, apply_mask
from matplotlib import pyplot as plt

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks):
        self.cap.set(cv2.CAP_PROP_FPS, 25)
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
                    for i in range(len(r["rois"])):
                        mask = r["masks"][:, :, i].astype(int)
                        temp_frame = apply_mask(frame_cp, mask)
                        plt.imshow(temp_frame)
                        plt.show()
                        h_crop, t_crop, l_crop = dpm(r["rois"][i], temp_frame)

                yield frame_cp

            else:
                break
        self.cap.release()

