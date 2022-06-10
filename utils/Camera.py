import cv2
from multiprocessing import Pool


class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                with Pool(6) as p:
                    r, output = p.map(extract_masks, frame)
                # r, output = extract_masks(frame)
                # yield r, output
                cv2.imshow("Output", output)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()
