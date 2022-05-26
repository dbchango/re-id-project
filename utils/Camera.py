import cv2

class Camera:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)

    def read_video(self, extract_masks):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                #r, output = extract_masks(frame)
                yield frame
                # cv2.imshow("Output", frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break
        self.cap.release()
        cv2.destroyAllWindows()