import cv2
import threading


from utils.video_reader import video_reader

lock = threading.Lock()
class Tracker:
    def __init__(self, src=None):
        self.src = src

        if src is None:
            self.cap = video_reader().start()
        else:
            self.cap = video_reader(src=src).start()

    def streamVideo(self):
        global lock
        while (True):
            retrieved, frame = self.cap.read()
            if retrieved:
                with lock:
                    (flag, encodedImage) = cv2.imencode(".jpg", frame)
                    if not flag:
                        continue
                    cv2.imshow('Video reading', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.cap.stop()
                cv2.destroyAllWindows()
                break
        self.cap.stop()
        cv2.destroyAllWindows()