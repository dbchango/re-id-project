import multiprocessing
import threading
import concurrent.futures
import cv2
import time
from utils.ReID import extract_masks


class CameraThreading(threading.Thread):
    def __init__(self, src, camID, op, lock):
        threading.Thread.__init__(self)
        self.op = op
        self.lock = lock
        self.src = src
        self.camID = camID

    def run(self):
        executor = concurrent.futures.ThreadPoolExecutor(max_workers=8)
        if self.op == "1":
            executor.submit(camPreview, self.src, self.camID, self.lock)
        else:
            executor.submit(camPreview, self.src, self.camID, self.lock)



lock = multiprocessing.Lock()


def camPreview(src, camID, lock):
    cv2.namedWindow(src)
    cam = cv2.VideoCapture(src)
    while True:
        time.sleep(0.03)
        lock.acquire()
        ret, frame = cam.read()
        r, output = extract_masks(frame)
        lock.release()
        cv2.imshow(src, output)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()


thread1 = CameraThreading(src='Datasets/videos/chaplin.mp4', op=0, camID="1", lock=lock)

thread1.start()
thread1.join()
