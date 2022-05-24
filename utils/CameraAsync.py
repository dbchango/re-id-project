# file: videoasync.py
import os
import threading
import cv2

from utils.Model import MaskRCNN
from utils.MaskRCNNPedestrian import MaskRCNNPedestrian
from Mask_RCNN.mrcnn import visualize
import utils.class_names
from utils.utility import *

class VideoCaptureAsync:
    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, key, value):
        self.cap.set(key, value)

    def start(self):
        if self.started:
            print('[Warning] Asynchronous video capturing is already started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        # self.cap.release()
        # cv2.destroyAllWindows()
        self.thread.join()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()

lock = threading.Lock()

def streamVideo(src=0, model = None, batch_size = 3):
    global lock
    print(src)
    cap = VideoCaptureAsync(src=src
                            ).start()

    frames = []
    frame_count = 0
    while (cap.cap.isOpened()):

        retrieved, frame = cap.read()
        yield frame
        # frame_count += 1
        # frames.append(frame)
        # print('frame_count :{0}'.format(frame_count))
        # if retrieved and len(frames) == batch_size:
        #     with lock:
        #
        #         # do tracking
        #         results = model.detect(frames, verbose=0)
        #         print('Predicted')
        #         for i, item in enumerate(zip(frames, results)):
        #             frame = item[0]
        #             r = item[1]
        #             frame = get_masked_image(frame, r)
        #             yield frame, r
        #         frames = []
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.stop()
            cv2.destroyAllWindows()
            break
    cap.stop()
    cv2.destroyAllWindows()
