import cv2
import threading


class video_reader:

    def __init__(self, src=0):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        if self.cap.isOpened() is False:
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        self.grabbed, self.frame = self.cap.read()
        if self.grabbed is False:
            print("[Exiting]: No more frames to read.")
            exit(0)
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, key, value):
        self.cap.set(key, value)

    def start(self):
        if self.started:
            print("[Warning] Asynchronous video capturing is already started.")
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
        self.thread.join()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
