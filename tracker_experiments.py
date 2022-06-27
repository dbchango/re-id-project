import cv2
import sys
import tensorflow as tf
import numpy as np

def check_gpus():
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':

    # tracker = cv2.TrackerCSRT_create()
    # video = cv2.VideoCapture('Datasets/chaplin.mp4')
    # if not video.isOpened():
    #     print("Could not open video")
    #     sys.exit()
    # ok, frame = video.read()
    # if not ok:
    #     print('Cannot read video file')
    #     sys.exit()
    # bbox = (287, 23, 86, 320)
    #
    # bbox = cv2.selectROI(frame, False)
    #
    # ok = tracker.init(frame, bbox)
    #
    # while True:
    #     ok, frame = video.read()
    #     w = int(frame.shape[0] / 3)
    #     h = int(frame.shape[1] / 3)
    #     # frame = cv2.resize(frame, (h, w))
    #     if not ok:
    #         break
    #     # with tf.device('/GPU:0'):
    #     ok, bbox = tracker.update(frame)
    #
    #     if ok:
    #         (x, y, w, h) = [int(v) for v in bbox]
    #         cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2, 1)
    #     else:
    #         cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
    #
    #     cv2.imshow("tracking", frame)
    #
    #     k = cv2.waitKey(1) & 0xff
    #     if k == 27: break
    # video.release()
    # cv2.destroyAllWindows()

    cap = cv2.VideoCapture('Datasets/chaplin.mp4')
    ret, frame = cap.read()
    x, y, w, h = cv2.selectROI(frame)
    track_window = (x, y, w, h)
    roi = frame[y:y + h, x:x + w]
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
    roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
    term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
    while True:
        ret, frame = cap.read()
        if ret == True:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, track_window = cv2.CamShift(dst, track_window, term_crit)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            print(pts)
            img2 = cv2.polylines(frame, [pts], True, 255, 2)
            cv2.imshow('img2', img2)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        else:
            break
    cv2.destroyAllWindows()





