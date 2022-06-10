import multiprocessing_testing
import timeit
import typing
from multiprocessing_testing import Lock
import cv2
import numpy as np
import pafy

urls = [
    "https://www.youtube.com/watch?v=tT0ob3cHPmE",
    "https://www.youtube.com/watch?v=XmjKODQYYfg",
    "https://www.youtube.com/watch?v=E2zrqzvtWio",

    "https://www.youtube.com/watch?v=6cQLNXELdtw",
    "https://www.youtube.com/watch?v=s_rmsH0wQ3g",
    "https://www.youtube.com/watch?v=QfhpNe6pOqU",

    "https://www.youtube.com/watch?v=C_9x0P0ebNc",
    "https://www.youtube.com/watch?v=Ger6gU_9v9A",
    "https://www.youtube.com/watch?v=39dZ5WhDlLE"
]

width = np.math.ceil(np.sqrt(len(urls)))
dim = 1920, 1080
streams = []

def main():
    global bestStreams
    streams = [pafy.new(url).getbest() for url in urls]
    print(streams)

    cv2.waitKey(0)
    videos = [cv2.VideoCapture() for stream in streams]
    bestURLs = []

    [bestURLs.append(best.url) for best in streams]

    print(bestURLs)
    cv2.waitKey(0)

    cv2.namedWindow('Video', cv2.WINDOW_FREERATIO)
    cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    LOCK = Lock()
    proc, pipes = get_framesULJ(bestURLs, LOCK)
    print("PROC, PIPES", proc, pipes)
    frames = []
    numStreams = len(streams)
    while True:
        start_time = timeit.default_timer()

        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()
        frames = [x.recv() for x in pipes]
        lf = len(frames)
        print("LEN(FRAMES)=", lf)
        dst = merge_frames(frames)
        print(timeit.default_timer() - start_time)
        start_time = timeit.default_timer()

        try:
            cv2.imshow('Video', dst)
        except: print("Skip")

        if cv2.waitKey(20) & 0xFF == ord('e'):
            break
        print(timeit.default_timer() - start_time)
        continue

    for proc in jobs:
        proc.join()
    cv2.destroyAllWindows()


def get_framesULJ(videosURL, L):
    print("get_framesULJ", videosURL)
    jobs = []
    pipe_list = []
    for videoURL in videosURL:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get)

def get_frame2L(videoURL, send_end, L):
    v = cv2.VideoCapture()
    v.open(videoURL)
    print("get_frame_2", videoURL, v, send_end)
    while True:
        ret, frame = v.read()
        if ret: send_end.send(frame)
        else: print("NOT READ!"); break


def get_framesUL(videosURL, L):
    jobs = []
    pipe_list = []
    print("VIDEOS: ", videosURL)
    for videoURL in videosURL:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get_frame2L, args=(videoURL, send_end, L))
        print("P = ", p)
        jobs.append(p)
        print("JOBS, len ", jobs, len(jobs))
        pipe_list.append(recv_end)
        print("pipe_list", pipe_list)
        p.start()

    for proc in jobs:
        proc.join()

    frames = [x.recv() for x in pipe_list]
    return frames

def get_frames(videos, L):
    jobs = []
    pipe_list = []
    print("VIDEOS: ", videos)
    for video in videos:
        recv_end, send_end = multiprocessing.Pipe(False)
        print(recv_end, send_end)
        p = multiprocessing.Process(target=get_frame, args=(video, send_end, L))
        print("P = ", p)
        jobs.append(p)
        print("JOBS, len", jobs, len(jobs))
        pipe_list.append(recv_end)
        print("pipe_list", pipe_list)
        p.start()

    for proc in jobs:
        proc.join()

    frames = [x.recv() for x in pipe_list]
    return frames

def get_frames(video, send_end, L):
    L.acquire()
    print("get_frame", video, send_end)
    send_end.send(video.read()[1])
    L.release()

def get_frame2(videoURL, send_end):
    v = video.open(videoURL)
    while True:
        ret, frame = v.read()
        if ret: send_end.send(frame)
        else: break

def merge_frames(frames: typing.List[np.ndarray]):
    width = np.math.ceil(np.sqrt(len(frames)))
    rows = []
    for row in range(width):
        i1, i2 = width * row, width * row + width
        rows.append(np.hstack(frames[i1: i2]))

        return np.vstack(rows)

if __name__ == '__main__':
    main()
