import os.path

import cv2

from utils.Camera import Camera
from utils.MaskRCNN import MaskRCNN
from utils.IdentificationModel import IdentificationModel
from utils.metrics.measuring import tracing_start, tracing_mem
from utils.metrics.Timer import Timer

def write_txt(path, lines):
    with open(path, 'w') as f:
        for line in lines:
            f.write(line)

if __name__ == '__main__':

    n = 2
    run = 'run_1_5'
    model_name = 'silhouette'
    output_folder_path = 'outputs/video/color-silhouette/silhouette/'
    base_path = os.path.join(output_folder_path, run)


    timer = Timer()
    timer.start()
    tracing_start()
    source_1 = 'Datasets/experiments/videos/pasillo/cam_1/soft/Pasillo_001_sf.mp4'
    source_2 = 'Datasets/experiments/videos/pasillo/cam_1/soft/Pasillo_001_sf.mp4'

    output_video_path = os.path.join(base_path, f'video_{n}.avi')

    model = MaskRCNN()

    # if model_name == 'combined':
    #     id_model = IdentificationModel(model_path='models/texture-silhouette/own/combined/model_3.h5')  # double input model
    # if model_name == 'silhouette':
    #     id_model = IdentificationModel(model_path='models/texture-silhouette/own/silhouette/model_1.h5')  # double input model
    # if model_name == 'texture':
    #     id_model = IdentificationModel(model_path='models/texture-silhouette/own/textures/model_1.h5')  # silhouette input model

    if model_name == 'combined':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/color_silueta/experiment_0/model_2.h5')  # double input model
    if model_name == 'silhouette':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/silueta_2/experiment_0/model_1.h5')  # double input model
    if model_name == 'color':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/color/experiment_0/model_1.h5')  # silhouette input model#3

    # second work
    # id_model = IdentificationModel(model_path='models/own/experiments/Pamela/own/color/experiment_0/model_3.h5')  # masked image model
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/silueta_2/experiment_0/model_3.h5')  # mask model
    # id_model = IdentificationModel(model_path='models/own/experiments/Pamela/color_silueta/experiment_0/model_3.h5')  # double branch model

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = None
    (h, w) = (None, None)

    # for o1, o2 in zip(Camera(src=source_1, id=0).read_video(model.segment, id_model, 'outputs/run_1/cam1_detections.csv'), Camera(src=source_2, id=1).read_video(model.segment, id_model, 'outputs/run_1/cam2_detections.csv')):
    for o in Camera(src=source_1 if n == 1 else source_2, id=1).read_video(model.segment, id_model, os.path.join(base_path, f'cam{n}_detections.csv')):
        # o = np.concatenate((o1, o2), axis=1)
        if writer is None:
            (h, w) = o.shape[:2]
            writer = cv2.VideoWriter(output_video_path, fourcc, 25.0, (w, h))
        vidout = cv2.resize(o, (w, h))
        writer.write(vidout)
        cv2.imshow("Output", o)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()
    timer.end()
    print(f'Processing time": {timer.calculate_time()}')
    peak = tracing_mem()
    write_txt(os.path.join(base_path, f'results_{n}.txt'), [f'Processing time: {timer.calculate_time()}', f'Peak size in MB: {peak}'])
