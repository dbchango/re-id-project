import cv2

from utils.Camera import Camera
from utils.MaskRCNN import MaskRCNN
from utils.IdentificationModel import IdentificationModel
from utils.metrics.measuring import tracing_start, tracing_mem
from utils.metrics.Timer import Timer

if __name__ == '__main__':
    timer = Timer()
    timer.start()
    tracing_start()
    source_1 = 'Datasets/Propio/cameras/pasillo/cam_1/soft/Pasillo_001_sf.mp4'
    source_2 = 'Datasets/Propio/cameras/pasillo/cam_2/soft/Pasillo_001_sf.mp4'

    output_path = "outputs/texture-silhouette/texture/video_1.avi"

    true_class_name = 'Luis'

    model = MaskRCNN()

    # id_model = IdentificationModel(model_path='models/texture-silhouette/combined/model_3.h5')  # double input model
    id_model = IdentificationModel(model_path='models/own/experiments/lbp_model/experiment_2/model_4.h5')  # double input model
    # id_model = IdentificationModel(model_path='models/own/experiments/sillhouette_model/experiment_2/model_3.h5')  # silhouette input model

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = None
    (h, w) = (None, None)

    # for o1, o2 in zip(Camera(src=source_1, id=0).read_video(model.segment, id_model, 'outputs/run_1/cam1_detections.csv'), Camera(src=source_2, id=1).read_video(model.segment, id_model, 'outputs/run_1/cam2_detections.csv')):
    for o in Camera(src=source_1, id=1).read_video(model.segment, id_model, 'outputs/texture-silhouette/texture/cam1_detections.csv'):
        # o = np.concatenate((o1, o2), axis=1)
        if writer is None:
            (h, w) = o.shape[:2]
            writer = cv2.VideoWriter(output_path, fourcc, 25.0, (w, h))
        vidout = cv2.resize(o, (w, h))
        writer.write(vidout)
        cv2.imshow("Output", o)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    writer.release()
    cv2.destroyAllWindows()
    timer.end()
    print(f'Processing time": {timer.calculate_time()}')
    tracing_mem()
    exit()
