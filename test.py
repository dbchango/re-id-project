import cv2
from utils.SequenceReader import read_sequence
from utils.MaskRCNN import MaskRCNN
from utils.IdentificationModel import IdentificationModel
import os
from processing import save_frame
seg = MaskRCNN()

if __name__ == '__main__':
    width = 0
    height = 0
    directory = 'Datasets/experiments/sequence/seq_1'

    # id_model = IdentificationModel()  # double input model
    # id_model = IdentificationModel(model_path='models/own/experiments/lbp_model/experiment_2/model_4.h5')  # double input model
    id_model = IdentificationModel(model_path='models/own/experiments/sillhouette_model/experiment_2/model_3.h5')  # silhouette input model


    output_path = "outputs/sequences/masked"

    # directory_cam1 = 'Datasets/Propio/Video/pasillo/cam_1/pasillo_001'
    # directory_cam2 = 'Datasets/Propio/Video/pasillo/cam_2/pasillo_001'
    counter = 0
    # for i, j in zip(Camera("Datasets/videos/chaplin.mp4").read_video(extract_masks), Camera("Datasets/videos/VIRAT_S_000002.mp4").read_video(extract_masks)):
    for o in read_sequence(directory, seg.segment, id_model, 'outputs/sequences/masked/logs.csv'):
        img_path = os.path.join(output_path, str(counter)+".jpg")

        save_frame(img_path, o)
        counter += 1
