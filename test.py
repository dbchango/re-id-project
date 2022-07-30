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
    model_name = 'combined'
    directory = 'Datasets/experiments/sequence/seq_2'
    output_path = 'outputs/sequences/texture_silhouette/market-1501/combined'

    # # first work models
    # if model_name == 'combined':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/own/combined/model_3.h5')  # double input model
    # if model_name == 'silhouette':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/own/silhouette/model_1.h5')  # double input model
    # if model_name == 'textures':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/own/textures/model_1.h5')  # silhouette input model

    if model_name == 'combined':
        id_model = IdentificationModel(
            model_path='models/texture-silhouette/market-1501/combined/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    if model_name == 'silhouette':
        id_model = IdentificationModel(
            model_path='models/texture-silhouette/market-1501/silhouette/model_1.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    if model_name == 'textures':
        id_model = IdentificationModel(
            model_path='models/texture-silhouette/market-1501/texture/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # silhouette input model

    # second work models
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/color/experiment_0/model_3.h5')  # masked image model
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/silueta_2/experiment_0/model_3.h5')  # mask model
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/color_silueta/experiment_0/model_3.h5')  # double branch model

    # Using Market-1501
    # id_model = IdentificationModel(model_path='models/market-1501/texture/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')
    # id_model = IdentificationModel(model_path='models/market-1501/combined/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')
    # id_model = IdentificationModel(model_path='models/market-1501/silhouette/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')



    counter = 0

    for o in read_sequence(directory, seg.segment, id_model, os.path.join(output_path, 'logs.csv')):
        img_path = os.path.join(output_path, str(counter)+".jpg")

        save_frame(img_path, o)
        counter += 1
