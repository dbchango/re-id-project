import cv2
#from utils.SequenceReader import read_sequence
from utils.SequenceReader import read_sequence_color
from utils.MaskRCNN import MaskRCNN
from utils.IdentificationModel import IdentificationModel
import os
from processing import save_frame
seg = MaskRCNN()

if __name__ == '__main__':
    width = 0
    height = 0
    model_name = 'combined'
    directory = 'Datasets/experiments/sequence/seq_1'
    output_path = 'outputs/sequences/color_silhouette/own_models/combined'

    # # first work models
    # if model_name == 'combined':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/own/combined/model_3.h5')  # double input model
    # if model_name == 'silhouette':
    #     id_model = IdentificationModel(
    #
    #         model_path='models/texture-silhouette/own/silhouette/model_1.h5')  # double input model
    # if model_name == 'textures':
    #     id_model = IdentificationModel(
    #         model_pat
    #         h='models/texture-silhouette/own/textures/model_1.h5')  # silhouette input model
    # first work
    # if model_name == 'combined':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/market-1501/combined/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    # if model_name == 'silhouette':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/market-1501/silhouette/model_1.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    # if model_name == 'textures':
    #     id_model = IdentificationModel(
    #         model_path='models/texture-silhouette/market-1501/texture/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # silhouette input model

    # second work models
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/color/experiment_0/model_3.h5')  # masked image model
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/silueta_2/experiment_0/model_3.h5')  # mask model
    # id_model = IdentificationModel(model_path='models/own_models/experiments/Pamela/color_silueta/experiment_0/model_3.h5')  # double branch model
    # Using own
    if model_name == 'combined':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/color_silueta/experiment_0/model_2.h5', classes_names_path='Datasets/classes_names.csv')  # double input model
    if model_name == 'silhouette':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/silueta_2/experiment_0/model_1.h5', classes_names_path='Datasets/classes_names.csv')  # double input model
    if model_name == 'color':
        id_model = IdentificationModel(
            model_path='models/color-silhouette/own/color/experiment_0/model_1.h5', classes_names_path='Datasets/classes_names.csv')  # silhouette input model
    # Using Market-1501
    # if model_name == 'combined':
    #     id_model = IdentificationModel(
    #         model_path='models/color-silhouette/market-1501/combined/experiment_0/model_1.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    # if model_name == 'silhouette':
    #     id_model = IdentificationModel(
    #         model_path='models/color-silhouette/market-1501/silhouette/experiment_0/model_1.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # double input model
    # if model_name == 'color':
    #     id_model = IdentificationModel(
    #         model_path='models/color-silhouette/market-1501/color/experiment_0/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')  # silhouette input model

    # Using Market-1501
    # id_model = IdentificationModel(model_path='models/market-1501/texture/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')
    # id_model = IdentificationModel(model_path='models/market-1501/combined/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')
    # id_model = IdentificationModel(model_path='models/market-1501/silhouette/model_3.h5', classes_names_path='Datasets/market_1501_class_names.csv')



    counter = 0

    for o in read_sequence_color(directory, seg.segment, id_model, os.path.join(output_path, 'logs.csv')):
        img_path = os.path.join(output_path, str(counter)+".jpg")

        save_frame(img_path, o)
        counter += 1
