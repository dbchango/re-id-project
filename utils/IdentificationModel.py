import tensorflow as tf
import pandas as pd
import numpy as np


class IdentificationModel:
    def __init__(self, model_path="models/own/experiments/double_branch_model/experiment_1/sample_model3.h5", classes_names_path="Datasets/classes_names.csv"):
        self.model = tf.keras.models.load_model(model_path)
        self.classes_names = pd.read_csv(classes_names_path, header=None)

    def identify(self, x):

        prediction = self.model.predict(x, use_multiprocessing=True)
        class_name, accuracy = self.get_class_results(prediction)
        return class_name, accuracy

    def get_class_results(self, prediction):
        class_id = np.argmax(prediction)
        accuracy = prediction[0][class_id]
        return self.classes_names[class_id][0], accuracy



