import tensorflow as tf
import pandas as pd
import numpy as np


class IdentificationModel:
    def __init__(self, model_path="models/own/saved_model/my_model.h5", classes_names_path="Datasets/classes_names.csv"):
        self.model = tf.keras.models.load_model(model_path)
        self.classes_names = pd.read_csv(classes_names_path, header=None)

    def identify(self, x):
        print(x.shape)
        prediction = self.model.predict(x)
        class_name = self.get_class_name(prediction)
        return class_name

    def get_class_name(self, prediction):
        class_id = np.argmax(prediction)
        return self.classes_names[class_id]



