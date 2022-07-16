

from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

# batch_size=200, epochs=30
def color_histogram_branch():
    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_shape=(512, ), activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    return model


def image_classification(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])
    return model
def  silhouette_cnn_model():
    cnn_model = keras.Sequential()
    cnn_model.add(keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(40,40,3)))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Dropout(0.3))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(500, activation='relu'))
    cnn_model.add(keras.layers.Dropout(0.4))
    cnn_model.add(keras.layers.Dense(7, activation='softmax'))
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, amsgrad=True)
    cnn_model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])
    return cnn_model

def cnn_branch():
    cnn_model = keras.Sequential()
    cnn_model.add(keras.layers.Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(40,40,3)))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    cnn_model.add(keras.layers.MaxPooling2D(pool_size=2))
    cnn_model.add(keras.layers.Dropout(0.3))
    cnn_model.add(keras.layers.Flatten())
    cnn_model.add(keras.layers.Dense(500, activation='relu'))
    cnn_model.add(keras.layers.Dropout(0.4))
    return cnn_model

def image_branch(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    return model

def multi_input_model():
    a =image_branch((40,40,3))
    c = cnn_branch()
    fussion = keras.layers.concatenate([a.output, c.output])
    d = keras.layers.Dense(800, activation='relu')(fussion)
    d = keras.layers.Dropout(0.8)(d)
    d = keras.layers.Dense(800, activation='relu')(d)
    d = keras.layers.Dropout(0.8)(d)
    d = keras.layers.Dense(7, activation='softmax')(d)
    model = keras.models.Model(inputs=[a.input, c.input], outputs=d)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.Adam(lr=0.0025, beta_1=0.9, beta_2=0.999, amsgrad=True)
    model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])