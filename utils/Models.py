
from keras.models import Sequential
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator


def vgg_net(input_shape=(64, 64, 1)):
    model = keras.Sequential()
    # block
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    # block
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    # block
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    # block
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    # block
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    # block N° 6
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(7, activation='relu'))
    return model


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


def prototype_model_for_reid(input_shape, output_shape):
    a = image_branch(input_shape)
    b = image_branch(input_shape)
    fussion = keras.layers.concatenate([a.output, b.output])
    c = keras.layers.Dense(256, activation='softmax')(fussion)
    c = keras.layers.Dense(output_shape, activation='softmax')(c)
    model = keras.models.Model(inputs=[a.input, b.input], outputs=c)
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.RMSprop(lr=1e-3)
    model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])
    return model


def lbp_image_classification(input_shape):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', input_shape=input_shape))
    model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.RMSprop(lr=1e-3)
    model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])
    return model


def generate_image_dataset(train_data_path, test_data_path, validation_path, target_size):
    gen = ImageDataGenerator()
    train_gen = gen.flow_from_directory(train_data_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    validation_gen = gen.flow_from_directory(validation_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    test_gen = gen.flow_from_directory(test_data_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    return train_gen, test_gen, validation_gen


def complete_image_dataset_loading(train_data_path, test_data_path, validation_path, target_size):
    from processing import load_image_dataset
    (rgb_train) = load_image_dataset('Datasets/espe/base/training', target_size, True)
    (rgb_test) = load_image_dataset('Datasets/espe/base/test', target_size, True)
    (rgb_validation) = load_image_dataset('Datasets/espe/base/validation', target_size, True)
    return rgb_train, rgb_test, rgb_validation


def individual_feature_model(input_shape, output_shape):
    model = image_branch(input_shape)
    model.add(keras.layers.Dense(256, activation='relu'))
    model.add(keras.layers.Dense(output_shape, activation='softmax'))
    loss_function = tf.keras.losses.CategoricalCrossentropy()
    optimization_function = tf.keras.optimizers.RMSprop(lr=1e-3)
    model.compile(loss=loss_function, optimizer=optimization_function, metrics=['acc'])
    return model