
from keras.models import Sequential
import keras.layers
from keras.models import Model
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def vgg_net(input_shape=(64, 64, 1)):

    model = Sequential()

    # block
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    # block N° 6

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='relu'))

    return model


def mlp(input_shape=(64, 64, 1)):

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='relu'))

    return model


def a_p_branch(input_shape):

    inputs = Input(input_shape)
    x = Dense(100, activation='relu')(inputs)
    x = Dense(100, activation='relu')(x)


# def multi_input_net(input_shape=(64, 64, 1))


def generate_image_dataset(train_data_path, test_data_path, validation_path, target_size):
    gen = ImageDataGenerator()
    train_gen = gen.flow_from_directory(train_data_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    validation_gen = gen.flow_from_directory(validation_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    test_gen = gen.flow_from_directory(test_data_path, target_size=target_size, class_mode='categorical', shuffle=False, color_mode='grayscale')
    return train_gen, test_gen, validation_gen


def complete_image_dataset_loading(train_data_path, test_data_path, validation_path, target_size):
    from processing import load_image_dataset
    (rgb_train) = load_image_dataset('Datasets/espe/base/train', target_size, True)
    (rgb_test) = load_image_dataset('Datasets/espe/base/test', target_size, True)
    (rgb_validation) = load_image_dataset('Datasets/espe/base/validation', target_size, True)
    return rgb_train, rgb_test, rgb_validation

def data_generator(train_data_path, validation_data_path):
    im_width, im_height = 32, 32
    nb_train_samples = 8144
    nb_validation_samples = 5041
    epochs = 10
    batch_size = 32
    n_classes = 7

    datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2,
        rotation_range=5,
        horizontal_flip=True
    )

    train_gen = datagen.flow_from_directory(train_data_path, target_size=(im_width, im_height), class_mode='categorical', batch_size=batch_size)
    validation_gen = datagen.flow_from_directory(validation_data_path, target_size=(im_width, im_height), class_mode='categorical', batch_size=batch_size)

    return train_gen, validation_gen