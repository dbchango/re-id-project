
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def alexnet_model():
    model = Sequential()

    # 1st conv layer
    model.add(Conv2D(filters=96, input_shape=(32, 32, 3), kernel_size=(11, 11), strides=(4, 4), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 2nd conv layer
    model.add(Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # 3rd conv layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 4th conv layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 5th conv layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    # FC layers
    model.add(Flatten())

    # 1st FC layer
    model.add(Dense(4096, input_shape=(32, 32, 3, )))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # drop out layer
    model.add(Dropout(0.4))

    # 2nd FC layer
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # drop out layer
    model.add(Dropout(0.4))

    # 3rd FC layer
    model.add(Dense(1000))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # drop out layer
    model.add(Dropout(0.3))

    # output layer
    model.add(Dense(7))
    model.add(BatchNormalization())
    model.add(Activation('softmax'))

    model.summary()

    return model


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