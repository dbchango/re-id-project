import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, BatchNormalization, Dropout
from keras.regularizers import l2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

model = Sequential()

model.add(Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding='valid', input_shape=(256, 256, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(1, 1), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(5, 5), strides=(1, 1), padding='same', kernel_regularizer=l2(0.0005), activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.0005), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.0005), activation='relu'))
model.add(BatchNormalization())

model.add(Conv2D(filters=256, kernel_size=(2, 2), strides=(1, 1), padding='same', kernel_regularizer=l2(0.0005), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid'))

model.add(Flatten())

model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=4096, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=8, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

from processing import load_image_dataset

x_rgb_train, y_rgb_train = load_image_dataset('Datasets/espe/base/train', (256, 256), True)
x_rgb_test, y_rgb_test = load_image_dataset('Datasets/espe/base/test', (256, 256), True)
x_rgb_validation, y_rgb_validation = load_image_dataset('Datasets/espe/base/validation', (256, 256), True)

model.fit(x_rgb_train, y_rgb_train, batch_size=16, epochs=100, validation_split=0.2)