
from keras.models import Sequential
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator


def vgg_net(input_shape=(64, 64, 1)):

    model = Sequential()

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


def image_branch():

    model = keras.Sequential()

    # block
    model.add(keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same',
                                  input_shape=(64, 64, 1)))
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

    return model


def lbp_histogram_branch():

    model = keras.Sequential()
    model.add(keras.layers.Dense(100, input_shape=(26, ), activation='relu'))
    model.add(keras.layers.Dense(4096, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(4096, activation='relu'))
    # model.add(keras.layers.Dropout(0.5))
    # model.add(keras.layers.Dense(7, activation='softmax'))

    return model


def silhouette_features_branch():
    model = keras.Sequential()
    model.add(keras.layers.Dense(4,  input_shape=(2, ), activation='relu'))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.8))
    model.add(keras.layers.Dense(32, activation='relu'))
    # model.add(keras.layers.Dropout(0.8))
    # model.add(keras.layers.Dense(7, activation='softmax'))

    return model


def multi_input_model():
    a = lbp_histogram_branch()
    b = silhouette_features_branch()
    c = image_branch()
    fussion = keras.layers.concatenate([a.output, b.output, c.output])
    d = keras.layers.Dense(4096, activation='relu')(fussion)
    d = keras.layers.Dropout(0.5)(d)
    d = keras.layers.Dense(4096, activation='relu')(d)
    d = keras.layers.Dropout(0.5)(d)
    d = keras.layers.Dense(7, activation='softmax')(d)
    model = keras.models.Model(inputs=[a.input, b.input, c.input], outputs=d)

    return model


def lbp_image(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(64, 64)))
    hp_units = hp.Int('units', min_value=64, max_value=1024, step=32)
    model.add(keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(keras.layers.Dense(7, activation='softmax'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), loss="categorical_crossentropy", metrics=['accuracy'])
    return model


def proto_model(hp):
    input = keras.Input(shape=(25,))
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    x = keras.layers.Dense(units=hp_units, activation='relu')(input)
    x = keras.layers.Dense(7, activation='relu')(x)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    model = keras.models.Model(inputs=input, outputs=x)
    model.compile(optimizer=keras.optimizers.Adam(lr=hp_learning_rate), loss="categorical_crossentropy", metrics=['accuracy'])
    return model


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

# def data_generator(train_data_path, validation_data_path):
#     im_width, im_height = 32, 32
#     nb_train_samples = 8144
#     nb_validation_samples = 5041
#     epochs = 10
#     batch_size = 32
#     n_classes = 7
#
#     datagen = ImageDataGenerator(
#         rescale=1./255,
#         zoom_range=0.2,
#         rotation_range=5,
#         horizontal_flip=True
#     )
#
#     train_gen = datagen.flow_from_directory(train_data_path, target_size=(im_width, im_height), class_mode='categorical', batch_size=batch_size)
#     validation_gen = datagen.flow_from_directory(validation_data_path, target_size=(im_width, im_height), class_mode='categorical', batch_size=batch_size)
#
#     return train_gen, validation_gen