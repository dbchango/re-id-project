
from tensorflow.keras import layers
from tensorflow.keras.models import Model

if __name__ == '__main__':
    inputs = layers.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation='relu')(inputs)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_1_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation='relu', padding='same')(block_2_output)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation='relu')(block_3_output)
    x = layers.GlobalAvgPool2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)

    model = Model(inputs, outputs, name='toy_resnet')

    from tensorflow.keras.utils import to_categorical
    from keras import datasets
    from tensorflow.keras.optimizers import RMSprop

    (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    model.compile(optimizer=RMSprop(1e-3),
                  loss='categorical_crossentropy',
                  metrics=['acc'])

    model.fit(x_train, y_train,
              batch_size=64,
              epochs=1,
              validation_split=0.2)