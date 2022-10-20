import argparse
from utils.Models import lbp_image_classification, silhouette_classifier_model
import tensorflow as tf
import pandas as pd
from utils.processing import load_image_dataset, load_csv_data, slice_labels
import matplotlib.pyplot as plt


def load_lbp_images(path, target_size):
    x, y = load_image_dataset(path=path, size=target_size, gray_scale=True)
    return x, y


def load_lbp_histograms_dataset(path):
    data = load_csv_data(path)
    x, y = slice_labels(data)
    return x, y


def load_slt_dataset(path):
    data = pd.read_csv(path, header=None)[1:]
    x, y = slice_labels(data)
    return x,y


def main():
    size = (40, 40, 1)
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", default='Datasets/espe/lbp_histograms - copia', type=str, help="Dataset path, it could be divided in three parts: training, "
                                                       "validation and test. Ex: Datasets/path")
    parser.add_argument("-m", "--model", default='default', type=str, help="Type the model name that you need to training.")
    parser.add_argument("-mp", "--model_path", type=str, help="Path to save trained model.")
    parser.add_argument("-lp", "--logs_path", type=str, help="Path to save training logs.")
    parser.add_argument("-lbp", "--lbp_dataset", action='store_true', help="Use lbp dataset?.")
    parser.add_argument("-lbph", "--lbp_h_dataset", action='store_true', help="Use lbp histograms dataset?.")
    parser.add_argument("-sl", "--silhouette_dataset", action='store_true', help="Use lbp histograms dataset?.")
    args = parser.parse_args()

    print("[INFO] loading dataset ...")
    base_path = args.data
    train_path, validation_path, test_path = base_path + '/training', base_path + '/validation', base_path + '/test'

    if args.lbp_dataset:
        print("[INFO] loading lbp images dataset")
        x_rgb_train, y_rgb_train = load_lbp_images(train_path, (40, 40))
        x_rgb_validation, y_rgb_validation = load_lbp_images(validation_path, (40, 40))
        x_rgb_test, y_rgb_test = load_lbp_images(test_path, (40, 40))

    if args.lbp_h_dataset:
        print("[INFO] loading lbp histograms dataset")
        x_lbph_train, y_lbph_train = load_lbp_histograms_dataset(train_path)
        x_lbph_validation, y_lbph_validation = load_lbp_histograms_dataset(validation_path)
        x_lbph_test, y_lbph_test = load_lbp_histograms_dataset(test_path)

    if args.silhouette_dataset:
        print("[INFO] loading silhouette dataset")
        x_slt_train, y_slt_train = load_slt_dataset(train_path + '/training.csv')
        x_slt_validation, y_slt_validation = load_slt_dataset(validation_path + '/validation.csv')
        x_slt_test, y_slt_test = load_slt_dataset(test_path + '/test.csv')

    model = None
    if args.model == 'default':
        print("[INFO] loading and compiling model ... ")
        model = lbp_image_classification(size)
        print("[INFO] model fitting ... ")
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, min_delta=1e-2)
        callbacks_list = [stop_early, lr_reducer]
        history = model.fit(x=[x_rgb_train], y=y_rgb_train, batch_size=200, epochs=30, validation_data=[[x_rgb_validation], y_rgb_validation], callbacks=callbacks_list)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig('df_history_acc.png')
        print("[INFO] model evaluation")
        m_score, m_acc = model.evaluate(x=x_rgb_test, y=y_rgb_test, verbose=0)
        print("[INFO] model evaluation score: {}, accuracy: {}".format(m_score, m_acc))
        print("[INFO] saving model")
        model.save('models/own_models/saved_model/lbp_images_model.h5')
    elif args.model == 'silhouette_identification':
        print("[INFO] loading and compiling model ... ")
        model = silhouette_classifier_model()
        print("[INFO] model fitting ... ")
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)
        lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1,
                                                          min_delta=1e-2)
        callbacks_list = [stop_early, lr_reducer]
        history = model.fit(x=[x_slt_train], y=y_slt_train, validation_data=[[x_slt_validation], y_slt_validation],
                            batch_size=200, epochs=70, callbacks=callbacks_list)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.savefig('slt_history_acc.png')
        print("[INFO] model evaluation")
        m_score, m_acc = model.evaluate(x=x_slt_test, y=y_slt_test, verbose=1)
        print("[INFO] model evaluation score: {}, accuracy: {}".format(m_score, m_acc))
        print("[INFO] saving model")
        model.save('models/own_models/saved_model/slt_model.h5')

if __name__ == '__main__':
    main()
