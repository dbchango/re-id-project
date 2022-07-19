import os
import matplotlib.pyplot as plt

def cv_training(model=None, n_splits=3, x_data=[], y_data=None, path_to_save_results='', batch_size=150, epochs=20):
    """
    It will train a model using cross validation technique
    :param model: model that will be trained.
    :param n_splits: number of folds.
    :param x_data: list of X datasets, if you have a model with more than one input, you should pass all between brackets. Ex: [a, b]
    :param y_data: list of Y datasets, if you have a model with more than one input, you should pass only one labels dataset. Ex: a
    :param path_to_save_model: the trained model will be saved in this path. Ex: 'Datasets/own/model.h5'
    :return:
    """
    from sklearn.model_selection import KFold
    import tensorflow as tf
    from tensorflow.keras.callbacks import ReduceLROnPlateau

    loss_per_fold = []
    acc_per_fold = []
    kf = KFold(n_splits=n_splits, shuffle=True)
    fold_no = 1  # folds counter
    for train_index, test_index in kf.split(x_data[0]):
        # creating callbacks
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=8)
        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.1, patience=4, verbose=1, min_delta=1e-3)
        callbacks_list = [stop_early, reduce_lr]

        # fitting model
        history = model.fit(x=[i[train_index] for i in x_data], y=y_data[train_index], batch_size=batch_size, epochs=epochs, validation_split=0.2)
        path_to_save_model = os.path.join(path_to_save_results, f'model_{fold_no}.h5')
        path_to_save_acc_graph = os.path.join(path_to_save_results, f'acc_{fold_no}.png')
        path_to_save_loss_graph = os.path.join(path_to_save_results, f'loss_{fold_no}.png')

        # save loss graph
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path_to_save_loss_graph)
        plt.clf()

        # save acc graph
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('Model accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(['train', 'test'], loc='upper left')
        plt.savefig(path_to_save_acc_graph)
        plt.clf()

        # saving model
        model.save(path_to_save_model)

        # generate generalization metrics
        scores = model.evaluate(x=[j[test_index] for j in x_data], y=y_data[test_index], verbose=True)
        print(
            f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1] * 100}%')
        acc_per_fold.append(scores[1] * 100)
        loss_per_fold.append(scores[0])

        fold_no += 1