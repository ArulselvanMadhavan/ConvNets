import pickle as pickle
import numpy as np
import os


def load_CIFAR_batch(filename):
    """
    Loads the datasets
    :param filename: Filename to be read
    :return: Format of the matrix 10000 x 3072
    """
    with open(filename, 'rb') as f:
        datadict = pickle.load(f)
        X = datadict['data']
        Y = datadict['labels']
        X = np.array(X)
        Y = np.array(Y)
        return X, Y

def getCIFAR_as_32Pixels_Image(arr):
    """
    Returns the array as 32x32 pixel images
    :param arr:
    :return:
    """
    return arr.reshape(arr.shape[0], 3, 32, 32).transpose(0,2,3,1).astype("float")

def load_CIFAR_Dataset(CIFAR_HOME):
    """
    Iterate over all the batches and concatenate them into one matrix
    :param CIFAR_HOME: directory where the data is stored
    :return: Train data and test data
    """
    x_temp = []
    y_temp = []
    for b in range(1,6):
        f = os.path.join(CIFAR_HOME, 'data_batch_%d' % (b, ))
        X, Y = load_CIFAR_batch(f)
        x_temp.append(X)
        y_temp.append(Y)
    X_train = np.concatenate(x_temp)
    Y_train = np.concatenate(y_temp)
    del X, Y
    X_test, Y_test = load_CIFAR_batch(os.path.join(CIFAR_HOME, 'test_batch'))
    return X_train, Y_train, X_test, Y_test